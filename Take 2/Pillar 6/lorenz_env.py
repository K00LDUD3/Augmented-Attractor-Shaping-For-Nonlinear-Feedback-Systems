"""
Pillar 6 — GALI-Informed Hybrid PID-Residual RL Environment
=============================================================
Gymnasium-compatible environment for the 2-Coupled Lorenz system.

Architecture:
    PID (Kp=30) provides coarse homing on y1, y2
    RL agent provides residual nudges scaled by lambda
    caSSM encoder provides manifold-aware latent observations
    v4 GALI surrogate provides stability reward shaping (Log-Barrier)

Observation (7D):
    [xi_1, xi_2, xi_dot_1, xi_dot_2, u_pid_1, u_pid_2, log_gali_norm]

Action (2D):
    Continuous [-1, 1] -> scaled by lambda and added to PID output on y1, y2
"""

import torch  # ALWAYS first (DLL conflict)
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json

from train_cassm_encoder import SSMEncoder


# ---------------------------------------------------------------------------
# v4 GALI Surrogate Architecture (frozen, inference-only)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.net(x))


class SALISurrogate(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.downsample = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res3 = ResidualBlock(hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.downsample(x)
        x = self.res3(x)
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# PID Controller (mirrors Pillar 4 implementation exactly)
# ---------------------------------------------------------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, u_max=250.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_max = u_max
        self.integral = 0.0
        self.prev_error = 0.0

    def get_action(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return np.clip(u, -self.u_max, self.u_max)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# ---------------------------------------------------------------------------
# Coupled Lorenz ODE (mirrors Pillar 4 implementation exactly)
# ---------------------------------------------------------------------------
class CoupledLorenzDynamics:
    """RK4 integrator for the 2-coupled Lorenz system with control on y-channels."""

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, k=2.5, dt=0.005):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.k = k
        self.dt = dt

    def _derivatives(self, state, u):
        x1, y1, z1, x2, y2, z2 = state
        uy1, uy2 = u  # Control applied only to y variables

        dx1 = self.sigma * (y1 - x1) + self.k * (x2 - x1)
        dy1 = x1 * (self.rho - z1) - y1 + self.k * (y2 - y1) + uy1
        dz1 = x1 * y1 - self.beta * z1 + self.k * (z2 - z1)

        dx2 = self.sigma * (y2 - x2) + self.k * (x1 - x2)
        dy2 = x2 * (self.rho - z2) - y2 + self.k * (y1 - y2) + uy2
        dz2 = x2 * y2 - self.beta * z2 + self.k * (z1 - z2)

        return np.array([dx1, dy1, dz1, dx2, dy2, dz2])

    def step(self, state, u):
        k1 = self._derivatives(state, u)
        k2 = self._derivatives(state + 0.5 * self.dt * k1, u)
        k3 = self._derivatives(state + 0.5 * self.dt * k2, u)
        k4 = self._derivatives(state + self.dt * k3, u)
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------
class CoupledLorenzHybridEnv(gym.Env):
    """
    GALI-Informed Hybrid PID-Residual RL environment for chaos control.

    The agent provides small residual nudges on top of a PID baseline
    to steer the coupled Lorenz system toward the origin while navigating
    chaotic manifolds via the caSSM encoder and GALI surrogate.

    Observation (dynamic, depends on caSSM latent_dim):
        [xi_1..xi_L,                   # L-D caSSM latent position
         xi_dot_1..xi_dot_L,           # L-D latent velocity (finite diff)
         u_pid_1, u_pid_2,             # PID actions (normalised /u_max)
         log_gali_norm]                # GALI stability index (normalised)

    Action (2D):
        Continuous [-1, 1] applied to y1, y2 channels, scaled by lambda.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate_path: str,
        pid_gains_path: str = None,
        # PID defaults (Kp=30 baseline)
        Kp: float = 30.0,
        Ki: float = 0.0286,
        Kd: float = 0.0,
        # Environment parameters
        dt: float = 0.005,
        episode_length: int = 4000,
        rl_lambda: float = 0.1,
        u_max: float = 250.0,
        freq_ratio: int = 1,
        # Reward coefficients
        reward_distance_coeff: float = 1.0,
        reward_effort_coeff: float = 0.01,
        reward_gali_coeff: float = 0.05,
        reward_convergence_bonus: float = 10.0,
        reward_sparsity_coeff: float = 0.5,
        reward_action_deriv_coeff: float = 1.0,
        reward_surf_coeff: float = 2.0,
        convergence_threshold: float = 0.5,
        attractor_radius: float = 30.0,
        # Physics parameters
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        k_coupling: float = 2.5,
        # State bounds
        state_bound: float = 40.0,
        init_bound: float = 40.0,
        use_cassm: bool = True,
    ):
        super().__init__()

        self.use_cassm = use_cassm

        # --- Store all parameters for logging ---
        self.dt = dt
        self.episode_length = episode_length
        self.rl_lambda = rl_lambda
        self.u_max = u_max
        self.freq_ratio = freq_ratio
        self.state_bound = state_bound
        self.init_bound = init_bound

        # Reward shaping coefficients
        self.reward_distance_coeff = reward_distance_coeff
        self.reward_effort_coeff = reward_effort_coeff
        self.reward_gali_coeff = reward_gali_coeff
        self.reward_convergence_bonus = reward_convergence_bonus
        self.reward_sparsity_coeff = reward_sparsity_coeff
        self.reward_action_deriv_coeff = reward_action_deriv_coeff
        self.reward_surf_coeff = reward_surf_coeff
        self.convergence_threshold = convergence_threshold
        self.attractor_radius = attractor_radius

        # Target state
        self.target = np.zeros(6)

        # --- Load PID gains ---
        if pid_gains_path and os.path.exists(pid_gains_path):
            with open(pid_gains_path, 'r') as f:
                gains = json.load(f)
            Kp, Ki, Kd = gains['Kp'], gains['Ki'], gains['Kd']

        self.pid_gains = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
        self.pid1 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        self.pid2 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)

        # --- Physics ---
        self.dynamics = CoupledLorenzDynamics(
            sigma=sigma, rho=rho, beta=beta, k=k_coupling, dt=dt
        )

        # --- GALI Surrogate (frozen) ---
        if os.environ.get("FORCE_CPU_SURROGATE", "0") == "1":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_surrogate(surrogate_path)
        if self.use_cassm:
            self._load_cassm()
        else:
            # 6c-compatible: no caSSM encoder, obs = [state(6), u_pid(2)/u_max, log_gali_norm(1)]
            self.latent_dim = 0
            self.encoder = None

        # --- Gymnasium spaces ---
        if self.use_cassm:
            obs_dim = 2 * self.latent_dim + 2 + 1
        else:
            obs_dim = 6 + 2 + 1  # state + u_pid + gali
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: residual nudge in [-1, 1] for y1, y2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Internal state ---
        self.state = None
        self.step_count = 0
        self.current_rl_action = np.zeros(2)
        self.last_rl_action = np.zeros(2)
        self.last_xi = np.zeros(self.latent_dim)
        self.current_xi_dot = np.zeros(self.latent_dim)  # For surfing reward

        # --- Per-episode accumulators (for metric logging) ---
        self.episode_total_effort = 0.0
        self.episode_total_rl_effort = 0.0
        self.episode_itwae = 0.0
        self.episode_max_overshoot = 0.0
        self.episode_convergence_step = None  # First step where ||state|| < threshold

    def _load_surrogate(self, path):
        """Load the frozen v4 GALI surrogate for inference."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.surrogate = SALISurrogate(input_dim=6, hidden_dim=512).to(self.device)
        self.surrogate.load_state_dict(checkpoint['model_state_dict'])
        self.surrogate.eval()
        for p in self.surrogate.parameters():
            p.requires_grad = False

        self.sali_min = checkpoint['sali_min']
        self.sali_max = checkpoint['sali_max']

    def _load_cassm(self):
        """Load the frozen caSSM encoder."""
        batch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "6d_cassm_batch")
        
        # Find latest cassm_encoder.pth
        latest_model = None
        latest_time = 0
        if os.path.exists(batch_dir):
            for run_dir in os.listdir(batch_dir):
                model_path = os.path.join(batch_dir, run_dir, "data", "cassm_encoder.pth")
                if os.path.exists(model_path):
                    mtime = os.path.getmtime(model_path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_model = model_path
        
        if latest_model is None:
            raise FileNotFoundError(f"Could not find cassm_encoder.pth in {batch_dir}")
            
        checkpoint = torch.load(latest_model, map_location=self.device, weights_only=False)
        params = checkpoint.get("params", {})
        latent_dim = params.get("latent_dim", 2)
        hidden_dim = params.get("hidden_dim", 256)
        
        self.latent_dim = latent_dim
        self.encoder = SSMEncoder(input_dim=6, hidden_dim=hidden_dim, latent_dim=latent_dim).to(self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _query_gali(self, state_6d):
        """Query the surrogate for log10(GALI2) and return normalised value."""
        with torch.no_grad():
            x = torch.tensor(state_6d / self.state_bound, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_norm = self.surrogate(x).cpu().item()
        # Denormalise: surrogate outputs normalised [0,1] -> log10(GALI)
        log_gali = pred_norm * (self.sali_max - self.sali_min + 1e-8) + self.sali_min
        # Normalise to approximately [-1, 1] for observation
        # log_gali ranges from ~-11.5 (deep chaos) to ~0.0 (stable)
        log_gali_norm = log_gali / abs(self.sali_min) if self.sali_min != 0 else 0.0
        return log_gali, log_gali_norm

    def _build_obs(self, state, u_pid, log_gali_norm):
        """Construct the dynamic observation vector."""
        if self.use_cassm:
            # caSSM latent-space observation (6d variant)
            with torch.no_grad():
                x = torch.tensor(state / self.state_bound, dtype=torch.float32).unsqueeze(0).to(self.device)
                xi = self.encoder(x).cpu().numpy().flatten()
            xi_dot = (xi - self.last_xi) / self.dt
            self.last_xi = xi.copy()
            self.current_xi_dot = xi_dot.copy()
            return np.concatenate([
                xi,
                xi_dot,
                u_pid / self.u_max,
                np.array([log_gali_norm])
            ]).astype(np.float32)
        else:
            # 6c-compatible: raw state + pid + gali
            return np.concatenate([
                state / self.state_bound,
                u_pid / self.u_max,
                np.array([log_gali_norm])
            ]).astype(np.float32)

    def _compute_reward(self, state, next_state, u_total, log_gali_next):
        """
        Three-component reward (Delta 15 — simplified):
            1. Distance to target (normalised)
            2. Absolute effort penalty (L1)
            3. Convergence bonus
        """
        dist = np.linalg.norm(next_state - self.target)
        max_dist = self.state_bound * np.sqrt(6)  # ~98 for bound=40

        # 1. Distance: maps [0, max_dist] -> [-1, 0], scaled by coeff
        r_dist = -self.reward_distance_coeff * (dist / max_dist)

        # 2. Effort penalty (absolute L1):
        u_norm = np.sum(np.abs(u_total)) / (len(u_total) * self.u_max)
        r_effort = -self.reward_effort_coeff * u_norm

        # 3. Convergence bonus (per-step when within threshold)
        r_conv = self.reward_convergence_bonus if dist < self.convergence_threshold else 0.0

        return r_dist + r_effort + r_conv

    def reset(self, seed=None, options=None):
        """Reset the environment with a random initial condition from [-init_bound, init_bound]^6."""
        super().reset(seed=seed)

        if options and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float64)
        else:
            self.state = self.np_random.uniform(
                low=-self.init_bound, high=self.init_bound, size=6
            )

        # Reset controllers
        self.pid1.reset()
        self.pid2.reset()
        self.step_count = 0
        self.current_rl_action = np.zeros(2)
        self.last_rl_action = np.zeros(2)

        # Reset accumulators
        self.episode_total_effort = 0.0
        self.episode_total_rl_effort = 0.0
        self.episode_itwae = 0.0
        self.episode_max_overshoot = 0.0
        self.episode_convergence_step = None

        # Build initial observation
        u_pid = np.array([
            self.pid1.get_action(self.target[1] - self.state[1]),
            self.pid2.get_action(self.target[4] - self.state[4])
        ])
        # Undo the PID side-effects from the observation query
        # (PID state was just set, we actually want a "clean" observation)
        self.pid1.reset()
        self.pid2.reset()

        # Initialize last_xi so initial xi_dot is 0
        if self.use_cassm:
            with torch.no_grad():
                x0 = torch.tensor(self.state / self.state_bound, dtype=torch.float32).unsqueeze(0).to(self.device)
                self.last_xi = self.encoder(x0).cpu().numpy().flatten()
        else:
            self.last_xi = np.zeros(0)

        _, log_gali_norm = self._query_gali(self.state)
        obs = self._build_obs(self.state, np.zeros(2), log_gali_norm)

        info = {"initial_state": self.state.copy()}
        return obs, info

    def step(self, action):
        """
        Execute one timestep:
            1. PID computes baseline control
            2. RL residual is applied (respecting freq_ratio)
            3. RK4 integration
            4. GALI query on new state
            5. Reward computation
        """
        action = np.clip(action, -1.0, 1.0)

        # 1. PID computes baseline from current error
        e1 = self.target[1] - self.state[1]
        e2 = self.target[4] - self.state[4]
        u_pid_1 = self.pid1.get_action(e1)
        u_pid_2 = self.pid2.get_action(e2)
        u_pid = np.array([u_pid_1, u_pid_2])

        # 2. RL residual (respects frequency ratio)
        self.last_rl_action = self.current_rl_action.copy()
        if self.step_count % self.freq_ratio == 0:
            self.current_rl_action = action
        u_rl = self.rl_lambda * self.current_rl_action

        # 3. Combined control with saturation
        u_total = np.clip(u_pid + u_rl, -self.u_max, self.u_max)

        # 4. RK4 integration
        next_state = self.dynamics.step(self.state, u_total)

        # 5. GALI query on next state
        log_gali, log_gali_norm = self._query_gali(next_state)

        # 6. Reward
        reward = self._compute_reward(self.state, next_state, u_total, log_gali)

        # --- Accumulate episode metrics ---
        self.step_count += 1
        time_val = self.step_count * self.dt
        error_norm = np.linalg.norm(next_state - self.target)

        self.episode_total_effort += np.sum(np.abs(u_total))
        self.episode_total_rl_effort += np.sum(np.abs(u_rl))
        self.episode_itwae += time_val * error_norm * self.dt

        if error_norm > self.episode_max_overshoot:
            self.episode_max_overshoot = error_norm

        if self.episode_convergence_step is None and error_norm < self.convergence_threshold:
            self.episode_convergence_step = self.step_count

        # --- State transition ---
        self.state = next_state

        # --- Termination ---
        truncated = self.step_count >= self.episode_length
        # Divergence check: if any state variable exceeds 500, terminate
        terminated = bool(np.any(np.abs(self.state) > 500.0))

        # --- Build observation ---
        obs = self._build_obs(self.state, u_pid, log_gali_norm)

        info = {
            "u_pid": u_pid.copy(),
            "u_rl": u_rl.copy(),
            "u_total": u_total.copy(),
            "log_gali": log_gali,
            "error_norm": error_norm,
            "step": self.step_count,
        }

        # On episode end, include summary metrics
        if truncated or terminated:
            info["episode_metrics"] = {
                "total_effort": self.episode_total_effort,
                "total_rl_effort": self.episode_total_rl_effort,
                "itwae": self.episode_itwae,
                "max_overshoot": self.episode_max_overshoot,
                "final_error": error_norm,
                "convergence_step": self.episode_convergence_step,
                "convergence_time": (
                    self.episode_convergence_step * self.dt
                    if self.episode_convergence_step else None
                ),
                "diverged": terminated,
            }

        return obs, reward, terminated, truncated, info

    def get_params_dict(self):
        """Return a dictionary of all environment parameters for experiment logging."""
        return {
            "env_type": "CoupledLorenzHybridEnv",
            "observation_dim": self.observation_space.shape[0],
            "action_dim": 2,
            "latent_dim": self.latent_dim,
            "episode_length": self.episode_length,
            "dt": self.dt,
            "freq_ratio": self.freq_ratio,
            "rl_lambda": self.rl_lambda,
            "u_max": self.u_max,
            "state_bound": self.state_bound,
            "init_bound": self.init_bound,
            "attractor_radius": self.attractor_radius,
            "pid_gains": self.pid_gains,
            "reward_coefficients": {
                "distance": self.reward_distance_coeff,
                "effort": self.reward_effort_coeff,
                "gali": self.reward_gali_coeff,
                "convergence_bonus": self.reward_convergence_bonus,
                "sparsity": self.reward_sparsity_coeff,
                "action_deriv": self.reward_action_deriv_coeff,
                "surf": self.reward_surf_coeff,
                "convergence_threshold": self.convergence_threshold,
            },
            "physics": {
                "sigma": self.dynamics.sigma,
                "rho": self.dynamics.rho,
                "beta": self.dynamics.beta,
                "k_coupling": self.dynamics.k,
            },
            "surrogate": {
                "sali_min": float(self.sali_min),
                "sali_max": float(self.sali_max),
            },
        }


# ---------------------------------------------------------------------------
# Sanity Check: Verify zero-residual reproduces pure PID trajectory
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    SURROGATE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "Pillar 5", "experiments",
        "2026-04-10_06-03-45_29331e02", "data", "checkpoints", "ckpt_best_model.pth"
    )
    PID_GAINS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "Pillar 4", "ideal_pid_gains_kp30.json"
    )

    print("=" * 60)
    print("SANITY CHECK: Zero-Residual vs Pure PID Comparison")
    print("=" * 60)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    env = CoupledLorenzHybridEnv(
        surrogate_path=SURROGATE_PATH,
        pid_gains_path=PID_GAINS_PATH,
        rl_lambda=0.1,
        freq_ratio=1,
    )

    # --- Run the environment with zero RL action ---
    init_state = np.array([12.0, 15.0, 35.0, -10.0, -20.0, 40.0])
    obs, info = env.reset(options={"initial_state": init_state.copy()})
    env_states = [init_state.copy()]

    total_reward = 0.0
    for step in range(4000):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        env_states.append(env.state.copy())
        total_reward += reward
        if terminated or truncated:
            break

    env_states = np.array(env_states)

    # --- Run pure PID (Pillar 4 style) ---
    from lorenz_env import CoupledLorenzDynamics, PIDController
    dynamics = CoupledLorenzDynamics()
    pid1 = PIDController(env.pid_gains['Kp'], env.pid_gains['Ki'], env.pid_gains['Kd'], 0.005, u_max=250.0)
    pid2 = PIDController(env.pid_gains['Kp'], env.pid_gains['Ki'], env.pid_gains['Kd'], 0.005, u_max=250.0)

    state = init_state.copy()
    pid_states = [state.copy()]
    for step in range(4000):
        e1 = 0.0 - state[1]
        e2 = 0.0 - state[4]
        u1 = pid1.get_action(e1)
        u2 = pid2.get_action(e2)
        state = dynamics.step(state, [u1, u2])
        pid_states.append(state.copy())
    pid_states = np.array(pid_states)

    # --- Compare ---
    max_diff = np.max(np.abs(env_states - pid_states))
    print(f"\nMax state difference (env vs pure PID): {max_diff:.6e}")
    print(f"Env final state:  {env_states[-1]}")
    print(f"PID final state:  {pid_states[-1]}")
    print(f"Total reward (zero-RL): {total_reward:.4f}")

    if "episode_metrics" in info:
        m = info["episode_metrics"]
        print(f"\nEpisode Metrics:")
        print(f"  Total Effort:      {m['total_effort']:.2f}")
        print(f"  Total RL Effort:   {m['total_rl_effort']:.2f}")
        print(f"  ITWAE:             {m['itwae']:.2f}")
        print(f"  Max Overshoot:     {m['max_overshoot']:.4f}")
        print(f"  Final Error:       {m['final_error']:.6f}")
        print(f"  Convergence Step:  {m['convergence_step']}")
        print(f"  Diverged:          {m['diverged']}")

    if max_diff < 1e-6:
        print("\n[PASS] Environment reproduces pure PID perfectly.")
    else:
        print(f"\n[WARN] Deviation detected: {max_diff:.2e}")
        print("       This is expected if PID reset logic differs slightly.")
