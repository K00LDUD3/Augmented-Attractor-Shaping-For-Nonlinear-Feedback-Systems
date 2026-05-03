"""
Pillar 6C — SAC Training Loop (GALI in Observation + Reward)
==============================================================
Trains a Soft Actor-Critic agent to provide residual nudges on top of a
Kp=30 PID baseline, using the v4 GALI surrogate for both observation
enrichment and reward shaping.

Experiment tracking follows the /experiment-tracker-and-logger-integration
workflow. Batch folder: 6c_gali_both_batch/

Usage:
    python train_sac.py --episodes 500 --variant 6c
    python train_sac.py --episodes 500 --variant 6a   # GALI obs only
    python train_sac.py --episodes 500 --variant 6b   # GALI reward only
"""

TEST_RUN = True

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # ALWAYS first (DLL conflict)
import numpy as np
import sys
import json
import time
import argparse
from datetime import datetime

from ExperimentTracker import ExperimentTracker
from Logger import Logger

from lorenz_env import CoupledLorenzHybridEnv
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PILLAR_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(PILLAR_DIR)

SURROGATE_PATH = os.path.join(
    PROJECT_ROOT, "Pillar 5", "experiments",
    "2026-04-10_06-03-45_29331e02", "data", "checkpoints", "ckpt_best_model.pth"
)
PID_GAINS_PATH = os.path.join(
    PROJECT_ROOT, "Pillar 4", "ideal_pid_gains_kp30.json"
)


# ---------------------------------------------------------------------------
# Default Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # --- Variant ---
    "variant": "6c",            # 6a = GALI obs only, 6b = GALI reward only, 6c = both

    # --- Training ---
    "total_episodes": 1000,
    "episode_length": 2000,     # Delta 13.3: 10s at dt=0.005. Transient + meaningful post-convergence.
    "warmup_steps": 8_000,      # Delta 13.3: Adjusted proportionally
    "batch_size": 256,
    "eval_interval": 10,        # Evaluate every N episodes
    "eval_episodes": 5,         # Number of eval episodes
    "checkpoint_interval": 50,  # Save checkpoint every N episodes

    # --- SAC ---
    "hidden_dim": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "init_alpha": 0.2,
    "replay_buffer_size": 1_000_000,

    # --- Environment ---
    "dt": 0.005,
    "rl_lambda": 50.0,          # Delta 13.1: 20% of u_max — doubled authority for meaningful effort reduction
    "u_max": 250.0,
    "freq_ratio": 10,           # DEC 7: Multi-timescale. PID at 500Hz, RL at 50Hz.
    "state_bound": 40.0,
    "init_bound": 25.0,         # Delta 13: Train within attractor basin. Eval uses 40.
    "attractor_radius": 30.0,   # Delta 13: Gate radius for surfing reward

    # --- Reward (all normalised to ~[-1, +1] per step) ---
    "reward_distance_coeff": 5.0,   # INCREASED: agent needs to care about tracking error.
    "reward_effort_coeff": 10.0,    # Delta 15: moderate effort nudge in simplified 3-component reward
    "reward_gali_coeff": 0.1,       # r_gali = coeff * gali_norm -> [0, 0.1]
    "reward_convergence_bonus": 1.0, # r_conv = bonus when dist < threshold -> {0, 1.0}
    "reward_sparsity_coeff": 0.1,    # DECREASED: Agent was too scared to act, letting error blow up.
    "reward_action_deriv_coeff": 0.1, # Delta 13.1: DECREASED from 1.0 — was creating startup barrier preventing agent from acting
    "reward_surf_coeff": 0.3,        # Delta 13: Manifold surfing penalty on ||xi_dot||^2
    "convergence_threshold": 0.5,
}


def get_variant_config(variant):
    """Return reward/obs configuration for each variant."""
    configs = {
        "6a": {"include_gali_obs": True,  "reward_gali_coeff": 0.0,  "label": "GALI Obs Only"},
        "6b": {"include_gali_obs": False, "reward_gali_coeff": 0.05, "label": "GALI Reward Only"},
        "6c": {"include_gali_obs": True,  "reward_gali_coeff": 0.05, "label": "GALI Both (Obs+Reward)"},
    }
    return configs.get(variant, configs["6c"])


def generate_lhs_ics(n_samples, dim=6, bounds=(-40.0, 40.0), seed=42):
    """
    Generate a fixed set of initial conditions on the SURFACE of the 6D hypercube
    using Latin Hypercube Sampling for the interior coordinates.
    """
    from scipy.stats import qmc
    import numpy as np
    
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # [0, 1]^d
    
    # Snap each sample to a random face (surface of the hypercube)
    np.random.seed(seed)
    for i in range(n_samples):
        face_dim = np.random.randint(0, dim)
        face_side = np.random.randint(0, 2)
        unit_samples[i, face_dim] = float(face_side)
        
    lo, hi = bounds
    ics = lo + unit_samples * (hi - lo)          # scale to [lo, hi]^d
    return ics


def create_env(params, variant_config, use_cassm=True):
    """Instantiate the hybrid environment with variant-specific config."""
    gali_coeff = variant_config["reward_gali_coeff"]

    env = CoupledLorenzHybridEnv(
        surrogate_path=SURROGATE_PATH,
        pid_gains_path=PID_GAINS_PATH,
        dt=params["dt"],
        episode_length=params["episode_length"],
        rl_lambda=params["rl_lambda"],
        u_max=params["u_max"],
        freq_ratio=params["freq_ratio"],
        reward_distance_coeff=params["reward_distance_coeff"],
        reward_effort_coeff=params["reward_effort_coeff"],
        reward_gali_coeff=gali_coeff,
        reward_convergence_bonus=params["reward_convergence_bonus"],
        reward_surf_coeff=params["reward_surf_coeff"],
        convergence_threshold=params["convergence_threshold"],
        attractor_radius=params["attractor_radius"],
        state_bound=params["state_bound"],
        init_bound=params["init_bound"],
        use_cassm=use_cassm,
    )

    # For variant 6b: mask GALI out of the observation
    env._include_gali_obs = variant_config["include_gali_obs"]

    # Override _build_obs if GALI is excluded from observation
    if not variant_config["include_gali_obs"]:
        original_build = env._build_obs
        def masked_build(state, u_pid, log_gali_norm):
            obs = original_build(state, u_pid, log_gali_norm)
            obs[-1] = 0.0  # Zero out the GALI dimension (always last element)
            return obs
        env._build_obs = masked_build

    return env


def evaluate(env, agent, eval_ics):
    """
    Run deterministic evaluation on a FIXED set of initial conditions.
    Returns averaged metrics across all ICs.
    """
    metrics = {
        "eval_reward": [],
        "eval_final_error": [],
        "eval_total_effort": [],
        "eval_rl_effort": [],
        "eval_itwae": [],
        "eval_convergence_step": [],
    }

    for ic in eval_ics:
        obs, _ = env.reset(options={"initial_state": ic})
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        metrics["eval_reward"].append(episode_reward)

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            metrics["eval_final_error"].append(m["final_error"])
            metrics["eval_total_effort"].append(m["total_effort"])
            metrics["eval_rl_effort"].append(m["total_rl_effort"])
            metrics["eval_itwae"].append(m["itwae"])
            metrics["eval_convergence_step"].append(
                m["convergence_step"] if m["convergence_step"] else env.episode_length
            )

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# PID Baseline Stress Test (parallelized across CPU cores)
# ---------------------------------------------------------------------------
def _baseline_worker(args):
    """
    Worker function for parallel PID baseline.
    Must be top-level (not nested) so it's picklable by multiprocessing.
    Each worker creates its own env instance with its own surrogate copy.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["FORCE_CPU_SURROGATE"] = "1"  # Bypass torch CUDA init entirely


    ic_batch, params, variant_config = args

    # Create a fresh env in this process (loads its own surrogate)
    env = create_env(params, variant_config)
    zero_action = np.zeros(env.action_space.shape[0])
    results = []

    for idx, ic in ic_batch:
        obs, info = env.reset(options={"initial_state": ic})
        done = False
        ep_reward = 0.0

        while not done:
            obs, reward, terminated, truncated, step_info = env.step(zero_action)
            ep_reward += reward
            done = terminated or truncated

        m = step_info.get("episode_metrics", {})
        result = {
            "ic_index": idx,
            "initial_state": ic.tolist(),
            "reward": float(ep_reward),
            "total_effort": float(m.get("total_effort", 0)),
            "final_error": float(m.get("final_error", 0)),
            "itwae": float(m.get("itwae", 0)),
            "convergence_step": m.get("convergence_step", None),
            "max_overshoot": float(m.get("max_overshoot", 0)),
            "diverged": m.get("diverged", False),
        }
        results.append(result)

    return results


def run_pid_baseline_stress_test(baseline_ics, params, variant_config, logger, run):
    """
    Run the environment with zero RL action over fixed LHS ICs.
    Uses multiprocessing for parallelism — each worker creates its own env.
    Results are logged and saved to the experiment directory.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    n_episodes = len(baseline_ics)
    n_workers = min(multiprocessing.cpu_count(), 8)
    # n_workers = max(multiprocessing.cpu_count()-1,1)

    logger.log(f"=" * 60, tag="DEF", level="INFO")
    logger.log(f"PID BASELINE STRESS TEST ({n_episodes} ICs, {n_workers} workers)", tag="DEF", level="INFO")
    logger.log(f"=" * 60, tag="DEF", level="INFO")

    start_t = time.time()

    # Split (index, ic) pairs across workers
    indexed_ics = list(enumerate(baseline_ics))
    chunk_size = max(1, len(indexed_ics) // n_workers)
    chunks = [indexed_ics[i:i + chunk_size] for i in range(0, len(indexed_ics), chunk_size)]

    worker_args = [(chunk, params, variant_config) for chunk in chunks]

    # --- Run in parallel ---
    results = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for batch_results in pool.map(_baseline_worker, worker_args):
                results.extend(batch_results)
    except Exception as e:
        logger.log(f"Parallel baseline failed ({e}), falling back to sequential", tag="DEF", level="WARNING")
        # Fallback: run sequentially
        results = _baseline_worker((indexed_ics, params, variant_config))

    # Sort by index so results are deterministic in ordering
    results.sort(key=lambda r: r["ic_index"])

    elapsed = time.time() - start_t

    # --- Aggregate stats ---
    efforts = [r["total_effort"] for r in results]
    errors  = [r["final_error"] for r in results]
    itwae   = [r["itwae"] for r in results]
    rewards = [r["reward"] for r in results]
    conv_steps = [r["convergence_step"] for r in results if r["convergence_step"] is not None]
    n_diverged = sum(1 for r in results if r["diverged"])

    baseline_summary = {
        "n_episodes": n_episodes,
        "n_workers": n_workers,
        "wall_time_sec": round(elapsed, 1),
        "n_diverged": n_diverged,
        "effort":  {"mean": float(np.mean(efforts)), "std": float(np.std(efforts)),
                    "min": float(np.min(efforts)),   "max": float(np.max(efforts)),
                    "median": float(np.median(efforts))},
        "error":   {"mean": float(np.mean(errors)),  "std": float(np.std(errors)),
                    "min": float(np.min(errors)),    "max": float(np.max(errors)),
                    "median": float(np.median(errors))},
        "itwae":   {"mean": float(np.mean(itwae)),   "std": float(np.std(itwae)),
                    "min": float(np.min(itwae)),     "max": float(np.max(itwae)),
                    "median": float(np.median(itwae))},
        "reward":  {"mean": float(np.mean(rewards)),  "median": float(np.median(rewards))},
        "convergence_step": {"mean": float(np.mean(conv_steps)) if conv_steps else None,
                             "median": float(np.median(conv_steps)) if conv_steps else None},
    }

    # --- Log ---
    logger.log(f"  Completed in {elapsed:.1f}s ({n_workers} workers)", tag="DEF", level="INFO")
    logger.log(f"  Diverged:      {n_diverged}/{n_episodes}", tag="DEF", level="INFO")
    logger.log(f"  Effort:        mean={baseline_summary['effort']['mean']:.1f}  "
               f"std={baseline_summary['effort']['std']:.1f}  "
               f"median={baseline_summary['effort']['median']:.1f}  "
               f"[{baseline_summary['effort']['min']:.1f}, {baseline_summary['effort']['max']:.1f}]",
               tag="DEF", level="INFO")
    logger.log(f"  Final Error:   mean={baseline_summary['error']['mean']:.6f}  "
               f"std={baseline_summary['error']['std']:.6f}  "
               f"median={baseline_summary['error']['median']:.6f}",
               tag="DEF", level="INFO")
    logger.log(f"  ITWAE:         mean={baseline_summary['itwae']['mean']:.2f}  "
               f"std={baseline_summary['itwae']['std']:.2f}  "
               f"median={baseline_summary['itwae']['median']:.2f}",
               tag="DEF", level="INFO")
    logger.log(f"  Reward:        mean={baseline_summary['reward']['mean']:.1f}  "
               f"median={baseline_summary['reward']['median']:.1f}",
               tag="DEF", level="INFO")
    if conv_steps:
        logger.log(f"  Conv Step:     mean={baseline_summary['convergence_step']['mean']:.0f}  "
                   f"median={baseline_summary['convergence_step']['median']:.0f}",
                   tag="DEF", level="INFO")
    logger.log(f"=" * 60, tag="DEF", level="INFO")

    # --- Save ---
    with open(run.get_path("logs/pid_baseline.json"), "w") as f:
        json.dump({"summary": baseline_summary, "episodes": results}, f, indent=2, cls=NumpyEncoder)
    logger.log("PID baseline results saved to logs/pid_baseline.json", tag="PAT", level="INFO")

    return baseline_summary


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def train(params):

    variant = params["variant"]
    variant_config = get_variant_config(variant)
    batch_folder = f"{variant}_gali_{'obs' if variant == '6a' else 'reward' if variant == '6b' else 'both'}_batch"

    # --- Experiment Tracker ---
    BATCH_DIR = os.path.join(PILLAR_DIR, batch_folder)
    tracker = ExperimentTracker(BATCH_DIR)

    all_params = {
        **params,
        "variant_config": variant_config,
        "surrogate_path": SURROGATE_PATH,
        "pid_gains_path": PID_GAINS_PATH,
        "baseline_run_path": params.get("baseline_run_path", None),
    }
    NOTES = f"SAC Hybrid PID-RL | Variant {variant.upper()}: {variant_config['label']}"
    if TEST_RUN:
        NOTES +=  f"{TEST_RUN=}"
    run = tracker.create_run(
        params=all_params,
        notes=NOTES
    )

    # --- Logger ---
    logger = Logger(
        f=run.get_path("logs/run.log"),
        sesh_file=run.get_path("logs/.sesh_num")
    )
    print(f"Run powershell to track log file live: \n\n\twatchlog \"{logger.f}\"\n")
    print(f"OR \n Linux to track log file live: \n\n\ttail -F -n +1 \"{logger.f}\"\n")

    # Copy this script for reproducibility
    run.copy_file(os.path.abspath(__file__), "configs/")

    logger.start_session()
    try:
        logger.log(f"Run UID: {run.uid}", tag="BEG", level="INFO")
        logger.log(f"Variant: {variant.upper()} - {variant_config['label']}", tag="DEF", level="INFO")
        logger.log(f"Params: {json.dumps(all_params, indent=2, default=str)}", tag="DEF", level="INFO")

        # --- Environment ---
        env = create_env(params, variant_config)
        eval_env = create_env(params, variant_config)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]        # 2

        logger.log(f"Obs dim: {obs_dim}, Act dim: {act_dim}", tag="DEF", level="INFO")
        logger.log(f"Environment params: {json.dumps(env.get_params_dict(), indent=2, default=str)}", tag="DEF", level="INFO")

        # --- Agent ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = SACAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=params["hidden_dim"],
            gamma=params["gamma"],
            tau=params["tau"],
            actor_lr=params["actor_lr"],
            critic_lr=params["critic_lr"],
            alpha_lr=params["alpha_lr"],
            init_alpha=params["init_alpha"],
            device=device,
        )
        logger.log(f"Agent params: {json.dumps(agent.get_params_dict(), indent=2, default=str)}", tag="DEF", level="INFO")
        logger.log(f"Device: {device}", tag="DEF", level="INFO")

        # --- Replay Buffer ---
        buffer = ReplayBuffer(obs_dim, act_dim, max_size=params["replay_buffer_size"])

        # --- Load Pre-computed ICs & Baseline ---
        baseline_run_path = params["baseline_run_path"]
        if baseline_run_path is None:
            raise ValueError(
                "--baseline-run is required. Run run_baseline_stress_test.py first, "
                "then pass the run directory path."
            )

        ic_path = os.path.join(baseline_run_path, "logs", "master_fixed_ics.json")
        baseline_path = os.path.join(baseline_run_path, "logs", "pid_baseline.json")

        if not os.path.exists(ic_path):
            raise FileNotFoundError(f"IC file not found: {ic_path}")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        # Load ICs
        with open(ic_path, "r") as f:
            ic_data = json.load(f)
        master_ics = np.array(ic_data["master_ics"])

        # Load baseline
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        baseline = baseline_data["summary"]

        n_eval_ics = params["eval_episodes"]
        eval_ics = master_ics[:n_eval_ics]

        # Save a copy of the ICs into this run for traceability
        with open(run.get_path("logs/master_fixed_ics.json"), "w") as f:
            json.dump(ic_data, f, indent=2)

        # --- Log Baseline Reference ---
        logger.log(f"=" * 60, tag="DEF", level="INFO")
        logger.log(f"LOADED PRE-COMPUTED PID BASELINE", tag="DEF", level="INFO")
        logger.log(f"  Source: {baseline_run_path}", tag="DEF", level="INFO")
        logger.log(f"  ICs loaded: {len(master_ics)} (eval subset: {n_eval_ics})", tag="DEF", level="INFO")
        logger.log(f"  IC strategy: {ic_data.get('strategy', 'unknown')}, seed: {ic_data.get('seed', 'unknown')}", tag="DEF", level="INFO")
        logger.log(f"  Baseline wall time: {baseline.get('wall_time_sec', '?')}s", tag="DEF", level="INFO")
        logger.log(f"  Diverged:      {baseline.get('n_diverged', '?')}/{baseline.get('n_episodes', '?')}", tag="DEF", level="INFO")
        logger.log(f"  Effort:        mean={baseline['effort']['mean']:.1f}  "
                   f"std={baseline['effort']['std']:.1f}  "
                   f"median={baseline['effort']['median']:.1f}  "
                   f"[{baseline['effort']['min']:.1f}, {baseline['effort']['max']:.1f}]",
                   tag="DEF", level="INFO")
        logger.log(f"  Final Error:   mean={baseline['error']['mean']:.6f}  "
                   f"std={baseline['error']['std']:.6f}  "
                   f"median={baseline['error']['median']:.6f}",
                   tag="DEF", level="INFO")
        logger.log(f"  ITWAE:         mean={baseline['itwae']['mean']:.2f}  "
                   f"std={baseline['itwae']['std']:.2f}  "
                   f"median={baseline['itwae']['median']:.2f}",
                   tag="DEF", level="INFO")
        logger.log(f"  Reward:        mean={baseline['reward']['mean']:.1f}  "
                   f"median={baseline['reward']['median']:.1f}",
                   tag="DEF", level="INFO")
        if baseline.get('convergence_step', {}).get('mean') is not None:
            logger.log(f"  Conv Step:     mean={baseline['convergence_step']['mean']:.0f}  "
                       f"median={baseline['convergence_step']['median']:.0f}",
                       tag="DEF", level="INFO")
        logger.log(f"=" * 60, tag="DEF", level="INFO")

        # --- Training State ---
        total_steps = 0
        best_eval_reward = -float("inf")
        best_eval_error = float("inf")
        training_history = []
        eval_history = []

        start_time = time.time()

        # ================================================================
        # Episode Loop
        # ================================================================
        for episode in range(1, params["total_episodes"] + 1):
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_critic_loss = 0.0
            episode_actor_loss = 0.0
            episode_alpha_val = 0.0
            update_count = 0

            done = False
            while not done:
                # Select action (random during warmup)
                if total_steps < params["warmup_steps"]:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                buffer.add(obs, action, reward, next_obs, float(terminated))
                obs = next_obs
                episode_reward += reward
                total_steps += 1

                # Train agent after warmup
                if total_steps >= params["warmup_steps"] and len(buffer) >= params["batch_size"]:
                    train_info = agent.update(buffer, params["batch_size"])
                    episode_critic_loss += train_info["critic_loss"]
                    episode_actor_loss += train_info["actor_loss"]
                    episode_alpha_val += train_info["alpha"]
                    update_count += 1

            # --- Episode Metrics ---
            ep_metrics = info.get("episode_metrics", {})
            avg_critic = episode_critic_loss / max(update_count, 1)
            avg_actor  = episode_actor_loss / max(update_count, 1)
            avg_alpha  = episode_alpha_val / max(update_count, 1)

            episode_data = {
                "episode": episode,
                "total_steps": total_steps,
                "reward": episode_reward,
                "critic_loss": avg_critic,
                "actor_loss": avg_actor,
                "alpha": avg_alpha,
                "final_error": ep_metrics.get("final_error", None),
                "total_effort": ep_metrics.get("total_effort", None),
                "rl_effort": ep_metrics.get("total_rl_effort", None),
                "itwae": ep_metrics.get("itwae", None),
                "convergence_step": ep_metrics.get("convergence_step", None),
                "diverged": ep_metrics.get("diverged", None),
                "buffer_size": len(buffer),
            }
            training_history.append(episode_data)

            # Log every episode
            logger.log(
                f"Ep {episode:4d}/{params['total_episodes']} | "
                f"R={episode_reward:10.2f} | "
                f"Err={ep_metrics.get('final_error', -1):8.4f} | "
                f"Eff={ep_metrics.get('total_effort', -1):8.1f} | "
                # f"RLEff={ep_metrics.get("total_rl_effort", None)} | "
                f"a={avg_alpha:.4f} | "
                f"Steps={total_steps}",
                tag="DEF", level="INFO"
            )

            # --- Periodic Evaluation ---
            if episode % params["eval_interval"] == 0:
                eval_metrics = evaluate(eval_env, agent, eval_ics)
                eval_metrics["episode"] = episode
                eval_metrics["total_steps"] = total_steps
                eval_history.append(eval_metrics)

                logger.log(
                    f"  [EVAL] R={eval_metrics['eval_reward']:10.2f} | "
                    f"Err={eval_metrics['eval_final_error']:8.4f} | "
                    f"Eff={eval_metrics['eval_total_effort']:8.1f} | "
                    f"ITWAE={eval_metrics['eval_itwae']:8.2f} | "
                    f"ConvStep={eval_metrics['eval_convergence_step']:.0f}",
                    tag="DEF", level="INFO"
                )

                # Save best model by eval reward
                if eval_metrics["eval_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval_reward"]
                    agent.save(run.get_path("data/best_model_reward.pth"))
                    logger.log(f"  [BEST] New best reward: {best_eval_reward:.2f}", tag="FIN", level="INFO")

                # Save best model by final error
                if eval_metrics["eval_final_error"] < best_eval_error:
                    best_eval_error = eval_metrics["eval_final_error"]
                    agent.save(run.get_path("data/best_model_error.pth"))
                    logger.log(f"  [BEST] New best error: {best_eval_error:.6f}", tag="FIN", level="INFO")

            # --- Periodic Checkpoint ---
            if episode % params["checkpoint_interval"] == 0:
                ckpt_path = run.get_path(f"data/checkpoints/ckpt_ep{episode:05d}.pth")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                agent.save(ckpt_path)
                logger.log(f"  Checkpoint saved: ckpt_ep{episode:05d}.pth", tag="PAT", level="INFO")

        # ================================================================
        # Training Complete
        # ================================================================
        elapsed = time.time() - start_time
        logger.log(f"Training complete. Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)", tag="FIN", level="INFO")
        logger.log(f"Best eval reward: {best_eval_reward:.2f}", tag="FIN", level="INFO")
        logger.log(f"Best eval error:  {best_eval_error:.6f}", tag="FIN", level="INFO")

        # --- Save final model ---
        agent.save(run.get_path("data/final_model.pth"))
        logger.log(f"Final model saved.", tag="PAT", level="INFO")

        # --- Save training history ---
        with open(run.get_path("logs/training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2, cls=NumpyEncoder)
        with open(run.get_path("logs/eval_history.json"), "w") as f:
            json.dump(eval_history, f, indent=2, cls=NumpyEncoder)
        logger.log("Training and eval histories saved.", tag="PAT", level="INFO")

        # --- Save summary metrics ---
        summary = {
            "variant": variant,
            "variant_label": variant_config["label"],
            "total_episodes": params["total_episodes"],
            "total_steps": total_steps,
            "training_time_sec": elapsed,
            "best_eval_reward": float(best_eval_reward),
            "best_eval_error": float(best_eval_error),
            "final_buffer_size": len(buffer),
            "device": device,
        }
        with open(run.get_path("logs/summary.json"), "w") as f:
            json.dump(summary, f, indent=4, cls=NumpyEncoder)
        logger.log(f"Summary: {json.dumps(summary, indent=2)}", tag="FIN", level="INFO")

        run.add_notes(
            f"Completed {params['total_episodes']} episodes. "
            f"Best reward={best_eval_reward:.2f}, Best error={best_eval_error:.6f}. "
            f"Time={elapsed:.0f}s."
        )

    except Exception as e:
        logger.log(f"Training failed: {e}", tag="DEF", level="ERROR")
        run.add_notes(f"FAILED: {e}")
        raise
    finally:
        logger.end_session()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Training for Hybrid PID-RL Chaos Control")
    parser.add_argument("--baseline-run", type=str, required=True,
                        help="Path to IC_basePID_stresstest_batch run directory containing "
                             "logs/master_fixed_ics.json and logs/pid_baseline.json")
    parser.add_argument("--variant", type=str, default="6c", choices=["6a", "6b", "6c"],
                        help="Experiment variant: 6a=GALI obs, 6b=GALI reward, 6c=both")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total_episodes")
    parser.add_argument("--freq-ratio", type=int, default=None,
                        help="Override PID:RL frequency ratio")
    parser.add_argument("--lambda", type=float, default=None, dest="rl_lambda",
                        help="Override RL scaling factor")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Override eval interval")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Override number of eval episodes")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params["variant"] = args.variant
    params["baseline_run_path"] = os.path.abspath(args.baseline_run)
    if args.episodes is not None:
        params["total_episodes"] = args.episodes
    if args.freq_ratio is not None:
        params["freq_ratio"] = args.freq_ratio
    if args.rl_lambda is not None:
        params["rl_lambda"] = args.rl_lambda
    if args.eval_interval is not None:
        params["eval_interval"] = args.eval_interval
    if args.eval_episodes is not None:
        params["eval_episodes"] = args.eval_episodes

    train(params)
