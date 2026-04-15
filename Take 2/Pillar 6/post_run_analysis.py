import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force headless rendering to completely bypass Qt platform errors
import glob
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import multiprocessing

# Local imports
from train_sac import DEFAULT_PARAMS, get_variant_config, create_env
from sac_agent import SACAgent

def find_latest_run(base_dir="Pillar 6"):
    # Assuming standard variant paths 6a, 6b, 6c. We will just check all of them.
    search_path = os.path.join(base_dir, "*_batch", "*")
    runs = [d for d in glob.glob(search_path) if os.path.isdir(d) and os.path.exists(os.path.join(d, "logs", "summary.json"))]
    if not runs:
        raise FileNotFoundError("Could not find any viable run directories. Ensure a run has completed.")
    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

def simulate_trajectory(env, agent=None, initial_state=None):
    """
    Simulates a full 4000 step trajectory. 
    If agent is None, computes the Pure PID baseline (no RL residual).
    Returns metrics and the full [step, 6] state trajectory array.
    """
    obs, info = env.reset(options={"initial_state": initial_state.copy()})
    
    trajectory = [initial_state.copy()]
    done = False
    
    while not done:
        if agent is None:
            # Pure PID runs with zero RL action
            action = np.zeros(env.action_space.shape[0])
        else:
            action = agent.select_action(obs, deterministic=True)
            
        obs, reward, terminated, truncated, step_info = env.step(action)
        trajectory.append(env.state.copy())
        done = terminated or truncated

    m = step_info.get("episode_metrics", {})
    return m, np.array(trajectory)

def evaluate_run(run_dir, n_eval=50):
    print(f"=== Beginning Post-Run Analysis for: {run_dir} ===")
    
    # 1. Load configuration and fixed master ICs
    summary_path = os.path.join(run_dir, "logs", "summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    master_ics_path = os.path.join(run_dir, "logs", "master_fixed_ics.json")
    with open(master_ics_path, 'r') as f:
        master_ics = np.array(json.load(f)["master_ics"])
        
    params = DEFAULT_PARAMS.copy()
    params["variant"] = summary["variant"]
    variant_config = get_variant_config(summary["variant"])
    
    # Usually evaluate operates on the first n_eval of the master_ics array.
    eval_ics = master_ics[:n_eval]
    
    # Force the CPU surrogate mapping for massive concurrent stability
    os.environ["FORCE_CPU_SURROGATE"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Create dummy environment to initialize configs
    print("Loading Environment and SAC Agent...")
    env = create_env(params, variant_config)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=params["hidden_dim"],
        device="cpu"  # Force CPU inference for massive parallel deterministic results
    )
    
    # 2. Load latest weights
    ckpt_path = os.path.join(run_dir, "data", "checkpoints", "ckpt_best_model.pth")
    # if best model isn't there, find latest
    if not os.path.exists(ckpt_path):
        ckpts = glob.glob(os.path.join(run_dir, "data", "checkpoints", "ckpt_ep*.pth"))
        ckpt_path = max(ckpts, key=os.path.getctime)
        
    print(f"Loading weights from {ckpt_path}")
    agent.load(ckpt_path, evaluate=True)
    
    print(f"\n--- Initiating EXHAUSTIVE Trajectory Resimulation ({n_eval} ICs) ---")
    
    # We will simulate all 50 ICs. For each, we get RL metrics, RL path, PID metrics, PID path.
    rl_metrics_list = []
    pid_metrics_list = []
    
    rl_paths = []
    pid_paths = []

    # Run sequentially (50 takes maybe 10 seconds total on CPU)
    for idx, ic in enumerate(eval_ics):
        # RL Trajectory
        rl_m, rl_p = simulate_trajectory(env, agent=agent, initial_state=ic)
        rl_metrics_list.append(rl_m)
        rl_paths.append(rl_p)
        
        # Pure PID Trajectory
        pid_m, pid_p = simulate_trajectory(env, agent=None, initial_state=ic)
        pid_metrics_list.append(pid_m)
        pid_paths.append(pid_p)
        
        # Comprehensive step-by-step logging
        eff_p, eff_r = pid_m['total_effort'], rl_m['total_effort']
        err_p, err_r = pid_m['final_error'], rl_m['final_error']
        diff = ((eff_r - eff_p) / eff_p) * 100 if eff_p != 0 else 0
        
        print(f"[IC {idx+1:02d}/{n_eval}] "
              f"Effort: RL={eff_r:7.1f} vs PID={eff_p:7.1f} ({diff:+.1f}%) | "
              f"Error: RL={err_r:.4f} vs PID={err_p:.4f} | "
              f"ITWAE: RL={rl_m['itwae']:5.1f} vs PID={pid_m['itwae']:5.1f}")

    # Aggregating the stats
    def calc_stats(m_list, key):
        arr = np.array([m[key] for m in m_list])
        return np.mean(arr), np.std(arr)

    print("\n" + "="*50)
    print("FINAL COMBINED STATISTICS (OVER IDENTICAL LHS ICs)")
    print("="*50)
    
    for key in ["total_effort", "final_error", "itwae", "max_overshoot"]:
        rl_mu, rl_sd = calc_stats(rl_metrics_list, key)
        pid_mu, pid_sd = calc_stats(pid_metrics_list, key)
        
        # Calculate Delta %
        delta = ((rl_mu - pid_mu) / pid_mu) * 100 if pid_mu != 0 else 0
        
        print(f"Metric: {key.upper()}")
        print(f"  PID Mean : {pid_mu:>10.2f} ± {pid_sd:<10.2f}")
        print(f"  RL Mean  : {rl_mu:>10.2f} ± {rl_sd:<10.2f} | Delta: {delta:+.1f}%")
        print("-" * 30)

    # 2.5 Save individual IC comparisons to JSON
    individual_logs = []
    for i in range(n_eval):
        individual_logs.append({
            "IC_index": i + 1,
            "Metrics_PID": {k: pid_metrics_list[i][k] for k in ["total_effort", "final_error", "itwae", "max_overshoot"]},
            "Metrics_RL": {k: rl_metrics_list[i][k] for k in ["total_effort", "final_error", "itwae", "max_overshoot"]},
            "Deltas_pct": {
                k: ((rl_metrics_list[i][k] - pid_metrics_list[i][k]) / pid_metrics_list[i][k] * 100) 
                   if pid_metrics_list[i][k] != 0 else 0 
                for k in ["total_effort", "final_error", "itwae", "max_overshoot"]
            }
        })
        
    out_json_path = os.path.join(run_dir, "logs", "post_run_individual_metrics.json")
    with open(out_json_path, "w") as f:
        json.dump(individual_logs, f, indent=4)
    print(f"Individual metrics successfully exported to {out_json_path}")

    # 3. Create Data visualizations and directories
    viz_dir = os.path.join(run_dir, "logs", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        plot_bounding_boxplots(rl_metrics_list, pid_metrics_list, viz_dir)
        print(f"\nDrawing 3D Phase Space paths for all {n_eval} Validation ICs...")
        generate_3d_flight_paths(eval_ics, rl_paths, pid_paths, viz_dir)
        print(f"\nAll plots generated and saved to {viz_dir}")
    except Exception as e:
        import traceback
        print(f"CRASH DURING PLOTTING: {e}")
        traceback.print_exc()


def plot_bounding_boxplots(rl, pid, out_dir):
    """Generates comparative boxplots (mean + std deviation shading/whisker equivalents)"""
    keys = ["total_effort", "final_error", "itwae"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, key in enumerate(keys):
        rl_vals = [m[key] for m in rl]
        pid_vals = [m[key] for m in pid]
        
        axes[i].boxplot([pid_vals, rl_vals], tick_labels=["Pure PID", "Hybrid RL"])
        axes[i].set_title(f"Distribution: {key.replace('_',' ').title()}")
        axes[i].set_ylabel(key)
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_distribution.png"), dpi=300)
    plt.close()


def generate_3d_flight_paths(ics, rl_paths, pid_paths, out_dir):
    """
    Generates side-by-side 3D flight paths comparing RL vs PID for each IC.
    Saves an animated MP4 file (if ffmpeg is available) or GIF simulation.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    path_dir = os.path.join(out_dir, "3d_trajectories")
    os.makedirs(path_dir, exist_ok=True)
    
    for i in range(len(ics)):
        rl_trj = rl_paths[i]
        pid_trj = pid_paths[i]
        
        fig = plt.figure(figsize=(16, 8))
        
        # Pure PID Plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f"Pure PID Path (IC {i+1})")
        
        # RL Overlay Plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f"Hybrid RL Path (IC {i+1})")
        
        # Set identical limits to aid visual comparison
        max_val = max(np.max(np.abs(pid_trj[:, :3])), np.max(np.abs(rl_trj[:, :3])))
        for ax in [ax1, ax2]:
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
            ax.set_xlabel('X1')
            ax.set_ylabel('Y1')
            ax.set_zlabel('Z1')
            ax.scatter(0, 0, 0, color='red', s=150, marker='*', label='Origin Target')
            ax.scatter(rl_trj[0, 0], rl_trj[0, 1], rl_trj[0, 2], color='green', s=100, label='Start')

        # Line objects for animation
        line1, = ax1.plot([], [], [], color='black', alpha=0.7, linewidth=1.5, label='PID Orbit')
        line2, = ax2.plot([], [], [], color='blue', alpha=0.7, linewidth=1.5, label='RL Augmented Orbit')
        ax1.legend()
        ax2.legend()
        plt.suptitle(f"Trajectory Comparison - Evaluation IC {i+1}", fontsize=16)
        plt.tight_layout()
        
        # Static save first
        line1.set_data_3d(pid_trj[:, 0], pid_trj[:, 1], pid_trj[:, 2])
        line2.set_data_3d(rl_trj[:, 0], rl_trj[:, 1], rl_trj[:, 2])
        plt.savefig(os.path.join(path_dir, f"path_comparison_ic_{i+1}.png"), dpi=150)

        # Animation (downsampled for speed: 100 frames total)
        total_steps = len(rl_trj)
        stride = max(1, total_steps // 100)
        
        def update(frame):
            idx = min(frame * stride, total_steps - 1)
            line1.set_data_3d(pid_trj[:idx, 0], pid_trj[:idx, 1], pid_trj[:idx, 2])
            line2.set_data_3d(rl_trj[:idx, 0], rl_trj[:idx, 1], rl_trj[:idx, 2])
            return line1, line2
        
        anim = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
        vid_path = os.path.join(path_dir, f"path_comparison_ic_{i+1}.mp4")
        try:
            anim.save(vid_path, fps=20, extra_args=['-vcodec', 'libx264'])
            print(f"  -> Saved MP4 animation for IC {i+1}")
        except Exception:
            # Fallback to GIF if FFMpeg is missing
            gif_path = os.path.join(path_dir, f"path_comparison_ic_{i+1}.gif")
            anim.save(gif_path, writer=PillowWriter(fps=20))
            print(f"  -> Saved GIF animation for IC {i+1} (FFMpeg missing)")
            
        plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to the standard run directory. If none, fetches the latest inside Pillar 6")
    parser.add_argument("--n-eval", type=int, default=50, help="Number of fixed validations ICs to simulate")
    args = parser.parse_args()
    
    target_dir = args.run_dir if args.run_dir else find_latest_run()
    evaluate_run(target_dir, args.n_eval)
