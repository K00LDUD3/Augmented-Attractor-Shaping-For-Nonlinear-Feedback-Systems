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

# --- Configuration Options ---
ONLY_NUMERICS = False  # Set to True to skip all MP4/PNG generation and only export JSONs

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
    actions = []
    pids = []
    totals = []
    done = False
    
    while not done:
        if agent is None:
            # Pure PID runs with zero RL action
            action = np.zeros(env.action_space.shape[0])
        else:
            action = agent.select_action(obs, deterministic=True)
            
        actions.append(action.copy())
        obs, reward, terminated, truncated, step_info = env.step(action)
        trajectory.append(env.state.copy())
        
        if "u_pid" in step_info:
            pids.append(step_info["u_pid"])
            totals.append(step_info["u_total"])
            
        done = terminated or truncated

    m = step_info.get("episode_metrics", {})
    return m, np.array(trajectory), np.array(actions), np.array(pids), np.array(totals)

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
    env.episode_length = 2000  # Cutoff at 2000 timesteps natively based on user request
    
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
    rl_actions = []
    rl_pids = []
    rl_totals = []

    # Run sequentially (50 takes maybe 10 seconds total on CPU)
    for idx, ic in enumerate(eval_ics):
        # RL Trajectory
        rl_m, rl_p, rl_a, rl_pid, rl_tot = simulate_trajectory(env, agent=agent, initial_state=ic)
        rl_metrics_list.append(rl_m)
        rl_paths.append(rl_p)
        rl_actions.append(rl_a)
        rl_pids.append(rl_pid)
        rl_totals.append(rl_tot)
        
        # Pure PID Trajectory
        pid_m, pid_p, _, _, _ = simulate_trajectory(env, agent=None, initial_state=ic)
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
    
    if ONLY_NUMERICS:
        print("\n[CONFIG] ONLY_NUMERICS is True. Skipping all Plotting/MP4 stages.")
        print("Exporting LLM Numerical Digests...")
        export_numerical_diagnostics(eval_ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, pid_metrics_list, rl_metrics_list, viz_dir)
        print(f"\nNumeric exports generated perfectly and saved to {viz_dir}")
        return

    try:
        plot_bounding_boxplots(rl_metrics_list, pid_metrics_list, viz_dir)
        print(f"\nDrawing original Legacy 3D Phase Space paths for all {n_eval} Validation ICs...")
        generate_3d_flight_paths(eval_ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, viz_dir)
        
        print("\nDrawing NEW Kinematic Flight Paths...")
        generate_kinematic_flight_paths(eval_ics, rl_paths, pid_paths, viz_dir)

        print("\nDrawing NEW Diagnostic Timelines...")
        generate_diagnostic_timelines(eval_ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, viz_dir)
        
        print("\nExporting LLM Numerical Digests...")
        export_numerical_diagnostics(eval_ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, pid_metrics_list, rl_metrics_list, viz_dir)
        
        print(f"\nAll plots and numeric exports generated perfectly and saved to {viz_dir}")
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
    plt.savefig(os.path.join(out_dir, "metrics_distribution.png"), dpi=500)
    plt.close()


def generate_3d_flight_paths(ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, out_dir):
    """
    Generates side-by-side 3D flight paths comparing RL vs PID for each IC (Oscillator 1 & 2).
    Also plots the RL Action Magnitude over time, and PID vs Total signals.
    Saves an animated MP4 file (if ffmpeg is available) or GIF simulation.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    path_dir = os.path.join(out_dir, "3d_trajectories")
    os.makedirs(path_dir, exist_ok=True)
    
    for i in range(len(ics)):
        rl_trj = rl_paths[i]
        pid_trj = pid_paths[i]
        rl_act = rl_actions[i]
        rl_pid_sig = rl_pids[i]
        rl_tot_sig = rl_totals[i]
        
        # NOTE: If DPI is set to e.g. 1000 here, with a 24x16 figure, the animation tries to render 
        # a 24000x16000 matrix per frame, causing MemoryErrors and ffmpeg Invalid Size crashes.
        # 150 DPI yields 3600x2400 out of the box, which is extremely crisp 4K-tier resolution.
        fig = plt.figure(figsize=(24, 16), dpi=150)
        
        # Pure PID Plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_title(f"Pure PID Path (IC {i+1})")
        
        # RL Overlay Plot
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.set_title(f"Hybrid RL Path (IC {i+1})")
        
        # Action Plot
        ax3 = fig.add_subplot(223)
        ax3.set_title("RL Residual Thrust Magnitude over Time")
        ax3.set_ylabel("L1 Control Effort (Sum of |u_RL|)")
        ax3.set_xlabel("Time Step")

        # Total vs PID Plot
        ax4 = fig.add_subplot(224)
        ax4.set_title("PID Signal vs Total Control Signal")
        ax4.set_ylabel("L1 Control Effort (Sum of |u|)")
        ax4.set_xlabel("Time Step")
        
        # Set identical limits to aid visual comparison
        max_val = max(np.max(np.abs(pid_trj[:, :6])), np.max(np.abs(rl_trj[:, :6])))
        for ax in [ax1, ax2]:
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.scatter(0, 0, 0, color='red', s=150, marker='*', label='Origin Target')
            # Start points for Osc 1 and Osc 2
            ax.scatter(rl_trj[0, 0], rl_trj[0, 1], rl_trj[0, 2], color='green', s=100, label='O1 Start')
            ax.scatter(rl_trj[0, 3], rl_trj[0, 4], rl_trj[0, 5], color='purple', s=100, label='O2 Start')

        # Line objects for animation
        line1_o1, = ax1.plot([], [], [], color='black', alpha=0.7, linewidth=1.5, label='O1 (PID)')
        line1_o2, = ax1.plot([], [], [], color='gray', alpha=0.5, linewidth=1.0, linestyle='--', label='O2 (PID)')
        
        line2_o1, = ax2.plot([], [], [], color='blue', alpha=0.7, linewidth=1.5, label='O1 (RL)')
        line2_o2, = ax2.plot([], [], [], color='cyan', alpha=0.5, linewidth=1.0, linestyle='--', label='O2 (RL)')
        
        ax1.legend()
        ax2.legend()
        
        # Action magnitude Line (L1 norm of the 5D action)
        act_mags = np.sum(np.abs(rl_act), axis=1) * 25.0  # Scale by rl_lambda=25.0 approximately to show true force size
        ax3.plot(act_mags, color='orange', alpha=0.8, linewidth=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, len(act_mags)])
        ax3.set_ylim([0, max(1.0, np.max(act_mags)) * 1.2])

        # Moving dot for action magnitude
        act_point, = ax3.plot([], [], 'ro', markersize=8)

        # PID vs Total Lines
        pid_mags = np.sum(np.abs(rl_pid_sig), axis=1)
        tot_mags = np.sum(np.abs(rl_tot_sig), axis=1)
        ax4.plot(pid_mags, color='black', alpha=0.3, linewidth=2, label='PID Baseline Signal')
        ax4.plot(tot_mags, color='purple', alpha=0.8, linewidth=2, label='Total Hybrid Signal')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, len(pid_mags)])
        ax4.set_ylim([0, max(1.0, np.max(tot_mags), np.max(pid_mags)) * 1.2])

        # Moving dots for signal comparison
        pid_point, = ax4.plot([], [], 'ko', markersize=6)
        tot_point, = ax4.plot([], [], 'mo', markersize=8)

        # On-screen timer display
        time_text = fig.text(0.02, 0.95, "Time: 0.00s", fontsize=20, color='darkred', weight='bold')

        plt.suptitle(f"Trajectory Comparison - Evaluation IC {i+1}", fontsize=20)
        plt.tight_layout()
        
        # Static save first
        line1_o1.set_data_3d(pid_trj[:, 0], pid_trj[:, 1], pid_trj[:, 2])
        line1_o2.set_data_3d(pid_trj[:, 3], pid_trj[:, 4], pid_trj[:, 5])
        
        line2_o1.set_data_3d(rl_trj[:, 0], rl_trj[:, 1], rl_trj[:, 2])
        line2_o2.set_data_3d(rl_trj[:, 3], rl_trj[:, 4], rl_trj[:, 5])
        
        act_point.set_data([len(act_mags)-1], [act_mags[-1]])
        pid_point.set_data([len(pid_mags)-1], [pid_mags[-1]])
        tot_point.set_data([len(tot_mags)-1], [tot_mags[-1]])
        
        plt.savefig(os.path.join(path_dir, f"path_comparison_ic_{i+1}.png"), dpi=500)

        # Animation (downsampled for speed: 100 moving frames, plus 40 hold frames at the start)
        total_steps = len(rl_trj)
        stride = max(1, total_steps // 100)
        hold_frames = 40  # 2 seconds at 20fps
        
        def update(frame):
            if frame < hold_frames:
                idx = 1
                act_idx = 0
                sig_idx = 0
                time_text.set_text("Time: 0.00s [HOLD]")
            else:
                active_frame = frame - hold_frames
                idx = min((active_frame + 1) * stride, total_steps)
                act_idx = min(idx, len(act_mags)-1)
                sig_idx = min(idx, len(pid_mags)-1)
                time_text.set_text(f"Time: {(idx * 0.005):.2f}s")
            
            line1_o1.set_data_3d(pid_trj[:idx, 0], pid_trj[:idx, 1], pid_trj[:idx, 2])
            line1_o2.set_data_3d(pid_trj[:idx, 3], pid_trj[:idx, 4], pid_trj[:idx, 5])
            
            line2_o1.set_data_3d(rl_trj[:idx, 0], rl_trj[:idx, 1], rl_trj[:idx, 2])
            line2_o2.set_data_3d(rl_trj[:idx, 3], rl_trj[:idx, 4], rl_trj[:idx, 5])
            
            act_point.set_data([act_idx], [act_mags[act_idx]])
            pid_point.set_data([sig_idx], [pid_mags[sig_idx]])
            tot_point.set_data([sig_idx], [tot_mags[sig_idx]])
            
            return line1_o1, line1_o2, line2_o1, line2_o2, act_point, pid_point, tot_point, time_text
        
        anim = FuncAnimation(fig, update, frames=100 + hold_frames, interval=50, blit=False)
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

def generate_kinematic_flight_paths(ics, rl_paths, pid_paths, out_dir):
    from matplotlib.animation import FuncAnimation, PillowWriter
    path_dir = os.path.join(out_dir, "kinematic_trajectories")
    os.makedirs(path_dir, exist_ok=True)
    
    for i in range(len(ics)):
        rl_trj = rl_paths[i]
        pid_trj = pid_paths[i]
        
        fig = plt.figure(figsize=(16, 8), dpi=100) # Safe for ffmpeg memory limits
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f"Pure PID Kinematics (IC {i+1})")
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f"Hybrid RL Kinematics (IC {i+1})")
        
        max_val = max(np.max(np.abs(pid_trj[:, :6])), np.max(np.abs(rl_trj[:, :6])))
        for ax in [ax1, ax2]:
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.scatter(0, 0, 0, color='red', s=150, marker='*', label='Origin Target')
            ax.scatter(rl_trj[0, 0], rl_trj[0, 1], rl_trj[0, 2], color='green', s=100, label='O1 Start')
            ax.scatter(rl_trj[0, 3], rl_trj[0, 4], rl_trj[0, 5], color='purple', s=100, label='O2 Start')

        line1_o1, = ax1.plot([], [], [], color='black', alpha=0.7, linewidth=1.5, label='O1 (PID)')
        line1_o2, = ax1.plot([], [], [], color='gray', alpha=0.5, linewidth=1.0, linestyle='--', label='O2 (PID)')
        line2_o1, = ax2.plot([], [], [], color='blue', alpha=0.7, linewidth=1.5, label='O1 (RL)')
        line2_o2, = ax2.plot([], [], [], color='cyan', alpha=0.5, linewidth=1.0, linestyle='--', label='O2 (RL)')
        
        ax1.legend()
        ax2.legend()
        
        time_text = fig.text(0.02, 0.92, "Time: 0.00s", fontsize=16, color='darkred', weight='bold')
        plt.tight_layout()
        
        line1_o1.set_data_3d(pid_trj[:, 0], pid_trj[:, 1], pid_trj[:, 2])
        line1_o2.set_data_3d(pid_trj[:, 3], pid_trj[:, 4], pid_trj[:, 5])
        line2_o1.set_data_3d(rl_trj[:, 0], rl_trj[:, 1], rl_trj[:, 2])
        line2_o2.set_data_3d(rl_trj[:, 3], rl_trj[:, 4], rl_trj[:, 5])
        plt.savefig(os.path.join(path_dir, f"kinematic_ic_{i+1}.png"), dpi=300)

        total_steps = len(rl_trj)
        stride = max(1, total_steps // 100)
        hold_frames = 40
        
        def update(frame):
            if frame < hold_frames:
                idx = 1
                time_text.set_text("Time: 0.00s [HOLD]")
            else:
                active_frame = frame - hold_frames
                idx = min((active_frame + 1) * stride, total_steps)
                time_text.set_text(f"Time: {(idx * 0.005):.2f}s")
            
            line1_o1.set_data_3d(pid_trj[:idx, 0], pid_trj[:idx, 1], pid_trj[:idx, 2])
            line1_o2.set_data_3d(pid_trj[:idx, 3], pid_trj[:idx, 4], pid_trj[:idx, 5])
            line2_o1.set_data_3d(rl_trj[:idx, 0], rl_trj[:idx, 1], rl_trj[:idx, 2])
            line2_o2.set_data_3d(rl_trj[:idx, 3], rl_trj[:idx, 4], rl_trj[:idx, 5])
            return line1_o1, line1_o2, line2_o1, line2_o2, time_text

        anim = FuncAnimation(fig, update, frames=100 + hold_frames, interval=50, blit=False)
        vid_path = os.path.join(path_dir, f"kinematic_ic_{i+1}.mp4")
        try:
            anim.save(vid_path, fps=20, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        except Exception as e:
            print(f"FAILED FFmpeg (Kinematic {i+1}): {e}")
        plt.close(fig)

def generate_diagnostic_timelines(ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, out_dir):
    from matplotlib.animation import FuncAnimation, PillowWriter
    path_dir = os.path.join(out_dir, "diagnostic_trajectories")
    os.makedirs(path_dir, exist_ok=True)
    
    for i in range(len(ics)):
        rl_trj = rl_paths[i]
        pid_trj = pid_paths[i]
        rl_act = rl_actions[i]
        rl_pid_sig = rl_pids[i]
        rl_tot_sig = rl_totals[i]
        
        pid_e1 = np.sum(np.abs(pid_trj[:, 0:3]), axis=1)
        pid_e2 = np.sum(np.abs(pid_trj[:, 3:6]), axis=1)
        rl_e1 = np.sum(np.abs(rl_trj[:, 0:3]), axis=1)
        rl_e2 = np.sum(np.abs(rl_trj[:, 3:6]), axis=1)
        
        act_mags = np.sum(np.abs(rl_act), axis=1) * 25.0
        pid_mags = np.sum(np.abs(rl_pid_sig), axis=1)
        tot_mags = np.sum(np.abs(rl_tot_sig), axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=100) # Safe memory size
        fig.suptitle(f"Diagnostics (IC {i+1})", fontsize=16)
        ax_e1, ax_e2 = axes[0, 0], axes[0, 1]
        ax_act, ax_sig = axes[1, 0], axes[1, 1]
        
        ax_e1.set_title("Oscillator 1 Absolute Error")
        ax_e1.plot(pid_e1, 'k', alpha=0.5, label='PID Baseline')
        ax_e1.plot(rl_e1, 'b', alpha=0.8, label='Hybrid RL')
        ax_e1.legend()
        ax_e1.grid(True)
        ax_e1.set_xlim([0, len(pid_e1)])
        ax_e1.set_ylim([0, max(1.0, np.max(pid_e1), np.max(rl_e1)) * 1.1])
        dot_pid_e1, = ax_e1.plot([], [], 'ko', markersize=6)
        dot_rl_e1, = ax_e1.plot([], [], 'bo', markersize=8)

        ax_e2.set_title("Oscillator 2 Absolute Error")
        ax_e2.plot(pid_e2, 'k', alpha=0.5, label='PID Baseline')
        ax_e2.plot(rl_e2, 'c', alpha=0.8, label='Hybrid RL')
        ax_e2.legend()
        ax_e2.grid(True)
        ax_e2.set_xlim([0, len(pid_e2)])
        ax_e2.set_ylim([0, max(1.0, np.max(pid_e2), np.max(rl_e2)) * 1.1])
        dot_pid_e2, = ax_e2.plot([], [], 'ko', markersize=6)
        dot_rl_e2, = ax_e2.plot([], [], 'co', markersize=8)

        ax_act.set_title("RL Residual Thrust")
        ax_act.plot(act_mags, color='orange', alpha=0.8)
        ax_act.grid(True)
        ax_act.set_xlim([0, len(act_mags)])
        ax_act.set_ylim([0, max(1.0, np.max(act_mags)) * 1.2])
        dot_act, = ax_act.plot([], [], 'ro', markersize=8)

        ax_sig.set_title("PID vs Total Signal")
        ax_sig.plot(pid_mags, color='black', alpha=0.3, label='PID Baseline Signal')
        ax_sig.plot(tot_mags, color='purple', alpha=0.8, label='Total Hybrid Signal')
        ax_sig.legend()
        ax_sig.grid(True)
        ax_sig.set_xlim([0, len(pid_mags)])
        ax_sig.set_ylim([0, max(1.0, np.max(tot_mags), np.max(pid_mags)) * 1.2])
        dot_pid_sig, = ax_sig.plot([], [], 'ko', markersize=6)
        dot_tot_sig, = ax_sig.plot([], [], 'mo', markersize=8)

        time_text = fig.text(0.02, 0.95, "Time: 0.00s", fontsize=16, color='darkred', weight='bold')
        plt.tight_layout()

        dot_pid_e1.set_data([len(pid_e1)-1], [pid_e1[-1]])
        dot_rl_e1.set_data([len(rl_e1)-1], [rl_e1[-1]])
        dot_pid_e2.set_data([len(pid_e2)-1], [pid_e2[-1]])
        dot_rl_e2.set_data([len(rl_e2)-1], [rl_e2[-1]])
        dot_act.set_data([len(act_mags)-1], [act_mags[-1]])
        dot_pid_sig.set_data([len(pid_mags)-1], [pid_mags[-1]])
        dot_tot_sig.set_data([len(tot_mags)-1], [tot_mags[-1]])
        plt.savefig(os.path.join(path_dir, f"diagnostic_ic_{i+1}.png"), dpi=300)

        total_steps = len(rl_trj)
        stride = max(1, total_steps // 100)
        hold_frames = 40

        def update(frame):
            if frame < hold_frames:
                idx = 1
                time_text.set_text("Time: 0.00s [HOLD]")
            else:
                active_frame = frame - hold_frames
                idx = min((active_frame + 1) * stride, total_steps)
                time_text.set_text(f"Time: {(idx * 0.005):.2f}s")
                
            act_idx = min(idx, len(act_mags)-1)
            sig_idx = min(idx, len(pid_mags)-1)
            idx_e = min(idx, len(pid_e1)-1)
            
            dot_pid_e1.set_data([idx_e], [pid_e1[idx_e]])
            dot_rl_e1.set_data([idx_e], [rl_e1[idx_e]])
            dot_pid_e2.set_data([idx_e], [pid_e2[idx_e]])
            dot_rl_e2.set_data([idx_e], [rl_e2[idx_e]])
            dot_act.set_data([act_idx], [act_mags[act_idx]])
            dot_pid_sig.set_data([sig_idx], [pid_mags[sig_idx]])
            dot_tot_sig.set_data([sig_idx], [tot_mags[sig_idx]])

            return dot_pid_e1, dot_rl_e1, dot_pid_e2, dot_rl_e2, dot_act, dot_pid_sig, dot_tot_sig, time_text
        
        anim = FuncAnimation(fig, update, frames=100 + hold_frames, interval=50, blit=False)
        vid_path = os.path.join(path_dir, f"diagnostic_ic_{i+1}.mp4")
        try:
            anim.save(vid_path, fps=20, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        except Exception as e:
            print(f"FAILED FFmpeg (Diagnostics {i+1}): {e}")
        plt.close(fig)

def export_numerical_diagnostics(ics, rl_paths, pid_paths, rl_actions, rl_pids, rl_totals, pid_metrics_list, rl_metrics_list, out_dir):
    """
    Exports a concise, aggregated JSON summary of the run metrics over time windows. 
    Designed for easy copy-pasting into LLMs.
    """
    num_dir = os.path.join(out_dir, "Numerical_info")
    os.makedirs(num_dir, exist_ok=True)
    
    dt = 0.005
    window_size = 200 # 200 steps * 0.005s = 1.0s intervals
    
    for i in range(len(ics)):
        rl_trj = rl_paths[i]
        pid_trj = pid_paths[i]
        
        # Ensure summary metrics are native python types for JSON serialization
        def serialize_dict(d):
            return {k: float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in d.items()}

        intervals = []
        num_steps = len(rl_actions[i])
        
        for start_idx in range(0, num_steps, window_size):
            end_idx = min(start_idx + window_size, num_steps)
            if start_idx == end_idx: break
            
            # Time Range
            t_start = start_idx * dt
            t_end = (end_idx - 1) * dt
            
            # Mean Abs Error in window (Sum over channels, mean over time)
            w_pid_e1 = np.mean(np.sum(np.abs(pid_trj[start_idx:end_idx, 0:3]), axis=1))
            w_pid_e2 = np.mean(np.sum(np.abs(pid_trj[start_idx:end_idx, 3:6]), axis=1))
            w_rl_e1 = np.mean(np.sum(np.abs(rl_trj[start_idx:end_idx, 0:3]), axis=1))
            w_rl_e2 = np.mean(np.sum(np.abs(rl_trj[start_idx:end_idx, 3:6]), axis=1))
            
            # Total Effort in window
            w_pid_sig = np.sum(np.sum(np.abs(rl_pids[i][start_idx:end_idx]), axis=1))
            w_tot_sig = np.sum(np.sum(np.abs(rl_totals[i][start_idx:end_idx]), axis=1))
            w_rl_sig  = np.sum(np.sum(np.abs(rl_actions[i][start_idx:end_idx]), axis=1))
            
            intervals.append({
                "Interval_S": f"{t_start:.1f}-{t_end:.1f}",
                "Avg_Error": {
                    "PID": (float(w_pid_e1) + float(w_pid_e2))/2.0,
                    "RL": (float(w_rl_e1) + float(w_rl_e2))/2.0
                },
                "Effort_Sum": {
                    "PID": float(w_pid_sig),
                    "Hybrid": float(w_tot_sig),
                    "RL_Resid": float(w_rl_sig)
                }
            })

        data = {
            "IC_Index": i + 1,
            "Episode_Summary": {
                "PID_Baseline": serialize_dict(pid_metrics_list[i]),
                "Hybrid_RL": serialize_dict(rl_metrics_list[i])
            },
            "One_Second_Intervals": intervals
        }
        
        with open(os.path.join(num_dir, f"numerical_ic_{i+1:02d}_summary.json"), 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to the standard run directory. If none, fetches the latest inside Pillar 6")
    parser.add_argument("--n-eval", type=int, default=50, help="Number of fixed validations ICs to simulate")
    parser.add_argument("--numerics-only", action="store_true", help="Skip all visualizations and only export JSON data")
    args = parser.parse_args()
    
    if args.numerics_only:
        ONLY_NUMERICS = True
        
    target_dir = args.run_dir if args.run_dir else find_latest_run()
    evaluate_run(target_dir, args.n_eval)
