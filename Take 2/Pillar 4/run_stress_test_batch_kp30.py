import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.stats import qmc
from datetime import datetime
import shutil

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Pillar 4')))
from baseline_pid_lorenz2 import CoupledLorenzEnv, PIDController

# Standard Experiment Tracker path
from ExperimentTracker import ExperimentTracker

def render_trajectory(history_states, history_t, output_path, title="AAS Stress Test"):
    """Renders a 3D trajectory animation with ultra-bright axes and restored reference planes."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # 1. Ultra-Bright Color-Coded Euclidean Axes (R-G-B) - No Labels for Legend
    ax.plot([-65, 65], [0, 0], [0, 0], color='#ff7675', lw=0.6, alpha=0.5) # Red X
    ax.plot([0, 0], [-65, 65], [0, 0], color='#55efc4', lw=0.6, alpha=0.5) # Green Y
    ax.plot([0, 0], [0, 0], [0, 65], color='#74b9ff', lw=0.6, alpha=0.5)  # Blue Z
    
    # 2. Restored Thin Floor Mesh Grid
    gx, gy = np.meshgrid(np.linspace(-60, 60, 11), np.linspace(-60, 60, 11))
    gz = np.zeros_like(gx)
    ax.plot_wireframe(gx, gy, gz, color='white', lw=0.3, alpha=0.15)
    
    # 3. Restored Core Bounding Box (-60, 60)
    r = [-60, 60]
    for s, e in [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]:
        x = [r[s&1], r[e&1]]; y = [r[(s>>1)&1], r[(e>>1)&1]]; z = [[0, 60][s>>2&1], [0, 60][e>>2&1]]
        ax.plot(x, y, z, color='#fdcb6e', lw=0.4, alpha=0.2)

    # 4. Sharpened Comet Trails (Length 50)
    line1, = ax.plot([], [], [], lw=1.5, color='#0984e3', alpha=0.9, label='Oscillator 1')
    line2, = ax.plot([], [], [], lw=1.5, color='#d63031', alpha=0.9, label='Oscillator 2')
    
    # 5. Expanded Volumetric Zoom to fit [-60, 60]
    ax.set_xlim([-65, 65]); ax.set_ylim([-65, 65]); ax.set_zlim([0, 70])
    ax.set_title(title, color='white', fontsize=12, alpha=0.4)
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    ax.set_axis_off()

    # Temporal Calibration: Hyper-Smooth 1x Speed Lock
    # 50 FPS * (4 steps * 0.005s/step) = 1.0s real-time / s video
    step_skip = 4
    trail_len = 50 
    writer = FFMpegWriter(fps=50)
    
    with writer.saving(fig, output_path, dpi=100):
        for i in range(0, history_states.shape[0], step_skip):
            start = max(0, i - trail_len)
            line1.set_data(history_states[start:i, 0, 0], history_states[start:i, 0, 1])
            line1.set_3d_properties(history_states[start:i, 0, 2])
            line2.set_data(history_states[start:i, 1, 0], history_states[start:i, 1, 1])
            line2.set_3d_properties(history_states[start:i, 1, 2])
            writer.grab_frame()
    plt.close(fig)

def run_batch():
    num_runs = 20
    print(f"--- AAS Pillar 4 Batch v5 Initiated [{num_runs} Runs] ---")
    
    # Gains from optimal kp30 run
    Kp = 30.0
    Ki = 0.028641224628840715
    Kd = 0.0
    u_max = 140.0
    T = 2000 
    dt = 0.005
    k_coupling = 2.5
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    target_state = np.zeros(6)

    drive_ready_root = os.path.abspath(os.path.join(os.getcwd(), 'Pillar 4', 'experiments', 'Drive_Upload_PID_KP30'))
    tracker = ExperimentTracker(drive_ready_root, template={"configs": {}, "logs": {}, "data": {}})

    # SURFACE (SHELL) SAMPLING
    # We sample from the volume and project to the boundary of the core (-30, 30)
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=num_runs)
    volume_states = qmc.scale(sample, [-60]*6, [60]*6)
    
    initial_states = []
    for state in volume_states:
        # Pick one random dimension to be "on the surface"
        dim = np.random.randint(0, 6)
        side = np.random.choice([-60.0, 60.0])
        state[dim] = side
        initial_states.append(state)

    for i, state0 in enumerate(initial_states):
        print(f"Executing Run [{i+1}/{num_runs}]...")
        state = state0.copy()
        
        params = {
            "Kp": Kp, "Ki": Ki, "Kd": Kd,
            "u_max": u_max, "T": T, "dt": dt,
            "k_coupling": k_coupling,
            "initial_state": state0.tolist()
        }
        
        run = tracker.create_run(params=params, notes=f"Drive_Upload Stress Test {i+1}")
        uid = run.uid
        run_root = run.path

        # Sim Setup
        env = CoupledLorenzEnv(sigma=sigma, rho=rho, beta=beta, k=k_coupling, dt=dt)
        pid1 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        pid2 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        
        history_states = np.zeros((T, 2, 3))
        history_t = np.zeros(T)
        
        total_effort = 0.0
        
        for step in range(T):
            e1 = target_state[1] - state[1]
            e2 = target_state[4] - state[4]
            u1 = pid1.get_action(e1); u2 = pid2.get_action(e2)
            total_effort += abs(u1) + abs(u2)
            state = env.step(state, [u1, u2])
            history_states[step, 0, :] = state[0:3]
            history_states[step, 1, :] = state[3:6]
            history_t[step] = step * dt

        # Metrics
        metrics = {
            "final_error": float(np.linalg.norm(state - target_state)),
            "total_effort": float(total_effort),
            "initial_state": state0.tolist()
        }
        
        # Video Rendering (Headless)
        mp4_path = os.path.join(run_root, f"{uid}.mp4")
        render_trajectory(history_states, history_t, mp4_path, title=f"Run {uid}")

        # Flatten Structure
        # 1. Save metrics to root
        with open(os.path.join(run_root, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # 2. Prune logs/data but keep configs
        try:
            shutil.rmtree(os.path.join(run_root, "logs"))
            shutil.rmtree(os.path.join(run_root, "data"))
        except Exception as e:
            print(f"Cleanup warning: {e}")

        print(f"Run {uid} Finalized. (Error: {metrics['final_error']:.4f})")

    print("\n--- Batch SUCCESS. Drive_Upload_PID_KP30 Ready. ---")

if __name__ == "__main__":
    run_batch()
