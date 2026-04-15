import os
import sys
import json
import numpy as np
import subprocess
from scipy.stats import qmc

sys.path.append(r"C:\CustomTools")
from ExperimentTracker import ExperimentTracker

from baseline_pid_lorenz2 import CoupledLorenzEnv, PIDController

from datetime import datetime

def main():
    num_runs = 50
    print(f"Starting Stress Test: {num_runs} randomized runs.")
    
    # Load Ideal Gains
    with open("D:\\Repos\\Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems\\Take 2\\Pillar 4\\ideal_pid_gains.json", 'r') as f:
        best = json.load(f)
    Kp, Ki, Kd = best['Kp'], best['Ki'], best['Kd']
    
    # Generate random initial states via Latin Hypercube
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=num_runs)
    initial_states = qmc.scale(sample, [-40]*6, [40]*6)
    
    root_dir = os.path.abspath(os.path.join(os.getcwd(), 'experiments'))
    tracker = ExperimentTracker(root_dir, template={"data": {}, "configs": {}, "logs": {}})
    
    T = 4000
    dt = 0.005
    k_coupling = 2.5
    u_max = 250.0
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    target_state = np.zeros(6)
    summary_metrics = []
    gain_values = {
        "kp": Kp,
        "ki": Ki,
        "kd": Kd
    }
    # summary_metrics.append(gain_values)
    
    for i, state0 in enumerate(initial_states):
        print(f"\n--- Running [{i+1}/{num_runs}] ---")
        state = state0.copy()
        
        params = {
            "Kp": Kp, "Ki": Ki, "Kd": Kd,
            "u_max": u_max, "T": T, "dt": dt,
            "k_coupling": k_coupling,
            "sigma": sigma, "rho": rho, "beta": beta,
            "initial_state": state0.tolist()
        }
        
        run = tracker.create_run(params=params, notes=f"Stress Test Run {i+1}/{num_runs}")
        
        env = CoupledLorenzEnv(sigma=sigma, rho=rho, beta=beta, k=k_coupling, dt=dt)
        pid1 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        pid2 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        
        history_states = np.zeros((T, 2, 3))
        history_t = np.zeros(T)
        
        total_effort = 0.0
        itwae = 0.0
        max_overshoot = 0.0
        
        for step in range(T):
            e1 = target_state[1] - state[1]
            e2 = target_state[4] - state[4]
            
            u1 = pid1.get_action(e1)
            u2 = pid2.get_action(e2)
            
            total_effort += abs(u1) + abs(u2)
            current_error_norm = np.linalg.norm(state - target_state)
            time = step * dt
            itwae += time * current_error_norm * dt
            
            if current_error_norm > max_overshoot:
                max_overshoot = current_error_norm
                
            state = env.step(state, [u1, u2])
            
            history_states[step, 0, :] = state[0:3]
            history_states[step, 1, :] = state[3:6]
            history_t[step] = time
            
        metrics = {
            "start_state": state0.tolist(),
            "final_state": state.tolist(),
            "final_error": float(np.linalg.norm(state - target_state)),
            "total_actuator_effort": float(total_effort),
            "ITWAE": float(itwae),
            "max_overshoot": float(max_overshoot)
        }
        
        run.save_json("logs/metrics.json", metrics)
        np.savez(run.get_path("data/states.npz"), states=history_states, t=history_t)
        
        metrics['uid'] = run.uid
        summary_metrics.append(metrics)
        
        print(f"Effort: {total_effort:.2f} | Error: {metrics['final_error']:.4f}")
    
    # Calculate Standard Deviations and Averages
    efforts = [m['total_actuator_effort'] for m in summary_metrics]
    itwaes = [m['ITWAE'] for m in summary_metrics]
    errors = [m['final_error'] for m in summary_metrics]
    overshoots = [m['max_overshoot'] for m in summary_metrics]
    
    stats_summary = {
        "kp":Kp,
        "ki":Ki,
        "kd":Kd,
        "batch_size": num_runs,
        "averages": {
            "effort": float(np.mean(efforts)),
            "itwae": float(np.mean(itwaes)),
            "error": float(np.mean(errors)),
            "overshoot": float(np.mean(overshoots))
        },
        "standard_deviations": {
            "effort": float(np.std(efforts)),
            "itwae": float(np.std(itwaes)),
            "error": float(np.std(errors)),
            "overshoot": float(np.std(overshoots))
        },
        "runs": summary_metrics
    }
    
    # Save overall summary locally with timestamp and batch size
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"stress_test_summary_batch{num_runs}_{timestamp}.json"
    with open(summary_filename, "w") as f:
        json.dump(stats_summary, f, indent=4)
        
    print(f"\nStress Test Completed successfully. Results saved to {summary_filename}")

if __name__ == '__main__':
    main()
