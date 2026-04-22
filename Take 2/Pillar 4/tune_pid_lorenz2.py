import os
import sys
import numpy as np
import json
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from typing import List

# Import tracking infrastructure
from ExperimentTracker import ExperimentTracker

# Import our baseline environment
from baseline_pid_lorenz2 import CoupledLorenzEnv, PIDController

def create_initial_states(num_states: int = 10, bounds=(-40, 40)):
    """Generate diverse initial states using Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=num_states)
    scaled_samples = qmc.scale(sample, [bounds[0]]*6, [bounds[1]]*6)
    return scaled_samples
    
def evaluate_pid(gains: List[float], initial_states: np.ndarray) -> float:
    Kp, Ki, Kd = gains
    dt = 0.005
    T = 600 # 3 seconds of integration time for rapid testing
    target_state = np.zeros(6)
    
    total_cost = 0.0
    u_max = 300.0
    
    env = CoupledLorenzEnv(k=2.5, dt=dt)
    
    for state0 in initial_states:
        state = state0.copy()
        pid1 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        pid2 = PIDController(Kp, Ki, Kd, dt, u_max=u_max)
        
        itwae = 0.0
        effort = 0.0
        
        for step in range(T):
            e1 = target_state[1] - state[1]
            e2 = target_state[4] - state[4]
            
            u1 = pid1.get_action(e1)
            u2 = pid2.get_action(e2)
            
            effort += abs(u1) + abs(u2)
            
            error_norm = np.linalg.norm(state - target_state)
            time = step * dt
            itwae += time * error_norm * dt
            
            state = env.step(state, [u1, u2])
            
            # Heavy penalty if it completely diverges
            if np.isnan(error_norm) or error_norm > 1e4:
                return 1e9
                
        final_error = np.linalg.norm(state - target_state)
        
        # Cost composition: Emphasize reaching the origin (ITWAE + final_error) 
        # while lightly penalizing ridiculous energy expenditure.
        cost = itwae + (100.0 * final_error) + (0.001 * effort)
        total_cost += cost
        
    return total_cost / len(initial_states)

def main():
    print("Starting PID Robustness Optimization...")
    print("Generating 40 diverse chaotic initial states...")
    initial_states = create_initial_states(num_states=40)
    
    # Kp, Ki, Kd bounds (Kp capped at 40.0 to prevent structural scale mismatch for RL)
    bounds = [
        (0.0, 35.0),  # Kp
        (0.0, 35.0), # Ki
        (0.0, 35.0)  # Kd
    ]
    
    print("Running Differential Evolution (this may take a minute)...")
    
    result = differential_evolution(
        evaluate_pid, 
        bounds, 
        args=(initial_states,), 
        maxiter=20, 
        popsize=10, 
        tol=0.01,
        disp=True
    )
    
    best_Kp, best_Ki, best_Kd = result.x
    
    print("\noptimization finished!")
    print(f"Best Kp: {best_Kp:.4f}")
    print(f"Best Ki: {best_Ki:.4f}")
    print(f"Best Kd: {best_Kd:.4f}")
    print(f"Minimum Average Cost: {result.fun:.4f}")
    
    out_dict = {
        "Kp": float(best_Kp),
        "Ki": float(best_Ki),
        "Kd": float(best_Kd),
        "avg_cost": float(result.fun),
        "test_states_count": 40,
        "system_params": {
            "sigma": 10.0,
            "rho": 28.0,
            "beta": 8.0/3.0,
            "k_coupling": 2.5,
            "dt": 0.005
        }
    }
    
    # Save loosely
    out_file = "ideal_pid_gains.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Saved ideal robust PID gains to '{os.path.abspath(out_file)}'")
    
    # Securely log into an experiment run!
    root_dir = os.path.abspath(os.path.join(os.getcwd(), 'experiments'))
    tracker = ExperimentTracker(root_dir, template={"configs": {}})
    run = tracker.create_run(params=out_dict, notes="PID Robustness Tuning using Scipy Differential Evolution. Test States: 40")
    print(f"Recorded tuning run into Experiment Suite: {run.uid}")

if __name__ == '__main__':
    main()
