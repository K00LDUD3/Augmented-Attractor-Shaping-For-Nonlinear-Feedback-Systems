import os
import sys
import numpy as np
import json
from datetime import datetime

# Import custom tools
from ExperimentTracker import ExperimentTracker
from Logger import Logger

# Import classes from our existing baseline script
from baseline_pid_lorenz2 import PIDController, CoupledLorenzEnv

def run_simulation(index):
    root_dir = os.path.abspath(os.path.join(os.getcwd(), 'experiments'))
    tracker = ExperimentTracker(root_dir, template={"data": {}, "configs": {}, "logs": {}})
    
    # Load gains
    with open('ideal_pid_gains.json', 'r') as f:
        best = json.load(f)
    kp, ki, kd = best['Kp'], best['Ki'], best['Kd']
        
    params = {
        "Kp": kp,
        "Ki": ki,
        "Kd": kd,
        "u_max": 250.0, 
        "T": 4000,
        "dt": 0.005,
        "k_coupling": 2.5,
        "sigma": 10.0,
        "rho": 28.0,
        "beta": 8.0/3.0
    }
    
    run = tracker.create_run(params=params, notes=f"Randomized Constrained PID Visualization Run #{index}")
    
    logger = Logger(f=run.get_path("logs/run.log"), sesh_file=run.get_path("logs/.sesh_num"))
    logger.start_session()
    
    env = CoupledLorenzEnv(
        sigma=params['sigma'], 
        rho=params['rho'], 
        beta=params['beta'], 
        k=params['k_coupling'], 
        dt=params['dt']
    )
    
    pid1 = PIDController(params['Kp'], params['Ki'], params['Kd'], params['dt'], u_max=params['u_max'])
    pid2 = PIDController(params['Kp'], params['Ki'], params['Kd'], params['dt'], u_max=params['u_max'])
    
    # Randomized initial state [-40, 40]
    initial_state = np.random.uniform(-40, 40, size=6)
    state = initial_state.copy()
    
    T = params['T']
    dt = params['dt']
    history_states = np.zeros((T, 2, 3))
    history_t = np.zeros(T)
    
    target_state = np.zeros(6)
    
    for step in range(T):
        e1 = target_state[1] - state[1]
        e2 = target_state[4] - state[4]
        
        u1 = pid1.get_action(e1)
        u2 = pid2.get_action(e2)
        
        state = env.step(state, [u1, u2])
        
        history_states[step, 0, :] = state[0:3]
        history_states[step, 1, :] = state[3:6]
        history_t[step] = step * dt

    metrics = {
        "start_state": initial_state.tolist(),
        "final_state": state.tolist(),
        "final_error": float(np.linalg.norm(state - target_state))
    }
    run.save_json("logs/metrics.json", metrics)
    np.savez(run.get_path("data/states.npz"), states=history_states, t=history_t)
    logger.end_session()
    return run.uid

if __name__ == "__main__":
    for i in range(2):
        uid = run_simulation(i+1)
        print(f"Generated Run {i+1}: {uid}")
