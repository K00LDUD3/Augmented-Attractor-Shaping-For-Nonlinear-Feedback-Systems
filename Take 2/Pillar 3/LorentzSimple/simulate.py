import os
import sys
import json
import numpy as np
from scipy.integrate import solve_ivp

# Add CustomTools to path
from ExperimentTracker import ExperimentTracker
from Logger import Logger

def lorenz_deriv(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def main():
    # Load config
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    eq_params = config["equation"]
    sim_params = config["simulation"]
    
    # Init Tracker
    experiments_dir = os.path.abspath(os.path.join(os.getcwd(), "experiments"))
    tracker = ExperimentTracker(experiments_dir, template={"data": {}, "logs": {}})
    run = tracker.create_run(params=config, notes="Single Lorenz Oscillator simulation.")
    
    # Init Logger
    logger = Logger(f=run.get_path("logs/run.log"), sesh_file=run.get_path("logs/.sesh_num"))
    logger.start_session()
    
    logger.log("Simulation started.", tag="BEG")
    logger.log(f"Run UID: {run.uid}")
    
    T = sim_params["T_steps"]
    dt = sim_params["dt"]
    N = sim_params["N"]
    state = np.array(sim_params["initial_conditions"], dtype=float)
    
    t_array = np.linspace(0, T * dt, T)
    states_array = np.zeros((T, N, 3))
    
    sigma, rho, beta = eq_params["sigma"], eq_params["rho"], eq_params["beta"]
    
    logger.log("Integrating with scipy.integrate.solve_ivp (RK45)...", tag="DEF")
    sol = solve_ivp(
        fun=lorenz_deriv, 
        t_span=(0, t_array[-1]), 
        y0=state, 
        args=(sigma, rho, beta),
        t_eval=t_array,
        method="RK45"
    )
    
    states_array = sol.y.T.reshape(T, N, 3)
        
    logger.log("Integration complete.", tag="DEF")
    
    # Save output
    logger.log("Saving data to states.npz...", tag="DEF")
    np.savez(run.get_path("data/states.npz"), states=states_array, t=t_array)
    
    logger.log("Data saved successfully.", tag="FIN")
    logger.end_session()
    
    print(f"Simulation completed and saved to {run.uid}")

if __name__ == "__main__":
    main()
