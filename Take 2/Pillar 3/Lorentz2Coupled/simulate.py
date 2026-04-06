import os
import sys
import json
import numpy as np
from scipy.integrate import solve_ivp

# Add CustomTools to path
sys.path.append(r"C:\CustomTools")
from ExperimentTracker import ExperimentTracker
from Logger import Logger

def lorenz_coupled_deriv(t, state, sigma, rho, beta, K):
    # state is flat, shape (6,) -> [x1, y1, z1, x2, y2, z2]
    x1, y1, z1, x2, y2, z2 = state
    
    # Diffusive coupling on the x components
    dx1 = sigma * (y1 - x1) + K * (x2 - x1)
    dy1 = x1 * (rho - z1) - y1
    dz1 = x1 * y1 - beta * z1
    
    dx2 = sigma * (y2 - x2) + K * (x1 - x2)
    dy2 = x2 * (rho - z2) - y2
    dz2 = x2 * y2 - beta * z2
    
    return [dx1, dy1, dz1, dx2, dy2, dz2]

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
    run = tracker.create_run(params=config, notes="Coupled 2-Lorenz Oscillators simulation. Increasing timesteps")
    
    # Init Logger
    logger = Logger(f=run.get_path("logs/run.log"), sesh_file=run.get_path("logs/.sesh_num"))
    logger.start_session()
    
    logger.log("Simulation started.", tag="BEG")
    logger.log(f"Run UID: {run.uid}")
    
    T = sim_params["T_steps"]
    dt = sim_params["dt"]
    N = sim_params["N"]
    
    # init conditions list of lists -> flat array
    state0 = np.array(sim_params["initial_conditions"], dtype=float).flatten()
    
    t_array = np.linspace(0, T * dt, T)
    
    sigma = eq_params["sigma"]
    rho = eq_params["rho"]
    beta = eq_params["beta"]
    K = eq_params["K"]
    
    logger.log("Integrating with scipy.integrate.solve_ivp (RK45)...", tag="DEF")
    sol = solve_ivp(
        fun=lorenz_coupled_deriv, 
        t_span=(0, t_array[-1]), 
        y0=state0, 
        args=(sigma, rho, beta, K),
        t_eval=t_array,
        method="RK45"
    )
    
    # sol.y has shape (6, T)
    # Target shape is (T, N, 3) 
    # sol.y.T -> shape (T, 6). reshape(T, 2, 3) 
    states_array = sol.y.T.reshape(T, N, 3)
    
    transient_drop_factor = sim_params.get("transient_drop_factor", 0.0)
    drop_idx = int(T * transient_drop_factor)
    if drop_idx > 0:
        logger.log(f"Dropping first {drop_idx} frames ({transient_drop_factor*100}%) as transient...", tag="DEF")
        states_array = states_array[drop_idx:]
        t_array = t_array[drop_idx:]
        
    logger.log("Integration complete.", tag="DEF")
    
    # Save output
    logger.log("Saving data to states.npz...", tag="DEF")
    np.savez(run.get_path("data/states.npz"), states=states_array, t=t_array)
    
    logger.log("Data saved successfully.", tag="FIN")
    logger.end_session()
    
    print(f"Simulation completed and saved to {run.uid}")

if __name__ == "__main__":
    main()
