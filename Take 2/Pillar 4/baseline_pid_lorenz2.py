import os
import sys
import numpy as np
from datetime import datetime

# Import custom tools
from ExperimentTracker import ExperimentTracker
from Logger import Logger

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, u_max=50.0):
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

class CoupledLorenzEnv:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, k=2.5, dt=0.005):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.k = k
        self.dt = dt
        
    def _derivatives(self, state, u):
        x1, y1, z1, x2, y2, z2 = state
        uy1, uy2 = u # Control applied only to y variables
        
        dx1 = self.sigma * (y1 - x1) + self.k * (x2 - x1)
        dy1 = x1 * (self.rho - z1) - y1 + self.k * (y2 - y1) + uy1
        dz1 = x1 * y1 - self.beta * z1 + self.k * (z2 - z1)
        
        dx2 = self.sigma * (y2 - x2) + self.k * (x1 - x2)
        dy2 = x2 * (self.rho - z2) - y2 + self.k * (y1 - y2) + uy2
        dz2 = x2 * y2 - self.beta * z2 + self.k * (z1 - z2)
        
        return np.array([dx1, dy1, dz1, dx2, dy2, dz2])
        
    def step(self, state, u):
        # RK4 Integration
        k1 = self._derivatives(state, u)
        k2 = self._derivatives(state + 0.5 * self.dt * k1, u)
        k3 = self._derivatives(state + 0.5 * self.dt * k2, u)
        k4 = self._derivatives(state + self.dt * k3, u)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

def main():
    root_dir = os.path.abspath(os.path.join(os.getcwd(), 'experiments'))
    tracker = ExperimentTracker(root_dir, template={"data": {}, "configs": {}, "logs": {}})
    
    # PID Parameters
    # We load the optimum from ideal_pid_gains.json if it exists
    ideal_gains_path = os.path.join(os.getcwd(), 'ideal_pid_gains.json')
    if os.path.exists(ideal_gains_path):
        import json
        with open(ideal_gains_path, 'r') as f:
            best = json.load(f)
        kp, ki, kd = best['Kp'], best['Ki'], best['Kd']
    else:
        kp, ki, kd = 50.0, 10.0, 5.0
        
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
    
    run = tracker.create_run(params=params, notes="Baseline Pure PID on 2-Coupled Lorenz controlling Y axis.")
    
    logger = Logger(f=run.get_path("logs/run.log"), sesh_file=run.get_path("logs/.sesh_num"))
    logger.start_session()
    logger.log("Starting Baseline PID Simulation for 2-Oscillator Lorenz.", tag="BEG")
    
    env = CoupledLorenzEnv(
        sigma=params['sigma'], 
        rho=params['rho'], 
        beta=params['beta'], 
        k=params['k_coupling'], 
        dt=params['dt']
    )
    
    # Target state: Both origin equilibrium points
    target_state = np.zeros(6) 
    
    # Controllers: One for each oscillator's Y variable
    pid1 = PIDController(params['Kp'], params['Ki'], params['Kd'], params['dt'], u_max=params['u_max'])
    pid2 = PIDController(params['Kp'], params['Ki'], params['Kd'], params['dt'], u_max=params['u_max'])
    
    # Initialize in a highly chaotic state (arbitrary starting point far from origin)
    state = np.array([12.0, 15.0, 35.0, -10.0, -20.0, 40.0])
    initial_state = state.copy()
    
    T = params['T']
    dt = params['dt']
    history_states = np.zeros((T, 2, 3))
    history_t = np.zeros(T)
    
    total_effort = 0.0
    itwae = 0.0
    max_overshoot = 0.0
    
    logger.log("Simulating trajectory...", level="INFO")
    for step in range(T):
        # Calculate error per oscillator's Y
        e1 = target_state[1] - state[1]
        e2 = target_state[4] - state[4]
        
        # Pure PID action
        u1 = pid1.get_action(e1)
        u2 = pid2.get_action(e2)
        
        # Track metrics
        total_effort += abs(u1) + abs(u2)
        current_error_norm = np.linalg.norm(state - target_state)
        time = step * dt
        itwae += time * current_error_norm * dt
        if current_error_norm > max_overshoot:
            max_overshoot = current_error_norm
            
        # Step environment
        state = env.step(state, [u1, u2])
        
        # Store for plotting: format (T, N, 3) where N=2
        history_states[step, 0, :] = state[0:3]
        history_states[step, 1, :] = state[3:6]
        history_t[step] = time

    # Save metrics
    metrics = {
        "start_state": initial_state.tolist(),
        "final_state": state.tolist(),
        "final_error": float(current_error_norm),
        "total_actuator_effort": float(total_effort),
        "ITWAE": float(itwae),
        "max_overshoot": float(max_overshoot)
    }
    run.save_json("logs/metrics.json", metrics)
    
    logger.log(f"Final Error Matrix: {current_error_norm:.4f}", level="INFO")
    logger.log(f"Total Effort: {total_effort:.2f}", level="INFO")
    logger.log(f"ITWAE: {itwae:.2f}", level="INFO")
    logger.log("Simulation complete. Data saved.", tag="FIN")
    logger.end_session()
    
    # Save arrays
    np.savez(run.get_path("data/states.npz"), states=history_states, t=history_t)
    
if __name__ == '__main__':
    main()
