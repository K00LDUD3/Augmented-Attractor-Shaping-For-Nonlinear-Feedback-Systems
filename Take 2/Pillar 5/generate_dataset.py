import numpy as np
import os
import argparse
import sys
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

sys.path.append(r"C:\CustomTools")
from ExperimentTracker import ExperimentTracker

# Lorenz system parameters (from Phase 2)
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
K_C = 2.0  # coupling strength

def coupled_lorenz_with_deviations(t, state):
    """
    18D system:
    indices 0..5: original state (x, y, z, u, v, w)
    indices 6..11: deviation vector 1 (dx1, dy1, dz1, du1, dv1, dw1)
    indices 12..17: deviation vector 2 (dx2, dy2, dz2, du2, dv2, dw2)
    """
    x, y, z, u, v, w = state[0:6]
    
    # 1. State derivative
    dx = SIGMA * (y - x) + K_C * (u - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    du = SIGMA * (v - u) + K_C * (x - u)
    dv = u * (RHO - w) - v
    dw = u * v - BETA * w
    
    # 2. Jacobian evaluated at current state
    J = np.array([
        [-SIGMA - K_C, SIGMA, 0, K_C, 0, 0],
        [RHO - z, -1, -x, 0, 0, 0],
        [y, x, -BETA, 0, 0, 0],
        [K_C, 0, 0, -SIGMA - K_C, SIGMA, 0],
        [0, 0, 0, RHO - w, -1, -u],
        [0, 0, 0, v, u, -BETA]
    ], dtype=np.float64)
    
    # 3. Deviation derivatives (J * w1, J * w2)
    w1 = state[6:12]
    w2 = state[12:18]
    
    dw1 = J @ w1
    dw2 = J @ w2
    
    return np.concatenate(([dx, dy, dz, du, dv, dw], dw1, dw2))

def compute_sali(start_state, t_max=10.0, steps=100):
    """
    Computes log10(SALI) for a given starting state over time t_max.
    """
    # Create two random orthonormal deviation vectors
    vecs = np.random.randn(2, 6)
    q, r = np.linalg.qr(vecs.T)
    w1_0 = q[:, 0]
    w2_0 = q[:, 1]
    
    current_state = np.zeros(18)
    current_state[0:6] = start_state
    current_state[6:12] = w1_0
    current_state[12:18] = w2_0
    
    time_points = np.linspace(0, t_max, steps)
    dt = time_points[1] - time_points[0]
    
    min_sali = 1.0 # Initial bounded upper limit
    
    for i in range(len(time_points)-1):
        t_span = (time_points[i], time_points[i+1])
        res = solve_ivp(coupled_lorenz_with_deviations, t_span, current_state, method='RK45', rtol=1e-6, atol=1e-6)
        
        # Get state at end of interval
        step_final = res.y[:, -1]
        
        # Extract and normalize deviation vectors
        w1 = step_final[6:12]
        w2 = step_final[12:18]
        
        n1 = np.linalg.norm(w1)
        n2 = np.linalg.norm(w2)
        
        w1_norm = w1 / (n1 + 1e-15)
        w2_norm = w2 / (n2 + 1e-15)
        
        # Calculate SALI
        sali = min(np.linalg.norm(w1_norm + w2_norm), np.linalg.norm(w1_norm - w2_norm))
        if sali < min_sali:
            min_sali = sali
            
        # Re-set normalized vectors into the state for next integration step
        step_final[6:12] = w1_norm
        step_final[12:18] = w2_norm
        current_state = step_final
        
        # Early stopping if completely chaotic (SALI < 1e-12)
        if min_sali < 1e-12:
            break
            
    # Add small epsilon to prevent log10(0)
    return float(np.log10(min_sali + 1e-15))

def worker(state):
    return compute_sali(state, t_max=10.0, steps=100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="Number of LHS points")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    args = parser.parse_args()
    
    print(f"Generating {args.samples} samples using LHS [-40, 40]^6")
    bounds = [-40.0, 40.0]
    
    # LHS Generation
    sampler = qmc.LatinHypercube(d=6)
    sample_raw = sampler.random(n=args.samples)
    states = qmc.scale(sample_raw, [bounds[0]]*6, [bounds[1]]*6)
    
    sali_values = []
    
    print(f"Starting SALI parallel computation with {args.workers} workers...")
    start_t = time.time()
    
    # Parallel map over states
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(worker, states), total=args.samples))
        
    sali_arr = np.array(results)
    
    # Define parameters to track
    params = {
        "SIGMA": SIGMA,
        "RHO": RHO,
        "BETA": BETA,
        "K_C": K_C,
        "t_max": 10.0,
        "integration_steps": 100,
        "samples": args.samples,
        "bounds": bounds,
        "chaotic_percentage": float((sali_arr < -8).mean() * 100)
    }

    # Use Experiment Tracker to version datasets and generator parameters!
    root_dir = os.path.abspath(os.path.join(os.getcwd(), 'experiments'))
    tracker = ExperimentTracker(root_dir, template={"configs": {}})
    run = tracker.create_run(params=params, notes="Offline Phase 3 GALI Surrogate mapping of 2-Oscillator Lorenz.")
    
    # Save directly to the version-controlled artifact registry
    dataset_path = run.get_path("data/sali_dataset.npz")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    np.savez_compressed(dataset_path, states=states, sali=sali_arr)
    
    # Save a local symlink-equivalent copy so `train_surrogate.py` has a default target
    np.savez_compressed("sali_dataset.npz", states=states, sali=sali_arr)
    
    print(f"\nFinished in {time.time()-start_t:.2f}s. Saved artifact to {dataset_path}")
    print(f"SALI Metric Range: [{sali_arr.min():.4f}, {sali_arr.max():.4f}]")
    print(f"Chaotic percentage (log10(SALI) < -8): {params['chaotic_percentage']:.2f}%")

if __name__ == "__main__":
    main()
