import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np  # Strictly after torch
import sys
import uuid
import time
from datetime import datetime
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ExperimentTracker import ExperimentTracker
from Logger import Logger

# --- PILLAR 6 CONFIG ---
PILLAR_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join(PILLAR_DIR, "6d_cassm_batch")

# --- REPRODUCIBILITY CONSTANTS (Coupled Lorenz) ---
SIGMA = 10.0
RHO = 28.0
BETA = 8/3
K_C = 2.5
STATE_BOUND = 40.0
T_MAX = 5.0  # Sufficient for transient convergence to manifold
STEPS = 50

PARAMS = {
    "num_samples": 50000,
    "t_max": T_MAX,
    "steps": STEPS,
    "method": "RK45",
    "seed": 42
}

def coupled_lorenz_vm(t, state):
    """18D Variational Equation System for 2-Oscillator Coupled Lorenz."""
    x, y, z, u, v, w = state[0:6]
    
    # 1. Physical ODEs
    dx = SIGMA * (y - x) + K_C * (u - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    du = SIGMA * (v - u) + K_C * (x - u)
    dv = u * (RHO - w) - v
    dw = u * v - BETA * w
    
    # 2. Jacobian Matrix (Open-loop)
    J = np.array([
        [-SIGMA - K_C, SIGMA, 0, K_C, 0, 0],
        [RHO - z, -1, -x, 0, 0, 0],
        [y, x, -BETA, 0, 0, 0],
        [K_C, 0, 0, -SIGMA - K_C, SIGMA, 0],
        [0, 0, 0, RHO - w, -1, -u],
        [0, 0, 0, v, u, -BETA]
    ], dtype=np.float64)
    
    # 3. Tangent Dynamics (J * w)
    w1 = state[6:12]
    w2 = state[12:18]
    dw1 = J @ w1
    dw2 = J @ w2
    
    return np.concatenate(([dx, dy, dz, du, dv, dw], dw1, dw2))

def integrate_ca_ssm_sample(x0):
    """Integrates the 18D system to retrieve the stable fiber-aligned tangents."""
    # Orthogonal initial vectors
    vecs = np.random.randn(2, 6)
    q, r = np.linalg.qr(vecs.T)
    
    state = np.zeros(18)
    state[0:6] = x0
    state[6:12] = q[:, 0]
    state[12:18] = q[:, 1]
    
    dt_step = T_MAX / STEPS
    
    for _ in range(STEPS):
        sol = solve_ivp(coupled_lorenz_vm, (0, dt_step), state, method='RK45', rtol=1e-6, atol=1e-6)
        state = sol.y[:, -1]
        
        # Gram-Schmidt re-orthogonalization (prevents tangent collapse)
        # w1: normalize
        w1 = state[6:12]
        w1_norm = np.linalg.norm(w1)
        if w1_norm > 1e-15:
            w1 = w1 / w1_norm
        state[6:12] = w1
        
        # w2: subtract projection onto w1, then normalize
        w2 = state[12:18]
        w2 = w2 - np.dot(w2, w1) * w1  # Remove w1 component
        w2_norm = np.linalg.norm(w2)
        if w2_norm > 1e-15:
            w2 = w2 / w2_norm
        state[12:18] = w2
        
    return state.astype(np.float32)

def main():
    tracker = ExperimentTracker(BATCH_DIR)
    run = tracker.create_run(params=PARAMS, notes="18D Data Generation for caSSM Identification.")
    
    # Workflow Implementation
    logger = Logger(f=run.get_path("logs/run.log"), sesh_file=run.get_path("logs/.sesh_num"))
    run.copy_file(os.path.abspath(__file__), "configs/")
    
    logger.start_session()
    try:
        logger.log(f"Starting 18D Data Generation | Run: {run.uid}", tag="BEG", level="INFO")
        
        # 1. LHS Sampling
        logger.log(f"Sampling {PARAMS['num_samples']} ICs via LHS...", tag="DEF", level="INFO")
        sampler = qmc.LatinHypercube(d=6, seed=PARAMS["seed"])
        lhc_samples = sampler.random(n=PARAMS["num_samples"])
        initial_points = qmc.scale(lhc_samples, [-STATE_BOUND]*6, [STATE_BOUND]*6)
        
        # 2. Parallel Integration
        logger.log("Beginning parallel 18D integration...", tag="DEF", level="INFO")
        results = []
        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(integrate_ca_ssm_sample, p) for p in initial_points]
            for f in tqdm(futures, total=len(futures), desc="18D Integration"):
                results.append(f.result())
        
        # 3. Saving
        data_matrix = np.array(results)
        states = data_matrix[:, 0:6]
        tangents = data_matrix[:, 6:18]
        
        out_path = run.get_path("data/cassm_raw_data.npz")
        np.savez(out_path, states=states, tangents=tangents)
        
        logger.log(f"Successfully generated 18D dataset at {out_path}", tag="FIN", level="INFO")
        run.add_notes(f"Gen Complete. N={len(states)}. Shape={data_matrix.shape}")
        
    except Exception as e:
        logger.log(f"Data Generation Failed: {e}", tag="DEF", level="ERROR")
        raise
    finally:
        logger.end_session()

if __name__ == "__main__":
    main()
