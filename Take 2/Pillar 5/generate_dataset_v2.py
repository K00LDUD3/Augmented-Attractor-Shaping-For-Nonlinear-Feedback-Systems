import os
import sys
import numpy as np
import pandas as pd
import json
import time
import argparse
import uuid
from datetime import datetime
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

# Ensure Pillar 4 is in path for environment access
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pillar 4"))

# --- REPRODUCIBILITY CONSTANTS ---
SIGMA = 10.0
RHO = 28.0
BETA = 8/3
K_C = 2.5
DT = 0.005
KP_BASELINE = 30.0 # Standardized Pillar 4 Constrained Gain

def coupled_lorenz_vm(t, state):
    """
    18D Variational Equation System for 2-Oscillator Coupled Lorenz.
    0-5: Physical Coordinates (x,y,z, u,v,w)
    6-11: Deviation Vector 1
    12-17: Deviation Vector 2
    """
    x, y, z, u, v, w = state[0:6]
    
    # 1. Physical ODEs (Base State Space)
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

def compute_gali2_vm(x0, t_max=12.0, steps=120):
    """Integrates VM to find GALI2 alignment index."""
    # Orthogonal initial vectors
    vecs = np.random.randn(2, 6)
    q, r = np.linalg.qr(vecs.T)
    
    state = np.zeros(18)
    state[0:6] = x0
    state[6:12] = q[:, 0]
    state[12:18] = q[:, 1]
    
    dt_step = t_max / steps
    min_gali = 1.0
    
    for i in range(steps):
        sol = solve_ivp(coupled_lorenz_vm, (0, dt_step), state, method='RK45', rtol=1e-6, atol=1e-6)
        state = sol.y[:, -1]
        
        # Normalize and compute alignment
        w1 = state[6:12] / (np.linalg.norm(state[6:12]) + 1e-15)
        w2 = state[12:18] / (np.linalg.norm(state[12:18]) + 1e-15)
        
        # GALI2 calculation (norm of wedge product via SALI proxy)
        sali = min(np.linalg.norm(w1 + w2), np.linalg.norm(w1 - w2))
        min_gali = min(min_gali, sali)
        
        # Reset normalized vectors
        state[6:12] = w1
        state[12:18] = w2
        
        if min_gali < 1e-12: break
            
    return float(min_gali)

def worker(x0):
    return compute_gali2_vm(x0)

class GALIProductionGenerator:
    def __init__(self, n_total=25000, split=0.8, refine_n=2500):
        self.n_total = n_total
        self.split = split
        self.refine_n = refine_n


    def generate(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
        
        run_uid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:8]
        run_path = os.path.join(output_dir, run_uid)
        os.makedirs(run_path, exist_ok=True)

        # PASS 1: Spatial Multi-Resolution LHS
        n_core = int(self.n_total * self.split)
        n_shell = self.n_total - n_core
        
        print(f"Pass 1: Generating {n_core} Core and {n_shell} Shell samples...")
        s_core = qmc.scale(qmc.LatinHypercube(d=6).random(n=n_core), [-40]*6, [40]*6)
        
        # Rejection sampling for shell
        s_shell_list = []
        lhc_shell = qmc.LatinHypercube(d=6)
        while len(s_shell_list) < n_shell:
            batch = qmc.scale(lhc_shell.random(n=1000), [-60]*6, [60]*6)
            mask = np.any(np.abs(batch) > 40, axis=1)
            s_shell_list.extend(batch[mask])
        s_shell = np.array(s_shell_list[:n_shell])
        initial_points = np.vstack([s_core, s_shell])

        # COMPUTE PASS 1
        print("Computing GALI Survey Data...")
        gali_results = []
        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(worker, p) for p in initial_points]
            for f in tqdm(futures, total=len(futures), desc="Pass 1 (Survey)"):
                gali_results.append(f.result())
        
        df = pd.DataFrame(initial_points, columns=['x1','y1','z1','x2','y2','z2'])
        df['gali2'] = gali_results
        df['is_core'] = [1]*n_core + [0]*n_shell

        # PASS 2: SSM Adaptive Refinement (High-Gradient Transition Zones)
        # Target GALI values in [0.1, 0.4] strictly within the Core attractor region
        core_only = df[df['is_core'] == 1]
        boundaries = core_only[(core_only['gali2'] > 0.1) & (core_only['gali2'] < 0.4)]
        
        if len(boundaries) > 0 and self.refine_n > 0:
            print(f"Pass 2: Identifying SSM Manifolds in Core. Refining {self.refine_n} points...")
            refine_points = []
            for _ in range(self.refine_n):
                parent = boundaries.sample(1).values[0][:6]
                refine_points.append(parent + np.random.normal(0, 1.2, size=6))
            
            with ProcessPoolExecutor() as ex:
                futures = [ex.submit(worker, p) for p in refine_points]
                refine_gali = []
                for f in tqdm(futures, total=len(futures), desc="Pass 2 (SSM Refine)"):
                    refine_gali.append(f.result())
            
            df_refine = pd.DataFrame(refine_points, columns=['x1','y1','z1','x2','y2','z2'])
            df_refine['gali2'] = refine_gali
            df_refine['is_core'] = 1 # Force core labeling since seeds were core-restricted
            df = pd.concat([df, df_refine], ignore_index=True)


        # SAVE EVERYTHING (Universal Reproducibility - MONOLITHIC PARAM LOGGING)
        df.to_csv(os.path.join(run_path, "gali_dataset.csv"), index=False)
        params = {
            "uid": run_uid, 
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df), 
            "sampling": {
                "base_samples": self.n_total,
                "split_ratio": self.split,
                "n_core": n_core,
                "n_shell": n_shell,
                "refine_n": self.refine_n
            },
            "physical_constants": {
                "SIGMA": SIGMA, "RHO": RHO, "BETA": BETA, "K_C": K_C,
                "system": "2-Coupled Lorenz",
                "variational_dim": 18
            },
            "simulation_settings": {
                "t_max": 12.0, "steps": 120, "solver": "RK45", "rtol": 1e-6, "atol": 1e-6, "DT": DT
            },
            "control_context": {
                "Kp_baseline": KP_BASELINE, "target": [0,0,0,0,0,0], "pillar_4_ref": "Constrained Gain"
            },
            "domain_bounds": {
                "shell_bounds": [-60.0, 60.0], "core_bounds": [-40.0, 40.0]
            }
        }
        with open(os.path.join(run_path, "params.json"), "w") as f: json.dump(params, f, indent=4)
        
        # Verify and Plot
        self.plot_verification(df, run_path)
        print(f"Production run complete: {run_path}")
        return run_path

    def plot_verification(self, df, run_path):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_df = df.sample(min(3000, len(df)))
        sc = ax.scatter(plot_df['x1'], plot_df['y1'], plot_df['z1'], c=plot_df['gali2'], cmap='viridis', s=2, alpha=0.6)
        plt.colorbar(sc, label='GALI2 (Stability)')
        ax.set_title(f"Multi-Resolution GALI Atlas (N={len(df)})")
        plt.savefig(os.path.join(run_path, "atlas_verification.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=25000)
    parser.add_argument("--refine", type=int, default=2500)
    args = parser.parse_args()
    
    gen = GALIProductionGenerator(n_total=args.samples, refine_n=args.refine)
    gen.generate()
