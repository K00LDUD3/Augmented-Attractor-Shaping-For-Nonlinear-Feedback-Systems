import os
import torch
import numpy as np
import time
import json
import argparse
from train_surrogate_v2 import SALISurrogate
from generate_dataset_v2 import worker
from scipy.stats import qmc

def validate(model_path):
    print(f"\n--- AAS SURROGATE VALIDATION (v2.1 Resilience) ---")
    print(f"Loading Model: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize 512-dim architecture (v2 Standard)
    model = SALISurrogate(input_dim=6, hidden_dim=512).to(device)
    
    # Handle both Raw state_dict and Metadata-Wrappers
    if 'model_state_dict' in checkpoint:
        print("   Status: Metadata-wrapped checkpoint detected.")
        model.load_state_dict(checkpoint['model_state_dict'])
        s_min, s_max = checkpoint['sali_min'], checkpoint['sali_max']
    else:
        print("   Status: Raw state_dict detected. Searching for metadata...")
        model.load_state_dict(checkpoint)
        # Attempt to find params.json in the experiment root
        exp_dir = os.path.dirname(os.path.dirname(model_path))
        params_path = os.path.join(exp_dir, "configs", "params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                # Note: We need to ensure we have the normalization constants
                # For this specific run, we'll assume the standard GALI log-range [-15, 0]
                s_min, s_max = -15.0, 0.0
        else:
            s_min, s_max = -15.0, 0.0
    
    model.eval()
    print(f"   Normalization: Log(GALI) Range [{s_min:.1f}, {s_max:.1f}]")
    
    # Sample 5 truly novel points via LHS
    sampler = qmc.LatinHypercube(d=6)
    test_points = qmc.scale(sampler.random(n=5), [-40.0]*6, [40.0]*6)
    
    print(f"\n{'Test':<6} | {'Ground Truth':<15} | {'Prediction':<15} | {'Error':<10} | {'Speedup':<10}")
    print("-" * 65)
    
    total_speedup = []
    for i, pt in enumerate(test_points):
        # 1. Physics Solve (Ground Truth)
        t0 = time.perf_counter()
        true_gali = worker(pt)
        true_log = np.log10(true_gali + 1e-15)
        t_phys = time.perf_counter() - t0
        
        # 2. Surrogate Inference
        t0 = time.perf_counter()
        x_tensor = torch.tensor(pt / 40.0, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out_norm = model(x_tensor).item()
        
        # Un-normalize
        pred_log = (out_norm * (s_max - s_min + 1e-8)) + s_min
        t_surr = time.perf_counter() - t0
        
        error = abs(true_log - pred_log)
        speedup = t_phys / (t_surr + 1e-12)
        total_speedup.append(speedup)
        
        print(f"#{i+1:<5} | {true_log:<15.4f} | {pred_log:<15.4f} | {error:<10.4f} | {speedup:<10.1f}x")

    print(f"\nValidation Complete. Avg Speedup: {np.mean(total_speedup):.1f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pth checkpoint")
    args = parser.parse_args()
    validate(args.model)
