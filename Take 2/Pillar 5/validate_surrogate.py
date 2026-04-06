import numpy as np
import torch
import time
from train_surrogate import SALISurrogate
from generate_dataset import compute_sali
from scipy.stats import qmc

def main():
    print("Loading PyTorch Surrogate Model for Validation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and normalization limits from the symlink proxy
    checkpoint = torch.load("sali_surrogate.pth", map_location=device)
    model = SALISurrogate(hidden_dim=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    sali_min = checkpoint['sali_min']
    sali_max = checkpoint['sali_max']
    state_max = checkpoint['state_max']
    
    # Create 5 brand new random points via LHS bounds to avoid training data overlap
    sampler = qmc.LatinHypercube(d=6)
    test_samples = qmc.scale(sampler.random(n=5), [-40.0]*6, [40.0]*6)
    
    print("\n--- Running Dynamic Validation ---\n")
    
    for i, state in enumerate(test_samples):
        # 1. Ground Truth Mathematical Integration (ODE)
        print(f"Test {i+1}: Numerically integrating 18D Jacobian equations...")
        t0 = time.time()
        true_sali = compute_sali(state, t_max=10.0, steps=100)
        t_math = time.time() - t0
        
        # 2. Neural Network Surrogate Feed-Forward (MLP)
        t0 = time.time()
        tensor_state = torch.tensor(state / state_max, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_sali_norm = model(tensor_state).item()
            
        # Un-normalize prediction to true log10(SALI) boundaries
        pred_sali = (pred_sali_norm * (sali_max - sali_min)) + sali_min
        t_net = time.time() - t0
        
        error = abs(true_sali - pred_sali)
        speedup = t_math / (t_net + 1e-9)
        
        print(f"   Ground Truth (log10): {true_sali:.4f}  | Time: {t_math*1000:.2f} ms")
        print(f"   Surrogate Prediction: {pred_sali:.4f}  | Time: {t_net*1000:.2f} ms")
        print(f"   Absolute Error      : {error:.4f}")
        print(f"   Speedup Factor      : {speedup:.1f}x\n")

if __name__ == "__main__":
    main()
