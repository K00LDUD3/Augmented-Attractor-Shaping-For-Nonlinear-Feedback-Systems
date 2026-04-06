import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import uuid
import time
from datetime import datetime
import onnx
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(r"C:\CustomTools")
from ExperimentTracker import ExperimentTracker

class SALISurrogate(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256):
        super(SALISurrogate, self).__init__()
        # Deeper MLP to capture highly non-linear 6D mapping for dense Phase 3 LHS mappings.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def load_latest_dataset(experiments_dir="experiments"):
    """Loads the latest dataset (CSV or NPZ) from the experiments directory."""
    if os.path.exists("sali_dataset.npz"):
        print("Loading legacy sali_dataset.npz...")
        data = np.load("sali_dataset.npz")
        return data['states'], data['sali']
    
    # Check for the newest experiment run
    runs = sorted([d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))], reverse=True)
    for run in runs:
        csv_path = os.path.join(experiments_dir, run, "gali_dataset.csv")
        if os.path.exists(csv_path):
            print(f"Loading Multi-Resolution dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            states = df[['x1','y1','z1','x2','y2','z2']].values
            sali = df['gali2'].values
            
            # Load associated params.json for monolithic logging
            params_path = os.path.join(experiments_dir, run, "params.json")
            metadata = {}
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    metadata = json.load(f)
            
            # Convert GALI to log10 if not already (legacy compatibility)
            if sali.max() > 0 and sali.min() >= 0:
                sali = np.log10(sali + 1e-15)
            return states, sali, metadata
            
    raise FileNotFoundError("No valid dataset found in 'experiments/' or current directory.")

def export_and_benchmark(model, run_path, sali_min, sali_max):
    """Exports to ONNX and benchmarks inference across all available device/runtime permutations."""
    model.eval()
    dummy_input = torch.randn(1, 6)
    onnx_path = os.path.join(run_path, "gali_surrogate.onnx")
    
    # Export to ONNX
    torch.onnx.export(
        model.cpu(), dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=15
    )
    
    results = {}
    iters = 500
    
    # Permutations: [Runtime] x [Device]
    # Runtimes: PyTorch, ONNX
    # Devices: CPU, CUDA
    
    devices = ['cpu']
    if torch.cuda.is_available(): devices.append('cuda')
    
    print("\n--- Phase 3 Cross-Platform Benchmarks (Batch=1) ---")
    print(f"{'Runtime':<10} | {'Device':<10} | {'Latency (ms)':<15} | {'Throughput (FPS)':<15}")
    print("-" * 60)
    
    # 1. PyTorch Benchmarks
    for device_name in devices:
        device = torch.device(device_name)
        model.to(device)
        x = dummy_input.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10): _ = model(x)
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters): 
                _ = model(x)
                if device_name == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        
        latency = (end - start) / iters * 1000
        fps = 1000 / (latency + 1e-15)
        print(f"{'PyTorch':<10} | {device_name:<10} | {latency:<15.4f} | {fps:<15.2f}")
        results[f"torch_latency_{device_name}_ms"] = latency
        results[f"torch_fps_{device_name}"] = fps

    # 2. ONNX Benchmarks
    avail_ort = ort.get_available_providers()
    ort_providers = [('cpu', 'CPUExecutionProvider')]
    if 'CUDAExecutionProvider' in avail_ort: 
        ort_providers.append(('cuda', 'CUDAExecutionProvider'))
        
    for device_name, provider in ort_providers:
        try:
            sess = ort.InferenceSession(onnx_path, providers=[provider])
            input_name = sess.get_inputs()[0].name
            x_np = dummy_input.numpy()
            
            # Warmup
            for _ in range(10): sess.run(None, {input_name: x_np})
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iters):
                sess.run(None, {input_name: x_np})
            end = time.perf_counter()
            
            latency = (end - start) / iters * 1000
            fps = 1000 / (latency + 1e-15)
            print(f"{'ONNX':<10} | {device_name:<10} | {latency:<15.4f} | {fps:<15.2f}")
            results[f"onnx_latency_{device_name}_ms"] = latency
            results[f"onnx_fps_{device_name}"] = fps
        except Exception as e:
            print(f"{'ONNX':<10} | {device_name:<10} | ERROR: {str(e)[:15]}... | N/A")
            
    return results

def train_surrogate(epochs_override=None, dataset_path=None):
    print("Initializing Phase 3 Surrogate Training...")
    try:
        if dataset_path:
            if dataset_path.endswith(".npz"):
                data = np.load(dataset_path)
                states = data['states']
                sali = data['sali']
            else:
                df = pd.read_csv(dataset_path)
                states = df[['x1','y1','z1','x2','y2','z2']].values
                sali = df['gali2'].values
            
            if sali.max() > 0: sali = np.log10(sali + 1e-15)
            ds_metadata = {"uid": os.path.basename(os.path.dirname(dataset_path))}
        else:
            states, sali, ds_metadata = load_latest_dataset()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Pre-processing
    # States are bounded [-40, 40]. Normalize them to roughly [-1, 1]
    states_normalized = states / 40.0
    
    # SALI Targets are already log10 (e.g. 0 down to -15). 
    # Let's shift and scale them slightly so the network learns easier.
    # Typically, -15 means maximum chaos, 0 means maximum order.
    # We can normalize them to [0, 1] for easier loss tracking.
    sali_min, sali_max = sali.min(), sali.max()
    print(f"Original SALI Bounds: [{sali_min:.4f}, {sali_max:.4f}]")
    
    # Normalize targets to [0, 1]
    sali_normalized = (sali - sali_min) / (sali_max - sali_min + 1e-8)
    
    # Convert to Tensors
    X = torch.tensor(states_normalized, dtype=torch.float32)
    y = torch.tensor(sali_normalized, dtype=torch.float32).unsqueeze(1)
    
    # Split Train/Val (80/20)
    indices = torch.randperm(X.size(0))
    split = int(0.8 * X.size(0))
    
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]
    
    batch_size = 512 
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup ExperimentTracker
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments'))
    tracker = ExperimentTracker(root_dir, template={"configs": {}})
    
    hyperparams = {
        "dataset_size": X.size(0),
        "split_ratio": 0.8,
        "batch_size": batch_size,
        "hidden_dim": 512,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "device": str(device),
        "sali_min": float(sali_min),
        "sali_max": float(sali_max),
        "source_dataset": ds_metadata.get("uid", "unknown"),
        "physical_constants": ds_metadata.get("physical_constants", {}),
        "domain_bounds": ds_metadata.get("domain_bounds", {})
    }
    
    run = tracker.create_run(params=hyperparams, notes="Surrogate MLP for resolving Offline GALI mappings.")
    
    print(f"Training on device: {device} | Tracker UID: {run.uid}")
    
    model = SALISurrogate(hidden_dim=hyperparams["hidden_dim"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    epochs = epochs_override if epochs_override else hyperparams["epochs"]
    best_val_loss = float('inf')
    best_model_path = run.get_path("data/sali_surrogate.pth")
    
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_maes = []
    epoch_val_accuracies = []
    
    pbar = tqdm(range(epochs), desc="Training Surrogate")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(X_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                # Accuracy tracking in true coordinate space
                preds_true = (preds * (float(sali_max) - float(sali_min))) + float(sali_min)
                y_true = (batch_y * (float(sali_max) - float(sali_min))) + float(sali_min)
                val_mae += torch.abs(preds_true - y_true).sum().item()
                
        val_loss /= len(X_val)
        val_mae /= len(X_val)
        
        # Calculate Percentage Accuracy based on normalized error relative to the bounds
        val_accuracy = max(0.0, 100.0 * (1.0 - (val_mae / (float(sali_max) - float(sali_min) + 1e-8))))
        
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_maes.append(val_mae)
        epoch_val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save Model, Min, Max for un-normalization during inference
            state_dict = {
                'model_state_dict': model.state_dict(),
                'sali_min': float(sali_min),
                'sali_max': float(sali_max),
                'state_max': 40.0,
                'system_metadata': ds_metadata # Monolithic provenance
            }
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(state_dict, best_model_path)
            # Create a local symlink-equivalent copy for easy access
            torch.save(state_dict, "sali_surrogate.pth")
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            pbar.set_postfix({
                "Train": f"{train_loss:.6f}", 
                "Val": f"{val_loss:.6f}", 
                "Acc": f"{val_accuracy:.2f}%"
            })
            
    print(f"Training Complete. Best Val Accuracy: {max(epoch_val_accuracies):.2f}%")
    print(f"Surrogate securely version-controlled to: {best_model_path}")
    
    # Generate Training Charts
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_train_losses, label="Train Loss (MSE)")
    plt.plot(epoch_val_losses, label="Val Loss (MSE)")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Surrogate Iterative Convergence")
    plt.legend()
    plt.savefig(run.get_path("logs/loss_curve.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_val_accuracies, label="Validation Accuracy (%)", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Surrogate Spatial Mapping Accuracy")
    plt.legend()
    plt.savefig(run.get_path("logs/accuracy_curve.png"))
    plt.close()
    
    
    # ----------------------------------------------------
    # PHASE 3: ONNX Export & Multi-Platform Benchmarking
    # ----------------------------------------------------
    print("\nOptimizing Surrogate for Real-Time Control Deployment...")
    onnx_metrics = export_and_benchmark(model, run.path, sali_min, sali_max)
    
    # Final Metrics Update
    metrics = {
        "final_val_loss": best_val_loss,
        "final_train_loss": train_loss,
        "final_val_mae": max(epoch_val_maes),  # Using last or best
        "final_val_accuracy": max(epoch_val_accuracies),
        **onnx_metrics
    }
    
    with open(run.get_path("logs/metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nFinalized Metadata saved to: {run.get_path('logs/metrics.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    
    train_surrogate(epochs_override=args.epochs, dataset_path=args.dataset)
