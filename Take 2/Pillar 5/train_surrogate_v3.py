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
import pandas as pd
from tqdm import tqdm
import argparse
import platform

# --- Pillar 5 Surrogate Architecture (v3: Residual) ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.net(x))

class SALISurrogate(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=512):
        super(SALISurrogate, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.downsample = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res3 = ResidualBlock(hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.downsample(x)
        x = self.res3(x)
        return self.output_layer(x)

class ExperimentTracker:
    def __init__(self, root_dir):
        self.root = root_dir
        os.makedirs(root_dir, exist_ok=True)
    def create_run(self, params=None, notes=""):
        uid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:8]
        path = os.path.join(self.root, uid)
        os.makedirs(os.path.join(path, "configs"), exist_ok=True)
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)
        if params:
            with open(os.path.join(path, "configs/params.json"), "w") as f:
                json.dump({"uid": uid, "notes": notes, **params}, f, indent=4)
        return Run(uid, path)

class Run:
    def __init__(self, uid, path):
        self.uid, self.path = uid, path
    def get_path(self, rel): return os.path.join(self.path, rel)

def weighted_mse_loss(pred, target, sali_norm, s_min, s_max):
    """Refined Loss prioritizes Transition Zones [0.1, 0.4]"""
    # Inverse-normalization to check regimes
    log_sali = sali_norm * (s_max - s_min + 1e-8) + s_min
    
    weights = torch.ones_like(log_sali)
    # Transition: 0.1 < GALI < 0.4 => -1.0 < log10(GALI) < -0.397
    mask_trans = (log_sali > -1.0) & (log_sali < -0.3979)
    weights[mask_trans] = 5.0
    
    # Stable: GALI > 0.4 => log10(GALI) > -0.397
    mask_stable = (log_sali >= -0.3979)
    # weights[mask_stable] = 3.0 # For V3.1 - Pillar 5\experiments\2026-04-09_18-09-00_c5f67923
    weights[mask_stable] = 3.75 # For V3.2 - 
    
    # Deep Chaos takes base weight 1.0
    return torch.mean(weights * (pred - sali_norm)**2)

def calculate_accuracy_pct(mae):
    return max(0.0, min(100.0, (1.0 - (mae / 15.0)) * 100.0))

def train_surrogate_v3(epochs=100, dataset_path=None):
    print("\n--- AAS SURROGATE PIPELINE v3 (Residual-Weighted) ---")
    if not dataset_path: raise ValueError("Dataset path must be specified for v3 execution.")
    
    df = pd.read_csv(dataset_path)
    sali = df['gali2'].values
    if sali.max() > 0: sali = np.log10(sali + 1e-15)
    s_min, s_max = sali.min(), sali.max()
    
    # Normalization
    X = torch.tensor(df[['x1','y1','z1','x2','y2','z2']].values / 40.0, dtype=torch.float32)
    y = torch.tensor((sali - s_min) / (s_max - s_min + 1e-8), dtype=torch.float32).unsqueeze(1)
    
    # EXACT Dataset Split Reproduction (Pillar 5 / v2 logic)
    core_idx = df.index[df['is_core'] == 1].tolist()
    np.random.shuffle(core_idx)
    val_idx = core_idx[:int(len(core_idx)*0.1)]
    train_idx = list(set(range(len(df))) - set(val_idx))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} | OS: {platform.system()} {platform.release()}")
    
    # Comprehensive Hyperparameter Logging
    params = {
        "architecture": "Residual MLP (3 Blocks)",
        "hidden_dim": 512,
        "epochs": epochs,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "loss_function": "Stability-Weighted MSE",
        "weight_coefficients": {"transition": 5.0, "stable": 3.0, "chaos": 1.0},
        "dataset": os.path.abspath(dataset_path),
        "data_normalization": {"s_min": float(s_min), "s_max": float(s_max)},
        "physical_constants": {
            "SIGMA": 10.0, "RHO": 28.0, "BETA": 8/3, "K_C": 2.5,
            "system": "2-Coupled Lorenz"
        },
        "system_info": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device)
        }
    }
    
    tracker = ExperimentTracker(os.path.join(os.path.dirname(__file__), 'experiments'))
    run = tracker.create_run(params=params, notes="Residual Architecture + Weighted Transition Loss (v3)")
    
    model = SALISurrogate().to(device)
    opt = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    
    history = {"loss": [], "mae": [], "acc": []}
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    
    print(f"\n[v3 Engine] Starting Training for {epochs} Epochs...")
    for epoch in range(epochs):
        model.train()
        l_sum = 0
        for bx, by in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bx)
            loss = weighted_mse_loss(preds, by, by, s_min, s_max)
            loss.backward()
            opt.step()
            l_sum += loss.item()
            
        model.eval()
        with torch.no_grad():
            v_preds = model(X_val.to(device)).cpu()
            mae = torch.abs(v_preds - y_val).mean().item()
            acc = calculate_accuracy_pct(mae)
            history["loss"].append(l_sum / len(loader))
            history["mae"].append(mae)
            history["acc"].append(acc)
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {history['loss'][-1]:.6f} | Val Accuracy: {acc:.2f}%")

    # Save v3 Model and Artifacts
    torch.save({
        "model_state_dict": model.state_dict(),
        "sali_min": s_min,
        "sali_max": sali_max if 'sali_max' in locals() else s_max
    }, run.get_path("data/sali_surrogate_v3.pth"))
    
    with open(run.get_path("logs/metrics.json"), "w") as f:
        json.dump({
            "final_val_acc": history["acc"][-1],
            "min_loss": min(history["loss"]),
            "training_time_sec": time.time() - time.mktime(datetime.strptime(run.uid.split('_')[0] + "_" + run.uid.split('_')[1], "%Y-%m-%d_%H-%M-%S").timetuple())
        }, f, indent=4)

    # Fidelity Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1); plt.plot(history['loss'], label='Weighted MSE'); plt.title("v3 Training Loss"); plt.legend()
    plt.subplot(1,2,2); plt.plot(history['acc'], label='Pct Accuracy'); plt.title("v3 Validation Accuracy"); plt.legend()
    plt.savefig(run.get_path("logs/v3_training_audit.png"))
    
    print(f"\n--- v3 SUCCESS ---")
    print(f"Experiment: {run.uid}")
    print(f"Path: {run.path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    train_surrogate_v3(epochs=args.epochs, dataset_path=args.dataset)
