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

# Import generator logic for Phase 2 Augmentation
from generate_dataset_v2 import worker

# --- Pillar 5 Surrogate Architecture ---
class SALISurrogate(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=512):
        super(SALISurrogate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.net(x)

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

def calculate_accuracy_pct(mae):
    """Maps MAE in log-space [-15, 0] to a percentage. 0 MAE = 100%."""
    return max(0.0, min(100.0, (1.0 - (mae / 15.0)) * 100.0))

def load_latest_dataset():
    """Hybrid Discovery: Searches Pillar 5 and Root experiments."""
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    ]
    all_runs = []
    for p in search_paths:
        if os.path.exists(p):
            all_runs.extend([os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])
    all_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    for run in all_runs:
        csv = os.path.join(run, "gali_dataset.csv")
        if os.path.exists(csv):
            print(f"Loading survey: {csv}")
            return pd.read_csv(csv), {"uid": os.path.basename(run)}
    raise FileNotFoundError("No dataset found.")

def export_and_benchmark(model, run_path, prefix=""):
    """Exhaustive Stage Dual-Benchmarking Suite."""
    model.eval()
    dummy = torch.randn(1, 6)
    onnx_file = f"{prefix}surrogate.onnx"
    onnx_path = os.path.join(run_path, "data", onnx_file)
    
    torch.onnx.export(model.cpu(), dummy, onnx_path, input_names=['input'], output_names=['output'], opset_version=15)
    
    results = {}
    iters = 200
    devices = ['cpu']
    if torch.cuda.is_available(): devices.append('cuda')
    
    # PyTorch Bench
    for dev_name in devices:
        dev = torch.device(dev_name)
        model.to(dev)
        x = dummy.to(dev)
        with torch.no_grad():
            for _ in range(10): _ = model(x)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters): 
                _ = model(x)
                if dev_name == 'cuda': torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / iters * 1000
        print(f"  [{prefix.upper()}][Torch][{dev_name.upper()}] Latency: {lat:.4f} ms")
        results[f"{prefix}torch_latency_{dev_name}_ms"] = lat

    # ONNX Bench
    providers = [('cpu', 'CPUExecutionProvider')]
    if 'CUDAExecutionProvider' in ort.get_available_providers(): providers.append(('cuda', 'CUDAExecutionProvider'))
    for dev_name, prov in providers:
        sess = ort.InferenceSession(onnx_path, providers=[prov])
        x_np = dummy.numpy()
        for _ in range(10): sess.run(None, {'input': x_np})
        t0 = time.perf_counter()
        for _ in range(iters): sess.run(None, {'input': x_np})
        lat = (time.perf_counter() - t0) / iters * 1000
        print(f"  [{prefix.upper()}][ONNX ][{dev_name.upper()}] Latency: {lat:.4f} ms")
        results[f"{prefix}onnx_latency_{dev_name}_ms"] = lat
    return results

def train_surrogate_v2(epochs=100, dataset_path=None):
    print("\n--- AAS IMMACULATE DUAL-STAGE PIPELINE (v2.1) ---")
    df = pd.read_csv(dataset_path) if dataset_path else load_latest_dataset()[0]
    
    sali = df['gali2'].values
    if sali.max() > 0: sali = np.log10(sali + 1e-15)
    s_min, s_max = sali.min(), sali.max()
    X = torch.tensor(df[['x1','y1','z1','x2','y2','z2']].values / 40.0, dtype=torch.float32)
    y = torch.tensor((sali - s_min) / (s_max - s_min + 1e-8), dtype=torch.float32).unsqueeze(1)
    
    core_idx = df.index[df['is_core'] == 1].tolist()
    np.random.shuffle(core_idx)
    val_idx = core_idx[:int(len(core_idx)*0.1)] # 10% of Core
    train_idx = list(set(range(len(df))) - set(val_idx))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = ExperimentTracker(os.path.join(os.path.dirname(__file__), 'experiments')).create_run(params={
        "epochs": epochs,
        "physical_constants": {
            "SIGMA": 10.0, "RHO": 28.0, "BETA": 8/3, "K_C": 2.5,
            "KP_BASELINE": 30.0, "system": "2-Coupled Lorenz"
        }
    }, notes="Dual-Stage Refinement")
    model = SALISurrogate().to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    # PHASE A: PRE-REFINEMENT
    print(f"\n[PHASE A] Starting Pre-Refinement Stage...")
    history_pre = {"loss": [], "mae": [], "acc": []}
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    for _ in tqdm(range(epochs), desc="Pre-Refining"):
        model.train()
        l_sum = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = loss_fn(model(bx), by); loss.backward(); opt.step()
            l_sum += loss.item()
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(device)).cpu()
            mae = torch.abs(preds - y_val).mean().item()
            history_pre["loss"].append(l_sum/len(loader))
            history_pre["mae"].append(mae)
            history_pre["acc"].append(calculate_accuracy_pct(mae))

    print(f"Phase A Accuracy: {history_pre['acc'][-1]:.2f}%")
    pre_bench = export_and_benchmark(model, run.path, "pre_")
    torch.save(model.state_dict(), run.get_path("data/sali_surrogate_pre.pth"))

    # PHASE B: UNCERTAINTY INJECTION
    print("\n[PHASE B] Adaptive GALI Refinement...")
    model.eval()
    with torch.no_grad(): errs = np.abs(model(X_train.to(device)).cpu().numpy().flatten() - y_train.numpy().flatten())
    seeds = states = df[['x1','y1','z1','x2','y2','z2']].values[train_idx][np.where(errs > np.percentile(errs, 92))[0]]
    aug_p, aug_g = [], []
    for s in tqdm(seeds[:2000], desc="Solving"): 
        pt = s + np.random.normal(0, 0.4, size=6)
        aug_p.append(pt); aug_g.append(worker(pt))
    X_aug = torch.cat([X_train, torch.tensor(np.array(aug_p)/40.0, dtype=torch.float32)])
    y_aug = torch.cat([y_train, torch.tensor((np.log10(np.array(aug_g)+1e-15) - s_min)/(s_max-s_min+1e-8), dtype=torch.float32).unsqueeze(1)])

    # PHASE C: POST-REFINEMENT
    print(f"\n[PHASE C] Starting Post-Refinement Fine-Tuning...")
    history_post = {"loss": [], "mae": [], "acc": []}
    aug_loader = DataLoader(TensorDataset(X_aug, y_aug), batch_size=512, shuffle=True)
    for _ in tqdm(range(epochs), desc="Post-Refining"):
        model.train()
        l_sum = 0
        for bx, by in aug_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = loss_fn(model(bx), by); loss.backward(); opt.step()
            l_sum += loss.item()
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(device)).cpu()
            mae = torch.abs(preds - y_val).mean().item()
            history_post["loss"].append(l_sum/len(aug_loader))
            history_post["mae"].append(mae)
            history_post["acc"].append(calculate_accuracy_pct(mae))

    print(f"Phase C Accuracy: {history_post['acc'][-1]:.2f}%")
    post_bench = export_and_benchmark(model, run.path, "post_")
    torch.save(model.state_dict(), run.get_path("data/sali_surrogate_post.pth"))

    # FINAL EXPORT
    metrics = {
        "pre_val_acc": history_pre["acc"][-1], "post_val_acc": history_post["acc"][-1],
        "acc_delta": history_post["acc"][-1]-history_pre["acc"][-1], **pre_bench, **post_bench
    }
    with open(run.get_path("logs/metrics.json"), "w") as f: json.dump(metrics, f, indent=4)
    
    # PLOTTING SUITE
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(history_pre["loss"], 'o-', label="Pre"); plt.plot(history_post["loss"], 's-', label="Post"); plt.yscale('log'); plt.title("Loss Delta"); plt.legend()
    plt.subplot(1,2,2); plt.plot(history_pre["acc"], 'o-', label="Pre"); plt.plot(history_post["acc"], 's-', label="Post"); plt.title("Accuracy Delta (%)"); plt.legend()
    plt.savefig(run.get_path("logs/refinement_audit.png"))
    print(f"\nImmaculate metadata saved to: {run.path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    train_surrogate_v2(epochs=args.epochs, dataset_path=args.dataset)
