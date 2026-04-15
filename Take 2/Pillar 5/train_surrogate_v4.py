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

def compute_distribution_accuracy(pred_norm_np, target_norm_np, s_min, s_max, n_bins=200):
    """
    Two distribution-similarity metrics, both in range [0, 100]%:

    1. JS Accuracy  = (1 - JSD_nats / ln2) * 100
       JSD with natural log is bounded by ln(2), so dividing by ln(2) maps [0, ln2] -> [0, 1].
       100% = distributions are identical. 0% = maximally divergent.

    2. Histogram Intersection = sum(min(P, Q)) * 100
       When P and Q are density-normalized histograms (sum=1), this is the overlap fraction.
       100% = perfect overlap. 0% = no overlap.
    """
    # Denormalize back to log-GALI space
    pred_log  = pred_norm_np  * (s_max - s_min + 1e-8) + s_min
    tgt_log   = target_norm_np * (s_max - s_min + 1e-8) + s_min

    bins = np.linspace(s_min - 0.1, min(s_max + 0.1, 1.0), n_bins + 1)

    P, _ = np.histogram(pred_log, bins=bins)
    Q, _ = np.histogram(tgt_log,  bins=bins)

    # Normalize to probability distributions
    P = P.astype(np.float64) + 1e-12   # Laplace smoothing
    Q = Q.astype(np.float64) + 1e-12
    P /= P.sum()
    Q /= Q.sum()

    # JS Divergence (natural log, bounded by ln2)
    M      = 0.5 * (P + Q)
    kl_pm  = np.sum(P * np.log(P / M))   # KL(P||M)
    kl_qm  = np.sum(Q * np.log(Q / M))   # KL(Q||M)
    jsd    = 0.5 * kl_pm + 0.5 * kl_qm  # in nats, in [0, ln2]
    js_acc = max(0.0, (1.0 - jsd / np.log(2)) * 100.0)

    # Histogram Intersection
    hi_acc = float(np.sum(np.minimum(P, Q))) * 100.0

    return js_acc, hi_acc

def build_inv_density_weights(y_norm_np, n_bins=200):
    """
    Computes per-sample inverse-sqrt-density weights from the training label distribution.
    Samples in dense regions (the mode) get low weight; rare samples (tails) get high weight.
    Result is L1-normalized so the mean weight == 1.0.
    """
    counts, bin_edges = np.histogram(y_norm_np, bins=n_bins)
    # Map each sample to its bin
    bin_indices = np.digitize(y_norm_np, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    # Inverse sqrt of count (add 1 to avoid division by zero for empty bins)
    inv_sqrt = 1.0 / np.sqrt(counts[bin_indices].astype(np.float32) + 1.0)
    # Normalize so mean == 1.0
    inv_sqrt /= inv_sqrt.mean()
    return torch.tensor(inv_sqrt, dtype=torch.float32)

def weighted_mse_loss_v4(pred, target, sali_norm, s_min, s_max, inv_density_w):
    """
    v4 Loss: Inverse-density base weights (break mode collapse) +
             Soft sigmoid zone multipliers (smooth gradient, no boundary spikes).
    """
    log_sali = sali_norm * (s_max - s_min + 1e-8) + s_min

    # Soft sigmoid zone multipliers — smooth transitions, no hard mask walls
    # Transition boost: peaks at logGALI = -0.7 (center of [−1.0, −0.4])
    trans_boost = 2.5 * torch.sigmoid(10.0 * (log_sali + 1.0)) * torch.sigmoid(10.0 * (-0.4 - log_sali))
    # Stable boost: activates above logGALI > -0.4
    stable_boost = 5.0 * torch.sigmoid(15.0 * (log_sali + 0.4))
    # Combined zone multiplier (additive on top of 1.0 base)
    zone_mult = 1.0 + trans_boost + stable_boost

    # Final per-sample weight: inverse density * zone multiplier
    w = inv_density_w.to(pred.device) * zone_mult
    return torch.mean(w * (pred - target) ** 2)

def train_surrogate_v4(epochs=100, dataset_path=None, checkpoint_every=20):
    print("\n--- AAS SURROGATE PIPELINE v4 (Inv-Density + Soft Zone Weighting) ---")
    if not dataset_path: raise ValueError("Dataset path must be specified for v4 execution.")
    
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
        "architecture": "Residual MLP (3 Blocks) v4",
        "hidden_dim": 512,
        "epochs": epochs,
        "checkpoint_every_n_epochs": checkpoint_every,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "loss_function": "Inv-Density + Soft Sigmoid Zone Weighting",
        "accuracy_metrics": [
            "JS Accuracy: (1 - JSD_nats / ln2) * 100",
            "Histogram Intersection: sum(min(P,Q)) * 100"
        ],
        "weight_coefficients": {
            "inv_density_bins": 200,
            "transition_soft_peak": 2.5,
            "stable_soft_peak": 5.0,
            "transition_center_logGALI": -0.7,
            "stable_threshold_logGALI": -0.4
        },
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
    run = tracker.create_run(params=params, notes="v4: Inv-Density base + Soft Sigmoid Zone Multipliers")
    
    model = SALISurrogate().to(device)
    opt = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-5)

    # Checkpoint directory for every-20-epoch saves
    ckpt_dir = run.get_path("data/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Precompute per-sample inverse-density weights from training labels
    y_train_np = y_train.numpy().flatten()
    inv_density_w_train = build_inv_density_weights(y_train_np, n_bins=200)
    print(f"  Inv-Density Weights: min={inv_density_w_train.min():.3f}, max={inv_density_w_train.max():.3f}, mean={inv_density_w_train.mean():.3f}")

    # -------------------------------------------------------------------
    # Weight Profile Plot: combined loss weight vs log(GALI)
    # Three components: (1) inv-density, (2) soft zone multiplier, (3) combined
    # -------------------------------------------------------------------
    _log_gali_range = np.linspace(s_min, min(s_max, 1.0), 2000)

    # Inv-density component: map normalised y back to log-GALI, scatter-sort weights
    _y_log_gali = y_train_np * (s_max - s_min + 1e-8) + s_min   # denormalise
    _sort_idx   = np.argsort(_y_log_gali)
    _x_sorted   = _y_log_gali[_sort_idx]
    _w_sorted   = inv_density_w_train.numpy()[_sort_idx]

    # Zone multiplier (analytical, over the full range)
    _lg = torch.tensor(_log_gali_range, dtype=torch.float32)
    _trans_boost  = 2.5 * torch.sigmoid(10.0 * (_lg + 1.0)) * torch.sigmoid(10.0 * (-0.4 - _lg))
    _stable_boost = 5.0 * torch.sigmoid(15.0 * (_lg + 0.4))
    _zone_mult    = (1.0 + _trans_boost + _stable_boost).numpy()

    # Combined weight: we approximate by binning the inv-density weights
    _counts, _bin_edges = np.histogram(_x_sorted, bins=200)
    _inv_w_binned = np.zeros(len(_log_gali_range))
    for _i, _lv in enumerate(_log_gali_range):
        _bi = np.digitize(_lv, _bin_edges) - 1
        _bi = np.clip(_bi, 0, len(_counts) - 1)
        # Mean inv-density weight for samples in this bin
        _mask = (_x_sorted >= _bin_edges[_bi]) & (_x_sorted < _bin_edges[min(_bi + 1, len(_bin_edges) - 1)])
        _inv_w_binned[_i] = _w_sorted[_mask].mean() if _mask.sum() > 0 else 0.0
    _combined = _inv_w_binned * _zone_mult

    fig_w, ax_w = plt.subplots(figsize=(12, 5))
    ax_w.plot(_log_gali_range, _inv_w_binned, color='#636E72', lw=1.2, label='Inv-Density Weight', alpha=0.8)
    ax_w.plot(_log_gali_range, _zone_mult,    color='#FDCB6E', lw=1.2, label='Soft Zone Multiplier', alpha=0.8)
    ax_w.plot(_log_gali_range, _combined,     color='#0984E3', lw=2.0, label='Combined Weight (product)', alpha=0.9)

    # Regime boundaries
    ax_w.axvline(-1.0,  color='#D63031', lw=0.8, ls='--', alpha=0.6, label='Transition start (GALI=0.1)')
    ax_w.axvline(-0.4,  color='#00B894', lw=0.8, ls='--', alpha=0.6, label='Stable start (GALI=0.4)')
    ax_w.axvspan(-1.0, -0.4, color='#FDCB6E', alpha=0.08)

    # GT density overlay (right y-axis)
    ax_density = ax_w.twinx()
    _gt_log = np.log10(df['gali2'].values + 1e-15)
    ax_density.hist(_gt_log, bins=200, density=True, color='#2D3436', alpha=0.12, label='GT Density')
    ax_density.set_ylabel('GT Log-GALI Density', color='#636E72', fontsize=9)
    ax_density.tick_params(axis='y', labelcolor='#636E72')

    ax_w.set_xlabel('Log10(GALI2)')
    ax_w.set_ylabel('Loss Weight')
    ax_w.set_title('v4 Loss Weight Profile vs Log(GALI)', fontsize=13)
    ax_w.legend(frameon=False, fontsize=8, loc='upper left')
    ax_w.set_xlim(s_min, min(s_max + 0.2, 1.0))
    fig_w.tight_layout()
    fig_w.savefig(run.get_path('logs/weight_profile.png'), dpi=200, transparent=True)
    plt.close(fig_w)
    print(f"  [Weight Profile] Saved -> logs/weight_profile.png")
    # -------------------------------------------------------------------


    history = {
        "epoch": [], "loss": [], "mae": [],
        "js_acc": [], "hi_acc": [], "epoch_time_sec": []
    }
    best_js_acc   = -1.0
    best_ckpt_path = None

    # Include inv_density_w in DataLoader so each batch gets its weights
    train_ds = TensorDataset(X_train, y_train, inv_density_w_train)
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    train_start_wall = time.time()
    print(f"\n[v4 Engine] Starting Training for {epochs} Epochs (ckpt every {checkpoint_every})...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        l_sum = 0
        for bx, by, bw in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bx)
            loss = weighted_mse_loss_v4(preds, by, by, s_min, s_max, bw)
            loss.backward()
            opt.step()
            l_sum += loss.item()

        model.eval()
        with torch.no_grad():
            v_preds = model(X_val.to(device)).cpu()
            mae = torch.abs(v_preds - y_val).mean().item()
            epoch_time = time.time() - epoch_start

            # Distribution-similarity accuracy metrics
            js_acc, hi_acc = compute_distribution_accuracy(
                v_preds.numpy().flatten(),
                y_val.numpy().flatten(),
                s_min, s_max
            )

            history["epoch"].append(epoch + 1)
            history["loss"].append(l_sum / len(loader))
            history["mae"].append(mae)
            history["js_acc"].append(js_acc)
            history["hi_acc"].append(hi_acc)
            history["epoch_time_sec"].append(round(epoch_time, 3))

        if (epoch + 1) % max(1, checkpoint_every // 2) == 0:
            print(
                f"  Epoch [{epoch+1}/{epochs}] - Loss: {history['loss'][-1]:.6f} "
                f"| JS Acc: {js_acc:.2f}% | HI Acc: {hi_acc:.2f}% "
                f"| MAE: {mae:.5f} | {epoch_time:.1f}s"
            )

        # Save best model whenever JS accuracy improves
        if js_acc > best_js_acc:
            best_js_acc = js_acc
            best_ckpt_path = os.path.join(ckpt_dir, "ckpt_best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "sali_min": s_min,
                "sali_max": s_max,
                "val_js_acc": js_acc,
                "val_hi_acc": hi_acc,
                "val_mae": mae,
                "loss": history["loss"][-1]
            }, best_ckpt_path)

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "sali_min": s_min,
                "sali_max": s_max,
                "val_js_acc": js_acc,
                "val_hi_acc": hi_acc,
                "val_mae": mae,
                "loss": history["loss"][-1]
            }, ckpt_path)
            print(f"  [Checkpoint] Saved -> {os.path.basename(ckpt_path)}")

    total_train_time = time.time() - train_start_wall

    # Save final v4 model
    final_js_acc = history["js_acc"][-1]
    final_hi_acc = history["hi_acc"][-1]
    torch.save({
        "model_state_dict": model.state_dict(),
        "sali_min": s_min,
        "sali_max": s_max,
        "val_js_acc": final_js_acc,
        "val_hi_acc": final_hi_acc
    }, run.get_path("data/sali_surrogate_v4.pth"))

    # Exhaustive metrics log
    best_js_epoch = int(np.argmax(history["js_acc"]) + 1)
    best_hi_epoch = int(np.argmax(history["hi_acc"]) + 1)
    with open(run.get_path("logs/metrics.json"), "w") as f:
        json.dump({
            "run_uid": run.uid,
            "version": "v4",
            "accuracy_metric_notes": {
                "js_acc": "(1 - JSD_nats / ln2) * 100 | JSD with natural log, bounded by ln(2)",
                "hi_acc": "sum(min(P,Q)) * 100 | histogram intersection over 200 bins in log-GALI space"
            },
            "training_summary": {
                "total_epochs": epochs,
                "checkpoint_every_n_epochs": checkpoint_every,
                "total_train_time_sec": round(total_train_time, 2),
                "avg_epoch_time_sec": round(total_train_time / epochs, 3),
                "final_js_acc_pct": round(final_js_acc, 4),
                "best_js_acc_pct": round(max(history["js_acc"]), 4),
                "best_js_acc_epoch": best_js_epoch,
                "final_hi_acc_pct": round(final_hi_acc, 4),
                "best_hi_acc_pct": round(max(history["hi_acc"]), 4),
                "best_hi_acc_epoch": best_hi_epoch,
                "final_val_mae": round(history["mae"][-1], 6),
                "best_val_mae": round(min(history["mae"]), 6),
                "final_loss": round(history["loss"][-1], 6),
                "min_loss": round(min(history["loss"]), 6),
                "min_loss_epoch": int(np.argmin(history["loss"]) + 1)
            },
            "dataset_stats": {
                "total_samples": len(df),
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "log_gali_min": float(s_min),
                "log_gali_max": float(s_max),
                "log_gali_mean": float(np.mean(sali)),
                "log_gali_std": float(np.std(sali)),
                "pct_core": round(df['is_core'].sum() / len(df) * 100, 2)
            },
            "inv_density_weight_stats": {
                "n_bins": 200,
                "min": round(float(inv_density_w_train.min()), 4),
                "max": round(float(inv_density_w_train.max()), 4),
                "mean": round(float(inv_density_w_train.mean()), 4),
                "std": round(float(inv_density_w_train.std()), 4)
            },
            "checkpoints_saved": [
                f"ckpt_epoch_{e:04d}.pth" for e in range(checkpoint_every, epochs + 1, checkpoint_every)
            ] + ["ckpt_best_model.pth"],
            "per_epoch_history": {
                "epoch": history["epoch"],
                "loss": [round(v, 6) for v in history["loss"]],
                "mae": [round(v, 6) for v in history["mae"]],
                "js_acc_pct": [round(v, 4) for v in history["js_acc"]],
                "hi_acc_pct": [round(v, 4) for v in history["hi_acc"]],
                "epoch_time_sec": history["epoch_time_sec"]
            }
        }, f, indent=4)

    # Fidelity Plotting
    epochs_x = history["epoch"]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['loss'], label='Inv-Density MSE', color='#636E72')
    plt.title("v4 Training Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['js_acc'], label='JS Accuracy (%)', color='#0984E3')
    plt.plot(epochs_x, history['hi_acc'], label='HI Accuracy (%)', color='#00B894', ls='--')
    plt.title("v4 Distribution Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['mae'], label='Val MAE', color='#D63031')
    plt.title("v4 Val MAE (log-GALI space)"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(run.get_path("logs/v4_training_audit.png"), dpi=150)

    print(f"\n--- v4 SUCCESS ---")
    print(f"Experiment: {run.uid}")
    print(f"Path: {run.path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,  default=100)
    parser.add_argument("--dataset",          type=str,  required=True)
    parser.add_argument("--checkpoint_every", type=int,  default=20,
                        help="Save a checkpoint every N epochs (default: 20)")
    args = parser.parse_args()
    train_surrogate_v4(
        epochs=args.epochs,
        dataset_path=args.dataset,
        checkpoint_every=args.checkpoint_every
    )
