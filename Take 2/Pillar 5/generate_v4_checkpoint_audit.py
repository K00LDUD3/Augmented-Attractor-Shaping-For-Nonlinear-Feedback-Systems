import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Architecture (must mirror train_surrogate_v4.py exactly)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(x + self.net(x))

class SALISurrogate(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.downsample = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res3 = ResidualBlock(hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x); x = self.res2(x)
        x = self.downsample(x); x = self.res3(x)
        return self.output_layer(x)

# ---------------------------------------------------------------------------
# Config — edit EXP_RUN to point at any v4 experiment folder
# ---------------------------------------------------------------------------
EXP_RUN    = r"D:\Repos\Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems\Take 2\Pillar 5\experiments\2026-04-10_06-03-45_29331e02"
DATASET    = r"D:\Repos\Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems\Take 2\Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_dataset.csv"
CKPT_DIR   = os.path.join(EXP_RUN, "data", "checkpoints")
FINAL_PTH  = os.path.join(EXP_RUN, "data", "sali_surrogate_v4.pth")
METRICS    = os.path.join(EXP_RUN, "logs", "metrics.json")
OUT_DIR    = os.path.join(EXP_RUN, "logs", "distribution_audit")
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent audit style (identical to generate_prediction_audit.py)
COLORS = {
    'ground_truth': '#636E72',
    'prediction':   '#0984E3',
}
PLOT_STYLE = {
    "axes.facecolor":    (0, 0, 0, 0),
    "figure.facecolor":  (0, 0, 0, 0),
    "savefig.facecolor": (0, 0, 0, 0),
    "axes.edgecolor":    "#636E72",
    "xtick.color":       "#2D3436",
    "ytick.color":       "#2D3436",
    "axes.labelcolor":   "#2D3436",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_gt(dataset_path):
    df = pd.read_csv(dataset_path)
    return df, np.log10(df['gali2'].values + 1e-15)

def run_inference(model, df, device, s_min, s_max):
    X = torch.tensor(df[['x1','y1','z1','x2','y2','z2']].values / 40.0, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            preds.append(model(X[i:i+10000].to(device)).cpu().numpy().flatten())
    preds_norm = np.concatenate(preds)
    return preds_norm * (s_max - s_min + 1e-8) + s_min

def plot_audit(gt_log_gali, pred_log_gali, title, subtitle, output_path):
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(gt_log_gali,   bins=100, density=True,
            color=COLORS['ground_truth'], alpha=0.35, label='Ground Truth (18D ODE)')
    ax.hist(pred_log_gali, bins=100, density=True,
            color=COLORS['prediction'],   alpha=0.6,
            histtype='step', linewidth=2,  label='Surrogate Prediction (v4)')

    # Annotate transition zone
    ax.axvspan(np.log10(0.1), np.log10(0.4), color='#FDCB6E', alpha=0.12, label='Transition Zone [0.1, 0.4]')
    ax.axvline(np.log10(0.1), color='#FDCB6E', lw=0.6, ls='--', alpha=0.5)
    ax.axvline(np.log10(0.4), color='#FDCB6E', lw=0.6, ls='--', alpha=0.5)

    # Surplus / deficit inline annotation
    bins  = np.linspace(-12, 1, 100)
    gh, _ = np.histogram(gt_log_gali,   bins=bins, density=True)
    ph, _ = np.histogram(pred_log_gali, bins=bins, density=True)
    delta = ph - gh
    bc    = (bins[:-1] + bins[1:]) / 2
    surplus_val  = delta.max()
    surplus_loc  = bc[delta.argmax()]
    deficit_val  = delta.min()
    deficit_loc  = bc[delta.argmin()]
    emd = float(np.sum(np.abs(np.cumsum(ph) - np.cumsum(gh))))

    stats_txt = (
        f"Largest Surplus: +{surplus_val:.4f} @ logGALI={surplus_loc:.2f}\n"
        f"Largest Deficit:  {deficit_val:.4f} @ logGALI={deficit_loc:.2f}\n"
        f"EMD: {emd:.4f}"
    )
    ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8, va='top', ha='left', color='#636E72',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.4, ec='none'))

    ax.set_title(f"Stability Topography Audit: {title}", fontsize=13, weight='bold', color='#2D3436')
    ax.set_xlabel("Log10(GALI2) Stability Signature", color='#2D3436')
    ax.set_ylabel("Systemic Density", color='#2D3436')
    ax.grid(alpha=0.1)
    ax.legend(frameon=False, fontsize=9)

    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha='center', fontsize=8, color='#636E72')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"  Saved -> {os.path.basename(output_path)}")

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset...")
    df, gt_log_gali = load_gt(DATASET)

    # Load metrics to identify best checkpoint
    best_epoch = None
    if os.path.exists(METRICS):
        with open(METRICS, 'r') as f:
            m = json.load(f)
        best_epoch = m.get("training_summary", {}).get("best_val_acc_epoch", None)
        print(f"Best Val Acc Epoch (from metrics.json): {best_epoch}")

    # Collect all checkpoints sorted by epoch
    ckpt_files = sorted([
        f for f in os.listdir(CKPT_DIR) if f.endswith('.pth')
    ])
    print(f"Found {len(ckpt_files)} checkpoints + 1 final model.\n")

    # --- Checkpoint sweep ---
    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        epoch    = ckpt.get("epoch", "?")
        s_min    = ckpt["sali_min"]
        s_max    = ckpt["sali_max"]
        js_acc   = ckpt.get("val_js_acc", ckpt.get("val_acc", None))  # fallback for old format
        hi_acc   = ckpt.get("val_hi_acc", None)
        val_mae  = ckpt.get("val_mae", None)

        model = SALISurrogate().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        pred_log_gali = run_inference(model, df, device, s_min, s_max)

        stem     = os.path.splitext(ckpt_name)[0]
        out_path = os.path.join(OUT_DIR, f"audit_{stem}.png")
        subtitle_parts = [f"Epoch {epoch}"]
        if js_acc  is not None: subtitle_parts.append(f"JS Acc: {js_acc:.2f}%")
        if hi_acc  is not None: subtitle_parts.append(f"HI Acc: {hi_acc:.2f}%")
        if val_mae is not None: subtitle_parts.append(f"MAE: {val_mae:.5f}")
        subtitle = "  |  ".join(subtitle_parts)

        print(f"[{stem}] Epoch={epoch}, JS={js_acc}")
        plot_audit(
            gt_log_gali, pred_log_gali,
            title=f"v4 Checkpoint — {stem}",
            subtitle=subtitle,
            output_path=out_path
        )
        del model

    # --- Final model ---
    print("\n[Final Model] Generating audit...")
    final_ckpt = torch.load(FINAL_PTH, map_location=device, weights_only=False)
    s_min = final_ckpt["sali_min"]
    s_max = final_ckpt["sali_max"]
    model  = SALISurrogate().to(device)
    model.load_state_dict(final_ckpt["model_state_dict"])
    model.eval()
    pred_log_gali = run_inference(model, df, device, s_min, s_max)
    plot_audit(
        gt_log_gali, pred_log_gali,
        title="v4 Final Model (Epoch 250)",
        subtitle="sali_surrogate_v4.pth",
        output_path=os.path.join(OUT_DIR, "audit_final_model.png")
    )
    del model

    # --- Best model (ckpt_best_model.pth, saved whenever JS acc improves) ---
    best_model_path = os.path.join(CKPT_DIR, "ckpt_best_model.pth")
    if os.path.exists(best_model_path):
        print("\n[Best Model] Generating audit...")
        ckpt    = torch.load(best_model_path, map_location=device, weights_only=False)
        s_min   = ckpt["sali_min"]
        s_max   = ckpt["sali_max"]
        epoch   = ckpt.get("epoch", "?")
        js_acc  = ckpt.get("val_js_acc", None)
        hi_acc  = ckpt.get("val_hi_acc", None)
        val_mae = ckpt.get("val_mae", None)
        model   = SALISurrogate().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        pred_log_gali = run_inference(model, df, device, s_min, s_max)
        subtitle_parts = [f"Best JS Model — Epoch {epoch}"]
        if js_acc  is not None: subtitle_parts.append(f"JS Acc: {js_acc:.2f}%")
        if hi_acc  is not None: subtitle_parts.append(f"HI Acc: {hi_acc:.2f}%")
        if val_mae is not None: subtitle_parts.append(f"MAE: {val_mae:.5f}")
        plot_audit(
            gt_log_gali, pred_log_gali,
            title=f"v4 Best Model (Epoch {epoch})",
            subtitle="  |  ".join(subtitle_parts),
            output_path=os.path.join(OUT_DIR, "audit_best_model.png")
        )
        del model
    else:
        print("  ckpt_best_model.pth not found — skipping best model audit.")

    print(f"\nAll audit plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
