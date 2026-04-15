import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from train_surrogate_v3 import SALISurrogate

# Minimalist Color Palette
COLORS = {
    'ground_truth': '#636E72', # Grey
    'prediction': '#0984E3',   # Blue
}

def generate_prediction_audit():
    dataset_path = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_dataset.csv"
    model_path = r"Pillar 5\experiments\2026-04-09_18-09-00_c5f67923\data\sali_surrogate_v3.pth"
    output_dir = r"Pillar 5\experiments\2026-04-09_18-09-00_c5f67923\logs"
    os.makedirs(output_dir, exist_ok=True)
    output_img = os.path.join(output_dir, "prediction_audit_v3_residual.png")

    print(f"Loading Dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    gt_log_gali = np.log10(df['gali2'] + 1e-15)

    print(f"Loading Model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SALISurrogate(input_dim=6, hidden_dim=512).to(device)
    
    # Resilient Load Logic - Using exact constants from the specific production run
    s_min, s_max = -11.494197070506956, 4.821637332766433e-16
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("Performing Monolithic Inference on 325k points...")
    X = torch.tensor(df[['x1','y1','z1','x2','y2','z2']].values / 40.0, dtype=torch.float32).to(device)
    
    # Inference in batches to preserve memory
    preds = []
    batch_size = 10000
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i : i + batch_size]
            out = model(batch_x).cpu().numpy().flatten()
            preds.extend(out)
    
    pred_log_gali = (np.array(preds) * (s_max - s_min + 1e-8)) + s_min

    # Plotting Refined Audit
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        "axes.facecolor": (0,0,0,0),
        "figure.facecolor": (0,0,0,0),
        "savefig.facecolor": (0,0,0,0),
        "axes.edgecolor": "#636E72",
        "xtick.color": "#2D3436",
        "ytick.color": "#2D3436",
        "axes.labelcolor": "#2D3436"
    })

    plt.hist(gt_log_gali, bins=100, density=True, color=COLORS['ground_truth'], alpha=0.3, label='Ground Truth (18D ODE)')
    plt.hist(pred_log_gali, bins=100, density=True, color=COLORS['prediction'], alpha=0.5, histtype='step', linewidth=2, label='Surrogate Prediction (MLP)')

    plt.title("Stability Topography Audit: Ground Truth vs. Surrogate Prediction", fontsize=14)
    plt.xlabel("Log10(GALI2) Stability Signature")
    plt.ylabel("Systemic Density")
    plt.grid(alpha=0.1)
    plt.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, transparent=True)
    print(f"Prediction Audit finalized and saved to: {output_img}")

if __name__ == "__main__":
    generate_prediction_audit()
