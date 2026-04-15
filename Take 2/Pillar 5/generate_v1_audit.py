import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# v1 Legacy Architecture (256/512 Mixed MLP)
class SALISurrogateV1(nn.Module):
    def __init__(self, input_dim=6):
        super(SALISurrogateV1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

def generate_v1_audit():
    dataset_path = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_dataset.csv"
    model_path = r"Pillar 5\experiments\2026-04-04_05-42-07_7ee47a7c0f\data\sali_surrogate.pth"
    output_dir = r"Pillar 5\experiments\2026-04-04_05-42-07_7ee47a7c0f\logs"
    os.makedirs(output_dir, exist_ok=True)
    output_img = os.path.join(output_dir, "prediction_audit_v1_pilot.png")

    print(f"Loading Dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    gt_log_gali = np.log10(df['gali2'] + 1e-15)

    print(f"Loading v1 Model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SALISurrogateV1(input_dim=6).to(device)
    
    # v1 models were typically straight state_dicts or wrapped
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Exact Normalization for the 325k dataset
    s_min, s_max = -11.494197070506956, 4.821637332766433e-16

    print("Benchmarking v1 against production dataset...")
    X = torch.tensor(df[['x1','y1','z1','x2','y2','z2']].values / 40.0, dtype=torch.float32).to(device)
    
    preds = []
    batch_size = 10000
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i : i + batch_size]
            out = model(batch_x).cpu().numpy().flatten()
            preds.extend(out)
    
    pred_log_gali = (np.array(preds) * (s_max - s_min + 1e-8)) + s_min

    # Plotting (AAS Standard Aesthetics)
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

    plt.hist(gt_log_gali, bins=100, density=True, color='#636E72', alpha=0.3, label='Ground Truth (18D ODE)')
    plt.hist(pred_log_gali, bins=100, density=True, color='#D63031', alpha=0.5, histtype='step', linewidth=2, label='v1 Pilot Prediction')

    plt.title("v1 Pilot Baseline Audit: Initial Distribution Capture", fontsize=14)
    plt.xlabel("Log10(GALI2) Stability Signature")
    plt.ylabel("Systemic Density")
    plt.grid(alpha=0.1)
    plt.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, transparent=True)
    print(f"v1 Audit finalized: {output_img}")

if __name__ == "__main__":
    generate_v1_audit()
