import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the v3/v4 architecture
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.net(x))

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
        x = self.res1(x); x = self.res2(x); x = self.downsample(x); x = self.res3(x)
        return self.output_layer(x)

def check_distribution_deltas():
    # Constructing paths from the logic of Pillar 5
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "experiments", "2026-04-06_02-10-39_0a402e42", "gali_dataset.csv")
    model_path = os.path.join(base_dir, "experiments", "2026-04-09_18-09-00_c5f67923", "data", "sali_surrogate_v3.pth")
    
    if not os.path.exists(dataset_path):
        # Fallback for relative vs absolute
        dataset_path = r"d:\Repos\Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems\Take 2\Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_dataset.csv"
        model_path = r"d:\Repos\Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems\Take 2\Pillar 5\experiments\2026-04-09_18-09-00_c5f67923\data\sali_surrogate_v3.pth"

    print(f"Loading data: {dataset_path}")
    df = pd.read_csv(dataset_path)
    gt_log_gali = np.log10(df['gali2'].values + 1e-15)
    
    print(f"Loading v3 model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = SALISurrogate()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    s_min = checkpoint['sali_min']
    s_max = checkpoint.get('sali_max', checkpoint.get('sali_max', 0.0))
    if s_max == 0.0:
        # Check if it's in the params.json or just recalculate
        s_max = gt_log_gali.max()

    X = torch.tensor(df[['x1','y1','z1','x2','y2','z2']].values / 40.0, dtype=torch.float32)
    with torch.no_grad():
        # Batched to avoid MemoryError
        preds_list = []
        for i in range(0, len(X), 10000):
            preds_list.append(model(X[i:i+10000]).cpu().numpy().flatten())
        preds_norm = np.concatenate(preds_list)
    
    pred_log_gali = preds_norm * (s_max - s_min + 1e-8) + s_min
    
    # Histogram analysis
    bins = np.linspace(-12, 1, 100)
    gt_hist, _ = np.histogram(gt_log_gali, bins=bins, density=True)
    pred_hist, _ = np.histogram(pred_log_gali, bins=bins, density=True)
    
    deltas = pred_hist - gt_hist
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    print("\n--- Numerical Distribution Analysis (v3 vs GT) ---")
    max_pos_delta_idx = np.argmax(deltas)
    max_neg_delta_idx = np.argmin(deltas)
    
    print(f"Largest Surplus (Spike): {deltas[max_pos_delta_idx]:.4f} at logGALI ~ {bin_centers[max_pos_delta_idx]:.2f}")
    print(f"Largest Deficit: {deltas[max_neg_delta_idx]:.4f} at logGALI ~ {bin_centers[max_neg_delta_idx]:.2f}")
    
    # Check transition [0.1, 0.4] => logGALI [-1.0, -0.4]
    trans_mask = (bin_centers >= -1.0) & (bin_centers <= -0.4)
    avg_trans_delta = np.mean(deltas[trans_mask])
    print(f"Average Delta in Transition Zone [-1.0, -0.4]: {avg_trans_delta:.4f}")

    # EMD Check (Earth Mover Distance as a proxy)
    emd = np.sum(np.abs(np.cumsum(pred_hist) - np.cumsum(gt_hist)))
    print(f"Approx Earth Mover Distance (CDF deviation): {emd:.4f}")

if __name__ == "__main__":
    check_distribution_deltas()
