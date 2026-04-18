import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # Strictly after torch
import time
import json
import argparse
from tqdm import tqdm

from ExperimentTracker import ExperimentTracker
from Logger import Logger, ModelLogger

# --- PILLAR 6 CONFIG ---
PILLAR_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join(PILLAR_DIR, "6d_cassm_batch")

# --- caSSM Architecture ---
class SSMEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, latent_dim=2):
        super(SSMEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class SSMDecoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=256, output_dim=6):
        super(SSMDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class caSSM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, latent_dim=2):
        super(caSSM, self).__init__()
        self.encoder = SSMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = SSMDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

# --- Training Logic ---
def train_cassm():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--tangency-weight", type=float, default=0.1)
    args = parser.parse_args()

    # 1. Initialize Tracker & Run
    tracker = ExperimentTracker(BATCH_DIR)
    params = vars(args)
    params["architecture"] = "caSSM-Principled"
    run = tracker.create_run(params=params, notes="Principled caSSM training with Tangency Alignment Loss.")

    # 2. Setup Logging
    logger = Logger(f=run.get_path("logs/run.log"), sesh_file=run.get_path("logs/.sesh_num"))
    model_logger = ModelLogger(filepath=run.get_path("logs/metrics.json"))
    run.copy_file(os.path.abspath(__file__), "configs/")

    logger.start_session()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"Training on device: {device}", tag="BEG", level="INFO")

        # 3. Load Data
        # We need to find the latest data gen run. 
        # Since 'run' was just created, tracker.get_latest_run() will return 'run'.
        # We search the registry for the previous run that contains the data.
        
        data_path = None
        registry_path = os.path.join(BATCH_DIR, ".tracker/registry.csv")
        try:
            import pandas as pd
            df_reg = pd.read_csv(registry_path)
            # Find runs involving "Data Generation" in notes, excluding the current one
            data_runs = df_reg[df_reg['notes'].str.contains("Data Generation", na=False)]
            data_runs = data_runs[data_runs['uid'] != run.uid]
            
            if not data_runs.empty:
                latest_data_uid = data_runs.iloc[-1]['uid']
                data_path = os.path.join(BATCH_DIR, latest_data_uid, "data/cassm_raw_data.npz")
        except Exception:
            # Fallback if pandas/registry lookup fails
            pass
        
        if not os.path.exists(data_path):
            logger.log(f"Data not found at {data_path}. Ensure data_gen_18d.py finished.", tag="DEF", level="ERROR")
            return

        raw_data = np.load(data_path)
        states = torch.FloatTensor(raw_data['states'])
        tangents = torch.FloatTensor(raw_data['tangents']) # (N, 12) -> [w1, w2]
        
        # Normalize states for training (Standardized Pillar 6 Scaling: /40)
        states_norm = states / 40.0
        
        dataset = TensorDataset(states_norm, tangents)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # 4. Model & Optimizer
        model = caSSM(input_dim=6, hidden_dim=256, latent_dim=args.latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion_mse = nn.MSELoss()

        logger.log(f"Starting Training Loop for {args.epochs} epochs...", tag="DEF", level="INFO")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            total_mse = 0
            total_tangency = 0

            for x_batch, w_batch in loader:
                x_batch = x_batch.to(device).requires_grad_(True)
                w_batch = w_batch.to(device) # (B, 12)
                
                optimizer.zero_grad()
                
                # Forward
                x_rec, z = model(x_batch)
                
                # 1. Reconstruction Loss
                loss_mse = criterion_mse(x_rec, x_batch)
                
                # 2. Tangency Alignment Loss (Physically Principled)
                # We want grad(E) * w to be maximized (aligning latent space with principal tangents)
                # This ensures the latent space captures the 'actionable' chaotic directions.
                
                # Split tangents into w1, w2
                w1 = w_batch[:, 0:6]
                w2 = w_batch[:, 6:12]
                
                # Compute Jacobian of Encoder w.r.t input
                # For efficiency in batches, we can take gradients of latent sum
                z_sum = z.sum(dim=0)
                grads = []
                for i in range(args.latent_dim):
                    g = torch.autograd.grad(z_sum[i], x_batch, create_graph=True)[0]
                    grads.append(g) # (B, 6)
                
                # grads is List of length latent_dim, each (B, 6)
                # Alignment: dot products between encoder gradients and physical tangents
                # We want the subspace spanned by Encoder Grad to match subspace of Tangents
                loss_tangency = 0
                for g in grads:
                    # Minimize 1 - cos_sim^2 (or similar) to align directions
                    # Here we just penalize misalignment
                    cos1 = torch.abs(torch.sum(g * w1, dim=1) / (torch.norm(g, dim=1) * torch.norm(w1, dim=1) + 1e-8))
                    cos2 = torch.abs(torch.sum(g * w2, dim=1) / (torch.norm(g, dim=1) * torch.norm(w2, dim=1) + 1e-8))
                    loss_tangency += (1.0 - (cos1 + cos2)/2.0).mean()

                loss = loss_mse + args.tangency_weight * loss_tangency
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_mse += loss_mse.item()
                total_tangency += loss_tangency.item()

            avg_loss = total_loss / len(loader)
            avg_mse = total_mse / len(loader)
            avg_tangency = total_tangency / len(loader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.log(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} (MSE: {avg_mse:.6f}, Tangent: {avg_tangency:.6f})", tag="DEF", level="INFO")
            
            model_logger.log(metrics={
                "epoch": epoch + 1,
                "loss_total": avg_loss,
                "loss_mse": avg_mse,
                "loss_tangency": avg_tangency
            }, version=params["architecture"])

        # 5. Export
        model_path = run.get_path("data/cassm_encoder.pth")
        torch.save({
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'params': params
        }, model_path)
        
        logger.log(f"Training Complete. Model saved to {model_path}", tag="FIN", level="INFO")
        run.add_notes(f"Final MSE: {avg_mse:.6e}, Final Tangency Loss: {avg_tangency:.4f}")

    except Exception as e:
        logger.log(f"Training Failed: {e}", tag="DEF", level="ERROR")
        run.add_notes(f"Failure: {e}")
        raise
    finally:
        logger.end_session()

if __name__ == "__main__":
    train_cassm()
