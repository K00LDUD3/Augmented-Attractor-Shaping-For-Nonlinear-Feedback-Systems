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


# --- Inverse Density Weighting (ported from v4 GALI Surrogate) ---
def build_inv_density_weights(values_np, n_bins=200):
    """
    Computes per-sample inverse-sqrt-density weights from a 1D distribution.
    Samples in dense regions get low weight; rare samples (tails) get high weight.
    Result is L1-normalized so the mean weight == 1.0.
    """
    counts, bin_edges = np.histogram(values_np, bins=n_bins)
    bin_indices = np.digitize(values_np, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    inv_sqrt = 1.0 / np.sqrt(counts[bin_indices].astype(np.float32) + 1.0)
    inv_sqrt /= inv_sqrt.mean()
    return torch.tensor(inv_sqrt, dtype=torch.float32)


def compute_reconstruction_distribution_accuracy(pred_np, target_np, n_bins=200):
    """
    Distribution-similarity metrics on reconstruction, averaged across all dimensions.
    
    1. JS Accuracy  = (1 - JSD_nats / ln2) * 100
    2. Histogram Intersection = sum(min(P, Q)) * 100
    """
    n_dims = pred_np.shape[1]
    js_accs = []
    hi_accs = []
    
    for d in range(n_dims):
        lo = min(target_np[:, d].min(), pred_np[:, d].min()) - 0.05
        hi = max(target_np[:, d].max(), pred_np[:, d].max()) + 0.05
        bins = np.linspace(lo, hi, n_bins + 1)
        
        P, _ = np.histogram(pred_np[:, d], bins=bins)
        Q, _ = np.histogram(target_np[:, d], bins=bins)
        
        P = P.astype(np.float64) + 1e-12
        Q = Q.astype(np.float64) + 1e-12
        P /= P.sum()
        Q /= Q.sum()
        
        M = 0.5 * (P + Q)
        kl_pm = np.sum(P * np.log(P / M))
        kl_qm = np.sum(Q * np.log(Q / M))
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        js_accs.append(max(0.0, (1.0 - jsd / np.log(2)) * 100.0))
        
        hi_accs.append(float(np.sum(np.minimum(P, Q))) * 100.0)
    
    return float(np.mean(js_accs)), float(np.mean(hi_accs))

# --- Training Logic ---
def train_cassm():
    parser = argparse.ArgumentParser(description="Train caSSM encoder with Tangency Alignment Loss")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--tangency-weight", type=float, default=None,
                        help="Fixed tangency weight (overrides two-phase schedule)")
    parser.add_argument("--tangency-weight-init", type=float, default=0.5,
                        help="Phase 1 tangency weight (while MSE stabilizes)")
    parser.add_argument("--tangency-weight-final", type=float, default=10.0,
                        help="Phase 2 tangency weight (after MSE < threshold)")
    parser.add_argument("--mse-threshold", type=float, default=0.008,
                        help="MSE threshold to trigger Phase 2 tangency step-up")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Direct path to cassm_raw_data.npz. If not set, auto-discovers from registry.")
    args = parser.parse_args()

    # 1. Initialize Tracker & Run
    tracker = ExperimentTracker(BATCH_DIR)
    params = vars(args)
    params["architecture"] = "caSSM-Principled"
    params["tangency_schedule"] = "fixed" if args.tangency_weight else "two-phase"
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
        data_path = args.data_path
        
        # Auto-discover if not explicitly provided
        if data_path is None:
            # Strategy 1: Search registry for Data Generation runs
            registry_path = os.path.join(BATCH_DIR, ".tracker/registry.csv")
            try:
                import pandas as pd
                df_reg = pd.read_csv(registry_path)
                data_runs = df_reg[df_reg['notes'].str.contains("Data Generation", na=False)]
                data_runs = data_runs[data_runs['uid'] != run.uid]
                
                if not data_runs.empty:
                    latest_data_uid = data_runs.iloc[-1]['uid']
                    data_path = os.path.join(BATCH_DIR, latest_data_uid, "data/cassm_raw_data.npz")
            except Exception:
                pass
        
            # Strategy 2: Filesystem fallback — find newest cassm_raw_data.npz
            if data_path is None or not os.path.exists(data_path):
                import glob
                candidates = glob.glob(os.path.join(BATCH_DIR, "*/data/cassm_raw_data.npz"))
                if candidates:
                    data_path = max(candidates, key=os.path.getmtime)
        
        if data_path is None or not os.path.exists(data_path):
            logger.log(f"Data not found. Run data_gen_18d.py first, or pass --data-path.", tag="DEF", level="ERROR")
            return
        
        logger.log(f"Loading data from: {data_path}", tag="DEF", level="INFO")

        raw_data = np.load(data_path)
        states = torch.FloatTensor(raw_data['states'])
        tangents = torch.FloatTensor(raw_data['tangents']) # (N, 12) -> [w1, w2]
        
        # Normalize states for training (Standardized Pillar 6 Scaling: /40)
        states_norm = states / 40.0
        
        # --- Inverse Density Weighting ---
        # Weight by L2 norm of state: samples near origin (stable fibers)
        # are underrepresented in chaotic data, so they get upweighted.
        state_norms = np.linalg.norm(states_norm.numpy(), axis=1)
        inv_density_w = build_inv_density_weights(state_norms, n_bins=200)
        logger.log(f"Inv-Density Weights: min={inv_density_w.min():.3f}, "
                   f"max={inv_density_w.max():.3f}, mean={inv_density_w.mean():.3f}", tag="DEF", level="INFO")
        
        # Train/Val split (90/10)
        n_total = len(states_norm)
        n_val = max(1, int(n_total * 0.1))
        perm = np.random.RandomState(42).permutation(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        
        states_train = states_norm[train_idx]
        tangents_train = tangents[train_idx]
        weights_train = inv_density_w[train_idx]
        
        states_val = states_norm[val_idx]
        tangents_val = tangents[val_idx]
        
        dataset = TensorDataset(states_train, tangents_train, weights_train)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # 4. Model & Optimizer
        model = caSSM(input_dim=6, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
        from pytorch_optimizer import SOAP
        optimizer = SOAP(model.parameters(), lr=args.lr)
        
        # LR Scheduler: ReduceLROnPlateau on tangency loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20,
            min_lr=1e-6
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.log(f"Model: caSSM (hidden={args.hidden_dim}, latent={args.latent_dim}) | {n_params:,} params", tag="DEF", level="INFO")
        logger.log(f"Data: {len(states_train)} train, {len(states_val)} val, {len(loader)} batches/epoch", tag="DEF", level="INFO")
        logger.log(f"LR Scheduler: ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-6) on tangency_loss", tag="DEF", level="INFO")

        # --- Tangency Weight Schedule ---
        if args.tangency_weight is not None:
            # Fixed weight mode
            current_tw = args.tangency_weight
            tw_phase = "fixed"
            logger.log(f"Tangency weight: FIXED at {current_tw}", tag="DEF", level="INFO")
        else:
            # Two-phase schedule: init → final when MSE < threshold
            current_tw = args.tangency_weight_init
            tw_phase = "phase1"
            logger.log(f"Tangency weight schedule: Phase 1 = {args.tangency_weight_init} → "
                       f"Phase 2 = {args.tangency_weight_final} (when MSE < {args.mse_threshold})",
                       tag="DEF", level="INFO")

        logger.log(f"Starting Training Loop for {args.epochs} epochs...", tag="DEF", level="INFO")

        best_js_acc = -1.0
        best_tangency = float('inf')
        prev_lr = args.lr

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            total_mse = 0
            total_tangency = 0

            for x_batch, w_batch, wt_batch in loader:
                x_batch = x_batch.to(device).requires_grad_(True)
                w_batch = w_batch.to(device) # (B, 12)
                wt_batch = wt_batch.to(device) # (B,) inv-density weights
                
                optimizer.zero_grad()
                
                # Forward
                x_rec, z = model(x_batch)
                
                # --- Geometric Subspaces ---
                w1 = w_batch[:, 0:6]
                w2 = w_batch[:, 6:12]
                W = torch.stack([w1, w2], dim=2) # (B, 6, 2)
                
                # Target Projection Matrix (Orthogonal projector onto unstable physical subspace)
                P_target = torch.bmm(W, torch.linalg.pinv(W)) # (B, 6, 6)
                
                # 1. Oblique Reconstruction Loss
                # Penalize the component of the error that lies in the unstable subspace (W)
                # This mathematically forces the reconstruction error into the stable fibers.
                error = x_rec - x_batch # (B, 6)
                proj_error = torch.bmm(P_target, error.unsqueeze(-1)).squeeze(-1) # (B, 6)
                per_sample_mse = torch.sum(proj_error ** 2, dim=1)  # (B,)
                loss_mse = torch.mean(wt_batch * per_sample_mse)
                
                # 2. Subspace Projection Alignment Loss (Tangency)
                z_sum = z.sum(dim=0)
                grads = []
                for i in range(args.latent_dim):
                    g = torch.autograd.grad(z_sum[i], x_batch, create_graph=True)[0]
                    grads.append(g) # (B, 6)
                
                J_E = torch.stack(grads, dim=1) # (B, latent_dim, 6)
                V = J_E.transpose(1, 2) # (B, 6, latent_dim)
                
                # Encoder Projection Matrix (Orthogonal projector onto encoder Jacobian subspace)
                # Use Ridge-regularized inverse for stability if latent_dim > 2
                VT_V = torch.bmm(V.transpose(1, 2), V) # (B, latent_dim, latent_dim)
                eps_I = 1e-6 * torch.eye(args.latent_dim, device=device).unsqueeze(0).expand(x_batch.size(0), -1, -1)
                P_encoder = torch.bmm(V, torch.bmm(torch.linalg.inv(VT_V + eps_I), V.transpose(1, 2))) # (B, 6, 6)
                
                # Trace Overlap Loss: ensures the 2D target subspace is contained in the encoder subspace
                # Maximum overlap for a 2D target is 2.0. Loss bounds: [0.0, 2.0] regardless of latent_dim.
                loss_tangency = 2.0 - torch.mean(torch.sum(P_target * P_encoder, dim=(1, 2)))

                loss = loss_mse + current_tw * loss_tangency
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_mse += loss_mse.item()
                total_tangency += loss_tangency.item()

            avg_loss = total_loss / len(loader)
            avg_mse = total_mse / len(loader)
            avg_tangency = total_tangency / len(loader)

            # --- Two-Phase Tangency Step-Up ---
            if tw_phase == "phase1" and avg_mse < args.mse_threshold:
                current_tw = args.tangency_weight_final
                tw_phase = "phase2"
                logger.log(
                    f">>> PHASE 2 ACTIVATED at epoch {epoch+1} (MSE={avg_mse:.6f} < {args.mse_threshold}) "
                    f"| tangency_weight: {args.tangency_weight_init} → {args.tangency_weight_final}",
                    tag="DEF", level="INFO"
                )

            # --- LR Scheduler Step (on tangency loss) ---
            scheduler.step(avg_tangency)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < prev_lr:
                logger.log(f"    LR reduced: {prev_lr:.2e} → {current_lr:.2e} (tangency plateau)",
                           tag="DEF", level="INFO")
                prev_lr = current_lr

            # --- Validation: JS Accuracy & Histogram Intersection ---
            js_acc, hi_acc = 0.0, 0.0
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
                model.eval()
                with torch.no_grad():
                    x_val = states_val.to(device)
                    x_rec_val, _ = model(x_val)
                    x_rec_val_np = x_rec_val.cpu().numpy()
                    x_val_np = states_val.numpy()
                    js_acc, hi_acc = compute_reconstruction_distribution_accuracy(
                        x_rec_val_np, x_val_np, n_bins=200
                    )
                
                logger.log(
                    f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} "
                    f"(MSE: {avg_mse:.6f}, Tangent: {avg_tangency:.6f}) | "
                    f"JS: {js_acc:.2f}% HI: {hi_acc:.2f}%",
                    tag="DEF", level="INFO"
                )
            
            model_logger.log(metrics={
                "epoch": epoch + 1,
                "loss_total": avg_loss,
                "loss_mse": avg_mse,
                "loss_tangency": avg_tangency,
                "val_js_acc": js_acc,
                "val_hi_acc": hi_acc,
                "lr": current_lr,
                "tangency_weight": current_tw,
            }, version=params["architecture"])

            # Save best model by JS accuracy
            if js_acc > best_js_acc and js_acc > 0:
                best_js_acc = js_acc
                best_path = run.get_path("data/cassm_encoder_best.pth")
                torch.save({
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'params': params,
                    'epoch': epoch + 1,
                    'val_js_acc': js_acc,
                    'val_hi_acc': hi_acc,
                    'loss_tangency': avg_tangency,
                }, best_path)

            # Save best model by tangency alignment
            if avg_tangency < best_tangency:
                best_tangency = avg_tangency
                best_tang_path = run.get_path("data/cassm_encoder_best_tangency.pth")
                torch.save({
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'params': params,
                    'epoch': epoch + 1,
                    'loss_tangency': avg_tangency,
                    'val_js_acc': js_acc,
                    'val_hi_acc': hi_acc,
                }, best_tang_path)

        # 5. Export final model
        model_path = run.get_path("data/cassm_encoder.pth")
        torch.save({
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'params': params
        }, model_path)
        
        logger.log(f"Training Complete. Model saved to {model_path}", tag="FIN", level="INFO")
        logger.log(f"Best JS Accuracy: {best_js_acc:.2f}%", tag="FIN", level="INFO")
        logger.log(f"Best Tangency Loss: {best_tangency:.4f}", tag="FIN", level="INFO")
        run.add_notes(
            f"Final MSE: {avg_mse:.6e}, Final Tangency: {avg_tangency:.4f}, "
            f"Best Tangency: {best_tangency:.4f}, "
            f"Best JS Acc: {best_js_acc:.2f}%, Final JS: {js_acc:.2f}%, Final HI: {hi_acc:.2f}%"
        )

    except Exception as e:
        logger.log(f"Training Failed: {e}", tag="DEF", level="ERROR")
        run.add_notes(f"Failure: {e}")
        raise
    finally:
        logger.end_session()

if __name__ == "__main__":
    train_cassm()
