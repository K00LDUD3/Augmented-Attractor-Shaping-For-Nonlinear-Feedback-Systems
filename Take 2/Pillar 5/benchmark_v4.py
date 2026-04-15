import torch
import torch.nn as nn
import numpy as np
import os
import time
import json
import onnx
import onnxruntime as ort

# Architecture (must mirror train_surrogate_v4.py)
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

def benchmark_v4_model():
    run_dir = r"Pillar 5\experiments\2026-04-10_06-03-45_29331e02"
    model_path = os.path.join(run_dir, "data", "sali_surrogate_v4.pth")
    metrics_path = os.path.join(run_dir, "logs", "latency_metrics.json")
    onnx_path = os.path.join(run_dir, "data", "sali_surrogate_v4.onnx")

    print(f"Loading v4 Residual Model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SALISurrogate(input_dim=6, hidden_dim=512).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- ONNX Export ---
    print("Exporting v4 to ONNX...")
    dummy_input = torch.randn(1, 6).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=['input'], 
        output_names=['output'], 
        opset_version=15,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    results = {"model": "sali_surrogate_v4_inv_density"}
    iters = 1000  # More iters for stability
    
    # --- Torch Native Benchmarks ---
    for dev_name in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        target_dev = torch.device(dev_name)
        model.to(target_dev)
        x = torch.randn(1, 6).to(target_dev)
        
        # Warmup
        for _ in range(50): _ = model(x)
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(x)
                if dev_name == 'cuda': torch.cuda.synchronize()
        
        latency = (time.perf_counter() - start) / iters * 1000
        print(f"  [Torch ][{dev_name.upper()}] Latency: {latency:.4f} ms")
        results[f"torch_latency_{dev_name}_ms"] = latency

    # --- ONNX Runtime Benchmarks ---
    providers = [('cpu', 'CPUExecutionProvider')]
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append(('cuda', 'CUDAExecutionProvider'))
    
    x_np = np.random.randn(1, 6).astype(np.float32)
    for dev_name, provider in providers:
        sess = ort.InferenceSession(onnx_path, providers=[provider])
        input_name = sess.get_inputs()[0].name
        
        # Warmup
        for _ in range(50): _ = sess.run(None, {input_name: x_np})
        
        start = time.perf_counter()
        for _ in range(iters):
            _ = sess.run(None, {input_name: x_np})
        
        latency = (time.perf_counter() - start) / iters * 1000
        print(f"  [ONNX  ][{dev_name.upper()}] Latency: {latency:.4f} ms")
        results[f"onnx_latency_{dev_name}_ms"] = latency

    # Save Metrics
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nLatency metrics saved to: {metrics_path}")

if __name__ == "__main__":
    benchmark_v4_model()
