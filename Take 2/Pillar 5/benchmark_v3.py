import torch
import torch.nn as nn
import numpy as np
import os
import time
import json
import onnx
import onnxruntime as ort
from train_surrogate_v3 import SALISurrogate

def benchmark_v3_model():
    run_dir = r"Pillar 5\experiments\2026-04-09_18-09-00_c5f67923"
    model_path = os.path.join(run_dir, "data", "sali_surrogate_v3.pth")
    metrics_path = os.path.join(run_dir, "logs", "latency_metrics.json")
    onnx_path = os.path.join(run_dir, "data", "sali_surrogate_v3.onnx")

    print(f"Loading v3 Residual Model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SALISurrogate(input_dim=6, hidden_dim=512).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- ONNX Export ---
    print("Exporting v3 to ONNX...")
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

    results = {"model": "sali_surrogate_v3_residual"}
    iters = 500
    
    # --- Torch Native Benchmarks ---
    for dev_name in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        target_dev = torch.device(dev_name)
        model.to(target_dev)
        x = torch.randn(1, 6).to(target_dev)
        
        # Warmup
        for _ in range(20): _ = model(x)
        
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
        for _ in range(20): _ = sess.run(None, {input_name: x_np})
        
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
    benchmark_v3_model()
