import torch
import torch.nn as nn
import numpy as np
import os
import time
import onnx
import onnxruntime as ort

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

def benchmark_v1():
    model_path = r"Pillar 5\experiments\2026-04-04_05-42-07_7ee47a7c0f\data\sali_surrogate.pth"
    onnx_path = "v1_temp.onnx"
    
    device = torch.device("cpu")
    model = SALISurrogateV1().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dummy = torch.randn(1, 6)
    torch.onnx.export(model, dummy, onnx_path, opset_version=15)
    
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    x_np = dummy.numpy()
    iters = 1000
    for _ in range(50): sess.run(None, {input_name: x_np})
    
    t0 = time.perf_counter()
    for _ in range(iters): sess.run(None, {input_name: x_np})
    lat = (time.perf_counter() - t0) / iters * 1000
    print(f"V1_LATENCY_MS: {lat:.6f}")
    
    os.remove(onnx_path)

if __name__ == "__main__":
    benchmark_v1()
