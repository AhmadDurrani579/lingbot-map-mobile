"""
Step 3 — Validate ONNX
Runs the same input through both PyTorch and ONNX
and compares outputs numerically.
"""

import torch
import numpy as np
import onnxruntime as ort
import sys
sys.path.insert(0, "/home/loq/lingbot-map-mobile")

# ─── Constants ────────────────────────────────────────────
CHECKPOINT = "/home/loq/lingbot-map/checkpoints/lingbot-map.pt"
ONNX_PATH  = "models/aggregator.onnx"
H, W       = 294, 518
NUM_LAYERS = 24
NUM_HEADS  = 16
HEAD_DIM   = 64
TOKENS_PER_FRAME = 783

# ─── Load PyTorch Model ───────────────────────────────────
print("Loading PyTorch model...")
from convert.onnx_wrapper import AggregatorONNXWrapper, build_wrapper

wrapper = build_wrapper(CHECKPOINT)
wrapper.eval()

# ─── Load ONNX Model ──────────────────────────────────────
print("Loading ONNX model...")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print("ONNX model loaded.")

# ─── Create Identical Inputs ──────────────────────────────
torch.manual_seed(42)
frame       = torch.randn(1, 1, 3, H, W)
k_cache     = torch.zeros(NUM_LAYERS, 1, NUM_HEADS, 1, TOKENS_PER_FRAME, HEAD_DIM)
v_cache     = torch.zeros(NUM_LAYERS, 1, NUM_HEADS, 1, TOKENS_PER_FRAME, HEAD_DIM)
frame_count = torch.tensor(1)

# ─── PyTorch Forward ──────────────────────────────────────
print("\nRunning PyTorch forward...")
with torch.no_grad():
    pt_tokens, pt_k, pt_v, pt_count = wrapper(
        frame, k_cache, v_cache, frame_count
    )
print(f"PyTorch tokens shape: {pt_tokens.shape}")

# ─── ONNX Forward ─────────────────────────────────────────
print("\nRunning ONNX forward...")
onnx_inputs = {
    "frame":       frame.numpy(),
    "k_cache":     k_cache.numpy(),
    "v_cache":     v_cache.numpy(),
    "frame_count": frame_count.numpy(),
}
onnx_outputs = sess.run(None, onnx_inputs)
onnx_tokens = torch.tensor(onnx_outputs[0])
print(f"ONNX tokens shape: {onnx_tokens.shape}")

# ─── Compare ──────────────────────────────────────────────

print("\nComparing outputs...")
diff = (pt_tokens - onnx_tokens).abs()
print(f"Max difference:    {diff.max().item():.6f}")
print(f"Mean difference:   {diff.mean().item():.6f}")
print(f"Median difference: {diff.median().item():.6f}")
print(f"99th percentile:   {torch.quantile(diff.float().flatten()[:100000], 0.99).item():.6f}")
print(f"Values > 0.01:     {(diff > 0.01).sum().item()} / {diff.numel()}")
print(f"Values > 0.001:    {(diff > 0.001).sum().item()} / {diff.numel()}")

if diff.mean().item() < 0.001:
    print("\n✅ ONNX validation passed — mean difference acceptable")
else:
    print("\n⚠️  Mean difference too high — check conversion")
