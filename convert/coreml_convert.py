"""
CoreML Conversion Pipeline
PyTorch → CoreML → INT4 Quantization → .mlpackage
"""

import torch
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import os
import sys
sys.path.insert(0, "/home/loq/lingbot-map-mobile")

# ─── Constants ────────────────────────────────────────────
CHECKPOINT   = "/home/loq/lingbot-map/checkpoints/lingbot-map.pt"
OUTPUT_PATH  = "models/aggregator.mlpackage"
OUTPUT_INT4  = "models/aggregator_int4.mlpackage"
H, W         = 294, 518
NUM_LAYERS   = 24
NUM_HEADS    = 16
HEAD_DIM     = 64
TOKENS_PER_FRAME = 783

# ─── Load Wrapper ─────────────────────────────────────────
print("Loading model...")
from convert.onnx_wrapper import AggregatorONNXWrapper, build_wrapper
wrapper = build_wrapper(CHECKPOINT)
wrapper.eval()
print("Model loaded.")

# ─── Example Inputs ───────────────────────────────────────
frame       = torch.randn(1, 1, 3, H, W)
k_cache = torch.zeros(NUM_LAYERS, NUM_HEADS, 1, TOKENS_PER_FRAME, HEAD_DIM)
v_cache = torch.zeros(NUM_LAYERS, NUM_HEADS, 1, TOKENS_PER_FRAME, HEAD_DIM)
frame_count = torch.tensor([1])  # rank-1 instead of scalar

example_inputs = (frame, k_cache, v_cache, frame_count)

# ─── Step 1 — Trace with torch.jit ────────────────────────
print("\nStep 1: Tracing model...")
with torch.no_grad():
    traced = torch.jit.trace(
        wrapper,
        example_inputs,
        strict=False,
        check_trace=False,
    )
print("✅ Trace done")

# ─── Step 2 — Convert to CoreML ───────────────────────────
print("\nStep 2: Converting to CoreML...")
print("This will take several minutes...")

model = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="frame",       shape=frame.shape),
        ct.TensorType(name="k_cache",     shape=k_cache.shape),
        ct.TensorType(name="v_cache",     shape=v_cache.shape),
        ct.TensorType(name="frame_count", shape=(1,), dtype=np.int64),
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
)

print("✅ CoreML conversion done")
model.save(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")

size = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, files in os.walk(OUTPUT_PATH)
    for f in files
) / 1e9
print(f"Model size: {size:.2f} GB")

# ─── Step 3 — INT4 Quantization ───────────────────────────
print("\nStep 3: Applying INT4 quantization...")

op_config = cto.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
    granularity="per_tensor",
)

config = cto.OptimizationConfig(global_config=op_config)
model_int4 = cto.linear_quantize_weights(model, config=config)

model_int4.save(OUTPUT_INT4)
print("✅ INT4 quantization done")
print(f"Saved to {OUTPUT_INT4}")

size_int4 = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, files in os.walk(OUTPUT_INT4)
    for f in files
) / 1e9
print(f"INT4 model size: {size_int4:.2f} GB")
print(f"Compression ratio: {size/size_int4:.1f}x")