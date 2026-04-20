"""
Step 1 — Trace Audit
Loads LingBot-Map in SDPA mode and traces a single frame
to identify which ops are incompatible with ONNX export.
"""

import torch
import sys
import warnings
warnings.filterwarnings("ignore")

# Add lingbot-map to path
sys.path.insert(0, "/home/loq/lingbot-map-mobile")

# ─── Config ───────────────────────────────────────────────
CHECKPOINT = "/home/loq/lingbot-map/checkpoints/lingbot-map.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
H, W       = 294, 518   # fixed resolution
DTYPE      = torch.bfloat16

# ─── Load Model ───────────────────────────────────────────
print("Loading model...")
from lingbot_map.models.gct_stream import GCTStream

model = GCTStream(use_sdpa=True)

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model", ckpt)
model.load_state_dict(state_dict, strict=False)

# Cast aggregator to bfloat16, keep heads fp32
model.aggregator = model.aggregator.to(dtype=DTYPE)
model = model.to(DEVICE).eval()
print("Model loaded.")

# ─── Single Frame Input ───────────────────────────────────
print("Creating single frame input...")
dummy_frame = torch.randn(1, 1, 3, H, W, dtype=DTYPE, device=DEVICE)

# ─── Check Output Structure ───────────────────────────────
print("\nChecking output structure...")
with torch.no_grad():
    output = model.aggregator(dummy_frame)
    print(f"Output type: {type(output)}")
    if isinstance(output, (list, tuple)):
        print(f"Output length: {len(output)}")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  [{i}] tensor shape: {o.shape} dtype: {o.dtype}")
            else:
                print(f"  [{i}] type: {type(o)}")
    elif isinstance(output, dict):
        for k, v in output.items():
            print(f"  {k}: {type(v)}")
    else:
        print(f"Output: {output}")

# ─── Trace Attempt ────────────────────────────────────────
print("\nAttempting torch.jit.trace...")


# ─── Trace Attempt ────────────────────────────────────────
print("\nAttempting torch.jit.trace...")
print("This will show every op that breaks tracing.\n")

try:
    with torch.no_grad():
        traced = torch.jit.trace(
            model.aggregator,
            (dummy_frame,),
            strict=False,
            check_trace=False,
        )
    print("Trace succeeded")
    
    # Save trace graph
    print("\nOps in traced graph:")
    print(traced.graph)
    
except Exception as e:
    print(f"Trace failed: {e}")

# ─── ONNX Export Attempt ──────────────────────────────────
print("\nAttempting ONNX export...")

try:
    with torch.no_grad():
        torch.onnx.export(
            model.aggregator,
            (dummy_frame,),
            "models/aggregator_test.onnx",
            opset_version=17,
            input_names=["frames"],
            output_names=["tokens"],
            do_constant_folding=True,
            verbose=False,
        )
    print("✅ ONNX export succeeded")

except Exception as e:
    print(f"❌ ONNX export failed:")
    print(f"   {e}")

print("\nAudit complete.")
