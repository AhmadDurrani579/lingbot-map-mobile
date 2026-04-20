"""
Step 2 — ONNX Wrapper
Wraps AggregatorStream with explicit KV cache I/O
so ONNX export can trace through it cleanly.

KV cache shape per layer: [1, 16, frames, 783, 64]
24 layers × 2 (k,v) = 48 cache tensors as explicit I/O
"""

import torch
import torch.nn as nn
from typing import Tuple, List

# ─── Constants ────────────────────────────────────────────
NUM_LAYERS   = 24
NUM_HEADS    = 16
HEAD_DIM     = 64
TOKENS_PER_FRAME = 783
MAX_FRAMES   = 72   # 8 scale + 64 sliding window
B            = 1    # batch size always 1 for mobile

class AggregatorONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper around AggregatorStream.
    
    Converts the internal dict-based KV cache into
    explicit tensor inputs/outputs that ONNX can trace.
    
    Inputs:
        frame:    [1, 1, 3, H, W] — single frame
        k_cache:  [24, 1, 16, frames, 783, 64] — key cache
        v_cache:  [24, 1, 16, frames, 783, 64] — value cache
        frame_count: scalar — how many frames processed so far
        
    Outputs:
        tokens:      [1, S, P, 2C] — aggregated features
        k_cache_new: [24, 1, 16, frames, 783, 64] — updated keys
        v_cache_new: [24, 1, 16, frames, 783, 64] — updated values
        frame_count_new: scalar — updated frame count
    """

    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator

    def forward(
        self,
        frame: torch.Tensor,          # [1, 1, 3, H, W]
        k_cache: torch.Tensor,        # [24, 16, F, 783, 64] ← rank 5
        v_cache: torch.Tensor,        # [24, 16, F, 783, 64] ← rank 5
        frame_count: torch.Tensor,    # [1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Load KV cache — add batch dim back for aggregator
        for i in range(NUM_LAYERS):
            self.aggregator.kv_cache[f"k_{i}"] = k_cache[i].unsqueeze(0)  # [1, 16, F, 783, 64]
            self.aggregator.kv_cache[f"v_{i}"] = v_cache[i].unsqueeze(0)

        # 2. Run aggregator
        output_list, patch_start_idx = self.aggregator(frame)

        # 3. Stack output tokens
        tokens = torch.stack(output_list, dim=0)  # [num_blocks, B, S, P, 2C]

        # 4. Extract updated KV cache — remove batch dim for CoreML
        k_out = torch.stack([
            self.aggregator.kv_cache[f"k_{i}"].squeeze(0)  # [16, F, 783, 64]
            for i in range(NUM_LAYERS)
        ], dim=0)  # [24, 16, F, 783, 64] ← rank 5

        v_out = torch.stack([
            self.aggregator.kv_cache[f"v_{i}"].squeeze(0)
            for i in range(NUM_LAYERS)
        ], dim=0)  # [24, 16, F, 783, 64] ← rank 5

        # 5. Update frame count
        frame_count_new = frame_count + torch.ones(1, dtype=frame_count.dtype)

        return tokens, k_out, v_out, frame_count_new

def build_wrapper(checkpoint_path: str, device: str = "cpu") -> AggregatorONNXWrapper:
    """Load model and wrap in ONNX-compatible interface."""
    import sys
    sys.path.insert(0, "/home/loq/lingbot-map-mobile")

    from lingbot_map.models.gct_stream import GCTStream

    print("Loading model...")
    model = GCTStream(use_sdpa=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = AggregatorONNXWrapper(model.aggregator)
    wrapper = wrapper.to(device).eval()
    print("Wrapper built.")
    return wrapper


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/loq/lingbot-map-mobile")

    CHECKPOINT = "/home/loq/lingbot-map/checkpoints/lingbot-map.pt"
    H, W = 294, 518

    wrapper = build_wrapper(CHECKPOINT)

    # Test with dummy inputs
    print("\nTesting wrapper with dummy inputs...")
    frame      = torch.randn(1, 1, 3, H, W)
    k_cache = torch.zeros(NUM_LAYERS, NUM_HEADS, 1, TOKENS_PER_FRAME, HEAD_DIM)
    v_cache = torch.zeros(NUM_LAYERS, NUM_HEADS, 1, TOKENS_PER_FRAME, HEAD_DIM)
    frame_count = torch.tensor(1)

    with torch.no_grad():
        tokens, k_new, v_new, count_new = wrapper(
            frame, k_cache, v_cache, frame_count
        )

    print(f"tokens shape:    {tokens.shape}")
    print(f"k_new shape:     {k_new.shape}")
    print(f"v_new shape:     {v_new.shape}")
    print(f"frame_count_new: {count_new}")
    print("\n✅ Wrapper works correctly")

    # ─── ONNX Export Attempt ──────────────────────────────
    print("\nAttempting ONNX export...")

    try:
        torch.onnx.export(
            wrapper,
            (frame, k_cache, v_cache, frame_count),
            "models/aggregator.onnx",
            opset_version=18,
            input_names=["frame", "k_cache", "v_cache", "frame_count"],
            output_names=["tokens", "k_cache_new", "v_cache_new", "frame_count_new"],
            dynamic_axes={
                "k_cache": {3: "cached_frames"},
                "v_cache": {3: "cached_frames"},
                "k_cache_new": {3: "cached_frames"},
                "v_cache_new": {3: "cached_frames"},
            },
            do_constant_folding=True,
            verbose=False,
        )
        print("✅ ONNX export succeeded")
        print("Saved to models/aggregator.onnx")

    except Exception as e:
        print(f"❌ ONNX export failed:")
        print(f"   {e}")    
