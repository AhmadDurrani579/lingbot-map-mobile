import coremltools as ct
import coremltools.optimize.coreml as cto
import os

OUTPUT_PATH  = "models/aggregator.mlpackage"
OUTPUT_INT8  = "models/aggregator_int8.mlpackage"

print("Loading CoreML model...")
model = ct.models.MLModel(OUTPUT_PATH)

print("Applying INT8 quantization...")
op_config = cto.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8",
)
config = cto.OptimizationConfig(global_config=op_config)
model_int8 = cto.linear_quantize_weights(model, config=config)
model_int8.save(OUTPUT_INT8)

size_int8 = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, files in os.walk(OUTPUT_INT8)
    for f in files
) / 1e9
print(f"✅ INT8 size: {size_int8:.2f} GB")