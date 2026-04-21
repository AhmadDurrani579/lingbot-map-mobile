import coremltools as ct
import coremltools.optimize.coreml as cto
import os

OUTPUT_PATH = "models/aggregator_ios18.mlpackage"
OUTPUT_INT4 = "models/aggregator_int4.mlpackage"

print("Loading CoreML model...")
model = ct.models.MLModel(OUTPUT_PATH)

print("Applying INT4 quantization...")
op_config = cto.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
)
config = cto.OptimizationConfig(global_config=op_config)
model_int4 = cto.linear_quantize_weights(model, config=config)
model_int4.save(OUTPUT_INT4)

size_int4 = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, files in os.walk(OUTPUT_INT4)
    for f in files
) / 1e9
print(f"✅ INT4 size: {size_int4:.2f} GB")