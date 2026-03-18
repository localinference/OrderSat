# Model Size and Precision Practices

Precision policy is phase-bound and deployment-bound, not data-size-bound.

## Training-phase rule

For phase `05_train_pytorch_model`, use:

- `FP32` training
- `FP32` checkpoints

Do not use mixed precision in this phase.

## Later export rule

When export and deployment modules are added, use this path:

1. Keep the canonical model in `FP32`.
2. Export one canonical `FP32 ONNX` model.
3. Derive deployment variants from that export:
   GPU: `FP16`
   CPU: `INT8`

## Why

- `FP32` is the stable canonical training and checkpoint format.
- phase `05` should solve training semantics first, not compression complexity.
- deployment compression belongs after the model is trained and validated.

## Rules

- Do not use `FP64` as the canonical format.
- Do not use `AMP`, `BF16`, or `FP16` in phase `05`.
- Do not treat `INT4` as a default deployment target.
- Always export from the canonical `FP32` model, not from an already compressed variant.
