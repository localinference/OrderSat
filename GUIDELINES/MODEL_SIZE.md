# Model Size and Precision Guidelines

Precision policy is deployment-target-bound, not training-size-bound.

Do not tie this file to scaling-law logic or `trainCount`.

## Capability inputs

Read precision support from `get_device_capabilities()`.

Use:

- `capabilities.supports_fp16`
- `capabilities.supports_bf16`

These values matter for training-time mixed precision support, not for the canonical checkpoint format.

## Canonical path

Use this path:

1. Train and keep the canonical model in `FP32`.
2. Export one canonical `FP32 ONNX` model.
3. Derive deployment variants from that export:
   GPU: `FP16`
   CPU: `INT8`

## Why

- `FP32` is the correct canonical baseline.
- GPU deployment usually benefits first from `FP16`.
- CPU deployment usually benefits first from `INT8`.
- device precision support may change training convenience, but it does not change the canonical export policy

## Rules

- Do not use `FP64` as the canonical format.
- Do not treat `INT4` as a default deployment target.
- Always export from the canonical `FP32` model, not from an already compressed variant.
- Evaluate task quality after conversion, not just latency or model size.
