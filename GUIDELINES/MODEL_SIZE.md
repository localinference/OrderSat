# Model Size and Precision Guidelines

Use one canonical model format and derive deployment variants from it.

## Default rule

Use this exact path:

1. train and keep the canonical model in `FP32`
2. export one canonical `FP32` ONNX model
3. create deployment variants from that export:
   GPU: `FP16`
   CPU: `INT8`

This is the project guideline.

## Why

- `FP32` is the correct canonical baseline for this model class
- GPU deployment usually benefits first from `FP16`
- CPU deployment usually benefits first from `INT8`
- this keeps the workflow simple, predictable, and aligned with standard deployment tooling

## Rules

- do not use `FP64` as the canonical training or export format
- do not treat `INT4` as a default deployment target
- always export from the canonical `FP32` model, not from an already compressed variant
- evaluate task quality after conversion, not just latency or model size

## Summary

- canonical model: `FP32`
- export format: `FP32 ONNX`
- GPU deployment: `FP16`
- CPU deployment: `INT8`
