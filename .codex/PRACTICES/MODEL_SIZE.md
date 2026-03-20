# Model Size and Precision Practices

Precision policy is phase-bound and deployment-bound, not sample-count-bound.

## Canonical precision path

The current best-practice path is:

1. Train in `FP32`
2. Save checkpoints in `FP32`
3. Export one canonical single-file `FP32` ONNX model
4. Derive deployment variants from that canonical export

Current deployment variants:

- WebGPU: `mixed-fp16`
- WASM / CPU: `uint8`

## Why this is the current best practice

- `FP32` is the stable canonical training and checkpoint format.
- Export validation is easier when there is one authoritative ONNX source.
- GPU and browser CPU targets have different optimal compression paths.
- The canonical model should stay reversible and inspectable.

## Implemented project rules

- `05` is `FP32` only.
- `06` exports a single-file `FP32` ONNX bundle with no external data sidecar.
- `07` derives a `mixed-fp16` model from `06`.
- `08` derives a `uint8` model from `06`.
- `tokenizer.model` is copied unchanged across `06`, `07`, and `08`.

## What not to do

- Do not use `FP64` as the canonical format.
- Do not introduce mixed precision into `05`.
- Do not quantize directly from a non-canonical or already compressed model.
- Do not default to `INT4` for this pipeline.
- Do not keep `.onnx.data` sidecars when the graph does not require them.

## Rules

- Treat `06` as the only export source of truth.
- Treat `07` and `08` as derived artifacts, not independent export pipelines.
- Keep deployment precision choices separate from tokenizer choice and
  checkpoint reuse.

## Sources

- ONNX Runtime float16 and mixed precision:
  https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
- ONNX Runtime quantization:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime Web performance guidance:
  https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
