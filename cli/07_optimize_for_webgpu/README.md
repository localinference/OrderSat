# WebGPU Optimizer Walkthrough

This module converts the canonical `FP32` ONNX export from
`src/06_fp32_export_onnx_models/{language}` into a WebGPU-targeted
`mixed-fp16` ONNX build under `src/07_mixed-fp16_gpu_onnx_models/{language}`.

It is intentionally focused on the browser GPU/WebGPU deployment case:

- read the current `06` FP32 ONNX model
- convert internal float tensors and ops to mixed-fp16
- keep the public model interface stable
- validate the mixed model against the FP32 source model
- copy only the tokenizer model needed by runtime

## Entrypoint

Run:

```powershell
python cli/07_optimize_for_webgpu/__main__.py --language eng
```

The entrypoint is [**main**.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/__main__.py).

## High-Level Flow

The optimizer does this, in this order:

1. Parse CLI args from [parse.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/args/parse.py).
2. Resolve source and destination paths in [consturctor.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/MixedPaths/consturctor.py).
3. Require the source FP32 ONNX model, source export config, and tokenizer model in [require.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/file/require.py).
4. Convert the FP32 model to mixed-fp16 in [fp16.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/mix/fp16.py).
5. Validate the mixed model against the FP32 source ONNX in [validate.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/mix/validate.py), reusing the full `06` validation case set when available.
6. Copy the tokenizer model in [copy.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/file/copy.py).
7. Write the mixed build metadata in [write.py](C:/Users/jorts/OrderSaT/cli/07_optimize_for_webgpu/mix/write.py).

## What It Reads

For `--language eng`, the optimizer reads:

- FP32 ONNX model from `src/06_fp32_export_onnx_models/eng/model.onnx`
- FP32 export config from `src/06_fp32_export_onnx_models/eng/config.json`
- tokenizer model from `src/06_fp32_export_onnx_models/eng/tokenizer.model`

## What It Writes

For `--language eng`, the optimizer writes:

- mixed-fp16 ONNX model to `src/07_mixed-fp16_gpu_onnx_models/eng/model.mixed-fp16.onnx`
- tokenizer model to `src/07_mixed-fp16_gpu_onnx_models/eng/tokenizer.model`
- mixed build config to `src/07_mixed-fp16_gpu_onnx_models/eng/config.json`

It does not intentionally produce `model.mixed-fp16.onnx.data`. This module’s
rule is single-file ONNX only.

When `06` provides `validation.cases`, this module validates the mixed model on
that same case list instead of only a single short reference case. That keeps
`07` aligned with the current dynamic-shape export contract.

## Design Intent

The optimizer is built around these rules:

1. Keep `06` as the canonical FP32 ONNX source.
2. Produce a separate WebGPU-targeted mixed-fp16 artifact instead of mutating `06`.
3. Keep I/O types stable while converting internal float compute to mixed-fp16.
4. Validate mixed output behavior against the FP32 ONNX source model.
5. Copy only the runtime artifacts that are actually needed.
