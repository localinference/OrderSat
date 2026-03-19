# WASM Optimizer Walkthrough

This module converts the canonical `FP32` ONNX export from
`src/06_FP32_export_onnx_models/{language}` into a WASM-targeted `uint8`
quantized ONNX build under `src/08_INT8_cpu_onnx_models/{language}`.

It is intentionally focused on the browser CPU/WASM deployment case:

- read the current `06` FP32 ONNX model
- preprocess it for quantization
- quantize it with `QUInt8` dynamic quantization
- validate the quantized model against the FP32 source model
- copy only the tokenizer model needed by runtime

## Entrypoint

Run:

```powershell
python cli/08_optimize_for_wasm/__main__.py --language eng
```

The entrypoint is [__main__.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/__main__.py).

## High-Level Flow

The optimizer does this, in this order:

1. Parse CLI args from [parse.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/args/parse.py).
2. Resolve source and destination paths in [consturctor.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/QuantizationPaths/consturctor.py).
3. Require the source FP32 ONNX model, source export config, and tokenizer model in [require.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/file/require.py).
4. Quantize the FP32 model to `QUInt8` in [uint8.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/quantize/uint8.py).
5. Validate the quantized model against the FP32 source ONNX in [validate.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/quantize/validate.py).
6. Copy the tokenizer model in [copy.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/file/copy.py).
7. Write the quantized build metadata in [write.py](C:/Users/jorts/OrderSaT/cli/08_optimize_for_wasm/quantize/write.py).

## What It Reads

For `--language eng`, the optimizer reads:

- FP32 ONNX model from `src/06_FP32_export_onnx_models/eng/model.onnx`
- FP32 export config from `src/06_FP32_export_onnx_models/eng/config.json`
- tokenizer model from `src/06_FP32_export_onnx_models/eng/tokenizer.model`

## What It Writes

For `--language eng`, the optimizer writes:

- quantized ONNX model to `src/08_INT8_cpu_onnx_models/eng/model.uint8.onnx`
- tokenizer model to `src/08_INT8_cpu_onnx_models/eng/tokenizer.model`
- quantized build config to `src/08_INT8_cpu_onnx_models/eng/config.json`

It does not intentionally produce `model.uint8.onnx.data`. This module’s rule
is single-file ONNX only.

## Design Intent

The optimizer is built around these rules:

1. Keep `06` as the canonical FP32 ONNX source.
2. Produce a separate WASM-targeted quantized artifact instead of mutating `06`.
3. Prefer `uint8` quantization for the browser CPU/WASM case.
4. Validate quantized output behavior against the FP32 ONNX source model.
5. Copy only the runtime artifacts that are actually needed.
