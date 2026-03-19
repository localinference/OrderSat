# ONNX Export Walkthrough

This module exports the canonical trained PyTorch seq2seq checkpoint from
`src/05_pytorch_models/{language}` into a single-file `FP32` ONNX bundle under
`src/06_FP32_export_onnx_models/{language}`.

It is intentionally minimal in this phase:

- load the current `best.pt`
- rebuild the exact trained model shape
- export one `FP32` `model.onnx`
- validate ONNX Runtime parity against PyTorch
- copy the tokenizer artifacts needed by downstream inference

It does not quantize. That belongs in later modules.

## Entrypoint

Run:

```powershell
python cli/06_export_to_onnx/__main__.py --language eng --opset-version 18
```

The entrypoint is [**main**.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/__main__.py).

## High-Level Flow

The exporter does this, in this order:

1. Parse CLI args from [parse.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/args/parse.py).
2. Resolve source and destination paths in [consturctor.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/ExportPaths/consturctor.py).
3. Load `best.pt` and read the exact model-export metadata in [load.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/checkpoint/load.py).
4. Rebuild the PyTorch seq2seq model from the checkpoint metadata in [build.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/pytorch_model/build.py).
5. Wrap the model in a clean ONNX-facing forward signature through [constructor.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/OnnxExportWrapper/constructor.py).
6. Export a single-file `FP32` ONNX graph in [export.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/onnx_model/export.py).
7. Validate ONNX structure and numeric parity against PyTorch in [export.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/onnx_model/export.py).
8. Copy tokenizer artifacts and write the export metadata bundle in [write.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/artifacts/write.py).

## What It Reads

For `--language eng`, the exporter reads:

- checkpoint from `src/05_pytorch_models/eng/best.pt`
- tokenizer model from `src/03_tokenizers/eng/tokenizer.model`

Those paths are resolved by `build_export_paths()`.

## What It Writes

For `--language eng`, the exporter writes:

- ONNX model to `src/06_FP32_export_onnx_models/eng/model.onnx`
- tokenizer model to `src/06_FP32_export_onnx_models/eng/tokenizer.model`
- export metadata to `src/06_FP32_export_onnx_models/eng/config.json`

It does not intentionally produce `model.onnx.data`. This module’s rule is
single-file ONNX only.

## Why Each Step Exists

### Path Resolution

[consturctor.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/ExportPaths/consturctor.py)
keeps the export contract explicit:

- one source checkpoint per language
- one tokenizer model per language
- one canonical ONNX export directory per language

That keeps downstream modules on stable paths.

### Checkpoint Metadata Loading

[load.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/checkpoint/load.py)
does not guess model shape from filenames or loose defaults. It reads the exact
metadata saved by `05`, including:

- vocabulary size
- `pad_id`, `bos_id`, `eos_id`
- model width and depth
- feed-forward dimension
- dropout
- maximum source and target positions

Why:

- export quality depends on rebuilding the exact trained architecture
- wrong ids or position sizes would silently corrupt inference behavior

### Exact PyTorch Model Reconstruction

[build.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/pytorch_model/build.py)
loads the same `Seq2SeqTransformer` class used by `05` and restores the exact
`model_state_dict`.

Why:

- the exporter should operate on the real trained model, not a second
  hand-written inference variant
- this keeps `06` aligned with the actual `05` training result

### Clean ONNX Forward Signature

[constructor.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/OnnxExportWrapper/constructor.py)
wraps the model so the ONNX graph has a simple inference-facing signature:

- `input_ids`
- `attention_mask`
- `decoder_input_ids`
- `logits`

Why:

- ONNX export should expose only the runtime inputs that matter
- a small wrapper is cleaner than forcing exporter logic into the training model

### Single-File FP32 Export

[export.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/onnx_model/export.py)
exports `model.onnx` in `FP32` and explicitly removes any stale
`model.onnx.data` sidecar before export.

Why:

- this phase wants one canonical `FP32` ONNX file
- external data files do not improve model quality
- if the graph does not actually require external tensor storage, a `.data`
  file is only dead weight

### Exporting the Real Transformer Safely

The exporter temporarily disables PyTorch MHA fastpath during export, then
restores the previous setting immediately afterward.

Why:

- PyTorch can route Transformer layers through fused internal operators that do
  not export cleanly to ONNX
- disabling the fastpath affects export tracing, not the learned weights
- restoring the previous setting avoids leaking exporter-specific behavior into
  the rest of the process

### Validation

[export.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/onnx_model/export.py)
validates quality in two ways:

1. `onnx.checker.check_model(...)` verifies structural ONNX validity.
2. ONNX Runtime output is compared against PyTorch output on the same dummy
   inputs, and `max_abs_diff` is reported.

Why:

- “the file exported” is not enough
- the export must also be numerically faithful to the PyTorch checkpoint

### Export Bundle Metadata

[write.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/artifacts/write.py)
writes `config.json` with:

- source checkpoint path
- source metrics
- exact model config
- validation result
- ONNX input/output names
- opset version

Why:

- downstream modules should not have to rediscover how the ONNX model was built
- the quality check result should travel with the exported artifact

## Quality Guarantees This Module Currently Makes

The current exporter is built to guarantee these things:

1. It exports the exact trained `05` checkpoint, not an approximate rewrite.
2. It preserves token-id conventions and position sizes from checkpoint
   metadata.
3. It produces a single-file `FP32` ONNX model, not an external-data bundle.
4. It validates the emitted graph with ONNX checker.
5. It validates ONNX Runtime parity against PyTorch and records the result in
   `config.json`.

For the current `eng` export, the parity check is already tight: the observed
`max_abs_diff` is on the order of `1e-6`.

## Terminal Output

This module currently prints a short export summary instead of the richer stage
logging used by `05`.

You can observe:

- language
- checkpoint path
- ONNX model path
- tokenizer artifact paths
- config path
- validation output shape
- numeric parity as `max_abs_diff`

## Design Intent

The exporter is built around these rules:

1. Export the exact trained model from `05`.
2. Keep the ONNX bundle single-file and `FP32` in this phase.
3. Validate runtime parity instead of trusting the exporter blindly.
4. Copy only the artifacts that the runtime actually needs.
5. Keep the module small, explicit, and easy to audit.
