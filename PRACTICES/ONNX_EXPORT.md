# ONNX Export and Runtime Practices

`06` is the canonical export and model-selection stage.

It should choose the best trained format for one language, export one canonical
FP32 ONNX bundle, and define the runtime contract that later deployment stages
inherit.

## Candidate discovery scope

For one `language`, compare only candidates that have all of:

- `src/05_pytorch_models/{language}/{format}/best.pt`
- `best_metrics.json`
- `run.json`
- matching `src/03_tokenizers/{language}/{format}/tokenizer.model`
- matching `src/04_training_datasets/{language}/{format}/stats.json`
- matching `validation.jsonl`

If those artifacts are not all present, the format is not exportable.

## Format-selection rule

Current implemented ranking in `06`:

1. higher validation exact-match Wilson lower bound
2. higher validation exact match
3. lower validation bits per output character
4. lower runtime proxy
5. lower parameter count
6. lower label `p95`
7. stable lexical tie-breaker

Why:

- exact match is the true task metric
- Wilson lower bound is a better primary comparator than raw rate alone on
  small validation sets
- raw token loss is not directly comparable across different tokenizers
- deployment cost should matter only after quality

## Canonical export rule

Export one single-file `FP32` ONNX model in:

- `src/06_fp32_export_onnx_models/{language}`

Bundle:

- `model.onnx`
- `tokenizer.model`
- `config.json`

Do not export `tokenizer.vocab`.
Do not use ONNX external data sidecars for this project.

## Validation rule

`06` should validate the exported ONNX model against PyTorch on multiple cases,
not just one short reference case.

Current implemented case set includes:

- reference
- longer source
- longer target
- max source with decode start
- wider window

`07` and `08` should reuse that same validation case list whenever it is
present in `06/config.json`.

## Runtime I/O contract

The current best public input contract is:

- `input_ids: int32`
- `attention_mask: int32`
- `decoder_input_ids: int32`

Why:

- token ids are indices, not continuous values
- `int32` is enough for this project’s vocab and positional ranges
- `int32` is easier to handle in JS and ORT Web than `int64`
- floats are the wrong semantic type for embedding indices

A future bool mask is acceptable, but the current project contract keeps the
three public inputs uniform as `int32`.

## Deployment rule

Derive deployment artifacts only from the canonical `06` export:

- `07` for WebGPU `mixed-fp16`
- `08` for WASM `uint8`

Do not treat `07` or `08` as alternative sources of truth.

## Sources

- PyTorch Embedding index types:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- ONNX Gather index types:
  https://onnx.ai/onnx/operators/onnx__Gather.html
- ONNX Runtime float16 and mixed precision:
  https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
- ONNX Runtime quantization:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime Web performance guidance:
  https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
