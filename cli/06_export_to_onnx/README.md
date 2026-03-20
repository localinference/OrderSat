# ONNX Export Walkthrough

This module exports one canonical `FP32` ONNX bundle for one language under
`src/06_fp32_export_onnx_models/{language}`.

It does not assume one fixed tokenizer format anymore. For the requested
language, it inspects the trained format candidates under `03`, `04`, and `05`,
chooses the better trained model for that language, and then exports only that
winner into the stable `06/{language}` bundle.

## Entrypoint

Run:

```powershell
python cli/06_export_to_onnx/__main__.py --language eng --opset-version 18
```

The entrypoint is [**main**.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/__main__.py).

## High-Level Flow

The exporter does this, in this order:

1. Parse CLI args in [parse.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/args/parse.py).
2. Discover trained candidates for the requested language in [discover.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/selection/discover.py).
3. Compare those candidates in [select.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/selection/select.py).
4. Resolve the winning checkpoint and tokenizer paths in [consturctor.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/ExportPaths/consturctor.py).
5. Load the winning `best.pt` and read its export metadata in [load.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/checkpoint/load.py).
6. Rebuild the exact trained PyTorch model in [build.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/pytorch_model/build.py).
7. Wrap it behind an ONNX-facing forward signature in [constructor.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/OnnxExportWrapper/constructor.py).
8. Export `model.onnx` and validate ONNX Runtime parity in [export.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/onnx_model/export.py).
9. Copy the winning `tokenizer.model` and write `config.json` in [write.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/artifacts/write.py).

## What It Reads

For `--language eng`, the selector/exporter reads per-format candidates from:

- `src/03_tokenizers/eng/{format}/tokenizer.model`
- `src/04_training_datasets/eng/{format}/stats.json`
- `src/04_training_datasets/eng/{format}/validation.jsonl`
- `src/05_pytorch_models/eng/{format}/best.pt`
- `src/05_pytorch_models/eng/{format}/best_metrics.json`
- `src/05_pytorch_models/eng/{format}/run.json`

`{format}` is typically `bpe` or `unigram`.

## What It Writes

For `--language eng`, the exporter writes one stable bundle:

- `src/06_fp32_export_onnx_models/eng/model.onnx`
- `src/06_fp32_export_onnx_models/eng/tokenizer.model`
- `src/06_fp32_export_onnx_models/eng/config.json`

It does not intentionally produce `model.onnx.data`. This phase stays
single-file `FP32`.

## Why The Selector Exists

`03`, `04`, and `05` now produce multiple trained candidates per language. `06`
has to choose one of them.

The selection policy is language-local and deterministic:

1. highest validation exact-match Wilson lower bound
2. highest raw validation exact match
3. lowest validation bits per output character
4. lowest runtime proxy
5. lowest parameter count
6. lowest `label_p95`
7. stable lexical tiebreak by format name

This keeps exact match primary, but still lets `06` choose a winner when exact
match ties.

## Why Raw Validation Loss Is Not Enough

Raw validation loss is not directly comparable across tokenizer formats because
the token units differ.

Example:

- `bpe` may use fewer, larger tokens
- `unigram` may use more, smaller tokens

A lower per-token loss under one tokenizer does not automatically mean the model
represents the output sequence better overall. That is why the selector
normalizes by validation output characters:

- `validation_bits_per_output_char`

That produces a fairer cross-tokenizer likelihood comparison when exact match is
tied.

## Fairness Checks Before Comparison

The selector refuses to compare candidates unless:

- all required files exist
- `language` and `format` agree across `04` and `05`
- `validation_exact_match_ran` is true
- validation sample-id sets match across candidates
- total validation output-character counts match across candidates

Those checks matter because tokenizer comparisons are only meaningful if they
are based on the same held-out examples.

## Exact Model Reconstruction

[load.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/checkpoint/load.py)
reads the exact training/export metadata from the winning checkpoint, including:

- vocabulary size
- `pad_id`, `bos_id`, `eos_id`
- model width and depth
- feed-forward dimension
- dropout
- maximum source and target positions

[build.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/pytorch_model/build.py)
then rebuilds that exact `Seq2SeqTransformer` and loads the saved weights.

Why:

- `06` must export the real trained model, not an approximate rewrite
- wrong ids or wrong sequence limits would silently corrupt inference

## ONNX Export And Validation

[export.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/onnx_model/export.py)
uses the modern `torch.onnx.export(..., dynamo=True)` path so source and target
sequence lengths stay dynamic in the emitted graph.

It then does two separate quality checks:

1. `onnx.checker.check_model(...)`
2. ONNX Runtime vs PyTorch numeric parity across multiple validation shapes, not
   just one `8x8` reference input

That parity result is written into `config.json` as part of the export record.

## Export Metadata

[write.py](C:/Users/jorts/OrderSaT/cli/06_export_to_onnx/artifacts/write.py)
writes `config.json` with:

- selected format
- source checkpoint path
- source training metrics
- exact model config
- selection analysis and all candidate scores
- ONNX validation result
- dynamic-shape validation cases
- opset and IO names

So downstream modules can consume the canonical `06/{language}` bundle without
having to rediscover why that format won.

## Terminal Output

This module prints a short summary:

- language
- selected format
- winning checkpoint path
- ONNX model path
- tokenizer model path
- config path
- selection confidence
- selection reason
- ONNX validation reference shape and worst-case `max_abs_diff`
- validated case count

## Design Intent

The exporter is built around these rules:

1. Compare only candidates for the requested language.
2. Keep `06` output stable even if upstream formats multiply.
3. Prefer exact-match quality first.
4. Use character-normalized likelihood when token losses are not directly comparable.
5. Export one canonical single-file `FP32` ONNX bundle plus the matching tokenizer.
6. Reject fake-dynamic exports by validating non-`8x8` sequence shapes before
   declaring the export good.
