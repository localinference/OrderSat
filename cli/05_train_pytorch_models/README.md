# Seq2Seq Trainer Walkthrough

This module trains the project’s PyTorch encoder-decoder model from the
tokenized JSONL datasets in `src/04_training_datasets/{language}/{format}`
using the matching tokenizer vocabulary in `src/03_tokenizers/{language}/{format}`.

It is intentionally `FP32`-only in this phase. Mixed precision belongs in a
later module.

## Entrypoint

Run one format:

```powershell
python cli/05_train_pytorch_model/__main__.py --language eng --format bpe --device auto --checkpoint-mode auto
```

Run all discovered formats in parallel:

```powershell
python cli/05_train_pytorch_model/__main__.py --language eng --format all --device auto --checkpoint-mode auto
```

The entrypoint is [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).

## High-Level Flow

The trainer does this, in this order:

1. Parse CLI args from [parse.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/args/parse.py).
2. If `--format all` is requested, discover the available format names in [discover.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/formats/discover.py).
3. If multiple formats are discovered and `--sequential-formats` is not set, launch one child training process per format through [run_formats.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/orchestration/run_formats.py).
4. For each concrete `{language, format}` run, set reproducible seeds in [set.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/seed/set.py).
5. Resolve tokenizer, dataset, stats, and save paths in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
6. Read dataset stats from [parse.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/stats/parse.py).
7. Detect environment and device capability in [capabilities.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/device/capabilities.py).
8. Build the adaptive training config in [build.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/config/build.py).
9. Read tokenizer vocab size from [read_size.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/vocab/read_size.py).
10. Index tokenized train and validation datasets from [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/TokenizedJsonlDataset/constructor.py).
11. Resolve effective sequence lengths in [get_effective_lenght.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/sequence/get_effective_lenght.py).
12. Build the collator in [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/Seq2SeqCollator/constructor.py).
13. Build token-budgeted, length-bucketed train, train-audit, and validation loaders in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
14. Build the model in [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/Seq2SeqTransformer/constructor.py).
15. Build the optimizer in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
16. Resolve checkpoint reuse policy in [load.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/checkpoint/load.py).
17. Train epoch-by-epoch with [train.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/epoch/train.py).
18. Evaluate validation loss every epoch with [evaluate.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/loss/evaluate.py).
19. Run full validation exact match only on the configured cadence with [compute.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/match/compute.py).
20. Rank checkpoints by validation exact match first and validation loss second using [rank.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/selection/rank.py).
21. Persist run state every epoch and refresh `best.pt` only when a checkpoint actually beats the current canonical best through [save.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/artifacts/save.py).
22. Run a final full train exact-match audit on the canonical best checkpoint.

## What It Reads

For `--language eng --format bpe`, the trainer reads:

- tokenizer vocab from `src/03_tokenizers/eng/bpe/tokenizer.vocab`
- training split from `src/04_training_datasets/eng/bpe/train.jsonl`
- validation split from `src/04_training_datasets/eng/bpe/validation.jsonl`
- dataset stats from `src/04_training_datasets/eng/bpe/stats.json`
- optional checkpoint from `src/05_pytorch_models/eng/bpe/best.pt`

For `--format all`, the parent run first discovers all format directories that
have both tokenizer and dataset artifacts, then launches one concrete training
run per format.

## What It Writes

For `--language eng --format bpe`, the trainer writes:

- `src/05_pytorch_models/eng/bpe/best.pt`
- `src/05_pytorch_models/eng/bpe/best_metrics.json`
- `src/05_pytorch_models/eng/bpe/history.json`
- `src/05_pytorch_models/eng/bpe/run.json`

The same structure is written independently for `unigram`.

## Checkpoint Reuse

`--checkpoint-mode` controls how the current `best.pt` inside each
`src/05_pytorch_models/{language}/{format}` directory is used.

Supported modes:

- `auto`
  Loads the current `best.pt` as a warm-start only when it is compatible.
- `fresh`
  Ignores `best.pt`.
- `warm_start`
  Requires a compatible `best.pt` and loads model weights only.
- `resume`
  Requires a compatible `best.pt` with optimizer and runtime state and resumes the same optimization run.

Compatibility is checked in [load.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/checkpoint/load.py).

## Why The Structure Changed

The trainer is now format-aware because `03` and `04` no longer produce one
flat tokenizer or one flat dataset per language.

That means:

- `bpe` must train against `bpe` artifacts only
- `unigram` must train against `unigram` artifacts only
- each format must keep its own checkpoint history and best model

Without that separation, `05` would mix incompatible vocabularies, dataset
length profiles, and checkpoints.

## Adaptive Training Logic

[build.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/config/build.py)
derives:

- model width and depth
- dropout
- token budget per batch
- accumulation steps
- effective batch size
- learning rate
- weight decay
- epoch count
- early stopping settings
- validation exact-match cadence
- whether to run a final full train exact-match audit

The adjusted options are always logged as a JSON block so the scaling decision
is visible in the terminal.

## Exact Match Policy

This module is correctness-first, but it does not waste the most expensive
exact-match pass every epoch.

The current policy is:

- validation loss every epoch
- validation exact match on the configured cadence
- full train exact match only once at the end as a memorization audit

Checkpoint selection is:

1. validation exact match
2. validation loss

That makes the saved `best.pt` reflect the real end task rather than the
training objective alone.

## Terminal Logging

Logging is produced by [log.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/reporting/log.py).

You can observe:

- parsed args
- discovered formats
- parallel orchestration start and completion
- resolved paths
- stats parsing
- device capabilities
- adaptive config build
- adjusted options JSON
- dataset load
- loader build with token-budget batch-plan summaries
- model build
- optimizer build
- checkpoint load decision
- train epoch timing
- validation loss timing
- validation exact-match timing
- checkpoint save events
- final train exact-match audit
- early stopping
- total run time

When `--format all` runs in parallel, child process output is prefixed with the
format name so `bpe` and `unigram` logs are still distinguishable in one
terminal stream.

## Design Intent

The trainer is built around these rules:

1. Keep every tokenizer format isolated from the others.
2. Use exact match as the checkpoint-selection metric.
3. Keep `FP32` training simple in this phase.
4. Scale model and batch policy from measured dataset and device facts.
5. Preserve one canonical best checkpoint per `{language, format}` pair.
