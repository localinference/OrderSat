# Seq2Seq Trainer Walkthrough

This module trains the project’s PyTorch encoder-decoder model from the
tokenized JSONL datasets in `src/04_training_datasets/{language}` using the
tokenizer vocabulary in `src/03_tokenizers/{language}`.

It is intentionally `FP32`-only in this phase. Mixed precision belongs in a
later module.

## Entrypoint

Run:

```powershell
python cli/05_train_pytorch_model/__main__.py --language eng --device auto --checkpoint-mode auto
```

The entrypoint is [__main__.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).

## High-Level Flow

The trainer does this, in this order:

1. Parse CLI args from [parse.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/args/parse.py).
2. Set reproducible seeds in [set.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/seed/set.py).
3. Resolve tokenizer, dataset, stats, and save paths in [__main__.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
4. Read dataset stats from [parse.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/stats/parse.py).
5. Detect environment and device capability in [capabilities.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/device/capabilities.py).
6. Build the adaptive training config in [build.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/config/build.py).
7. Read tokenizer vocab size from [read_size.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/vocab/read_size.py).
8. Index tokenized train and validation datasets from [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/TokenizedJsonlDataset/constructor.py).
9. Resolve effective sequence lengths in [get_effective_lenght.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/sequence/get_effective_lenght.py).
10. Build the collator in [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/Seq2SeqCollator/constructor.py).
11. Build token-budgeted, length-bucketed train, train-audit, and validation loaders in [__main__.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
12. Build the model in [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/Seq2SeqTransformer/constructor.py).
13. Build the optimizer in [__main__.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
14. Resolve checkpoint reuse policy in [load.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/checkpoint/load.py).
15. Train epoch-by-epoch with [train.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/epoch/train.py).
16. Evaluate validation loss every epoch with [evaluate.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/loss/evaluate.py).
17. Run full validation exact match only on the configured cadence with [compute.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/match/compute.py).
18. Rank checkpoints by validation exact match first and validation loss second using [rank.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/selection/rank.py).
19. Persist run state every epoch and refresh `best.pt` only when a checkpoint actually beats the current canonical best through [save.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/artifacts/save.py).
20. Run a final full train exact-match audit on the canonical best checkpoint.

## What It Reads

For `--language eng`, the trainer reads:

- tokenizer vocab from `src/03_tokenizers/eng/tokenizer.vocab`
- training split from `src/04_training_datasets/eng/train.jsonl`
- validation split from `src/04_training_datasets/eng/validation.jsonl`
- dataset stats from `src/04_training_datasets/eng/stats.json`
- optional checkpoint from `src/05_pytorch_models/eng/best.pt`

Those paths are resolved by `build_run_paths()`.

## Checkpoint Reuse

`--checkpoint-mode` controls how the current `best.pt` is used.

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

The loader verifies:

- vocab and padding layout
- model width and depth
- feed-forward size
- position-embedding sizes
- BOS, EOS, and label-pad conventions when metadata is available

If the existing checkpoint is from an older incompatible tokenizer or model
shape, the trainer logs the mismatch and falls back to a fresh run in `auto`
mode.

## Why Each Step Exists

### Stats and Capabilities

The trainer reads measured dataset facts instead of guessing:

- `trainCount`
- `validationCount`
- tokenized input lengths
- tokenized label lengths

The trainer also reads machine capability facts:

- resolved device
- accelerator memory
- system memory
- CPU count
- bounded `device_scale`

Those two inputs are combined so model capacity reflects both what the data can
justify and what the machine can actually train.

### Adaptive Config

[build.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/config/build.py)
derives:

- model width and depth
- dropout
- estimated examples per batch
- max batch size
- target tokens per batch
- target tokens per optimizer step
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

### Indexed Lazy Dataset

The dataset layer no longer keeps the full JSONL payload resident in memory.

Instead [constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/TokenizedJsonlDataset/constructor.py) now:

- scans the file once
- validates each record
- stores byte offsets and length metadata
- reads records lazily by offset during `__getitem__`

Why:

- scaling to larger datasets should not require keeping every parsed record in memory
- the batcher still needs cheap length access
- one scan at startup is acceptable, but a fully resident dataset stops scaling

### Token-Budgeted Batching

The trainer no longer batches by a fixed raw sample count.

Instead it uses [sampler.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/batching/sampler.py), which:

- estimates batch cost as padded source tokens plus padded target tokens
- groups similar-length samples together
- caps each batch by a token budget
- still enforces a maximum batch size

Why:

- padding waste is a real cost in seq2seq training
- similar-length batches are cheaper to train and evaluate
- token budget is a better proxy for memory and step cost than sample count alone

### Training Objective vs Selection Objective

The trainer uses token-level cross-entropy for optimization.

That is the efficient training objective.

Checkpoint selection is different:

- primary metric: validation exact match
- tie-breaker: validation loss

This matters because the project is correctness-first. A lower loss does not
matter if the whole structured output is still wrong.

### Exact-Match Cost Control

Exact match stays central, but the expensive part is controlled.

The trainer now does this:

- validation loss every epoch
- full validation exact match only on the configured cadence
- full train exact match only once at the end as an audit

This reduces monitoring cost without removing exact match from the training
process.

### Target-Aware Exact Match

[compute.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/match/compute.py)
uses target-aware greedy decoding for exact-match evaluation.

That means:

- the decoder only runs up to the gold target length for that batch
- a sample stops as soon as exact match becomes impossible
- a sample also stops as soon as the correct `EOS` position is reached

This is much cheaper than always decoding to the dataset-wide maximum target
length.

### Canonical Best Checkpoint

The save policy is still intentionally overwrite-by-design:

- `src/05_pytorch_models/{language}/best.pt`
- `src/05_pytorch_models/{language}/best_metrics.json`
- `src/05_pytorch_models/{language}/history.json`
- `src/05_pytorch_models/{language}/run.json`

`best.pt` is the canonical best checkpoint for that language, not an archive of
every run.

The current run can warm-start from it, but a new checkpoint only overwrites it
if it actually beats the incumbent canonical best by the validation-first
ranking rule.

## Terminal Logging

Logging is produced by [log.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/reporting/log.py).

You can observe:

- parsed args and checkpoint mode
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

Each log block is separated by a blank line so the terminal output stays
readable during long runs.

## Design Intent

The trainer is built around these rules:

1. Read measured facts from the dataset instead of guessing.
2. Keep `FP32` training simple in this phase.
3. Use exact match as the correctness metric, but do not waste compute on the most expensive version of it every epoch.
4. Prefer validation generalization signals over train memorization signals.
5. Reuse the current canonical best checkpoint when compatible, but do not overwrite it with weaker candidates.
