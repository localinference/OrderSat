# Checkpointing and Reuse Practices

This project intentionally keeps one canonical model directory per language.

That policy is correct, but checkpoint reuse must still distinguish between full
resume and weight warm-start.

## Canonical files

Within `src/05_pytorch_models/{language}`:

- `best.pt` is the current best validation checkpoint
- `best_metrics.json` stores the current best validation metrics
- `history.json` stores the current run history
- `run.json` stores the current run metadata

## Best-checkpoint rule

`best.pt` should represent the strongest known generalizing checkpoint.

Rank checkpoints by:

1. validation exact match
2. validation loss as tie-breaker

Do not rank checkpoints by train exact match.

## Reuse modes

Use one of these modes explicitly.

### Full resume

Use full resume only when:

- the previous run was interrupted
- tokenizer and vocab are unchanged
- architecture is unchanged
- dataset semantics are unchanged
- you want to continue the same optimization run

In this mode:

- load model weights
- load optimizer state
- restore relevant history and runtime state

### Weight warm-start

Use weight warm-start when:

- the task is still the same
- tokenizer and vocab are unchanged
- architecture is unchanged
- you want to continue training from the best known weights in a new run

In this mode:

- load model weights from `best.pt`
- reset the optimizer
- reset patience and run history

This is the default reuse mode for generalization-focused retraining.

### Fresh start

Use a fresh start when any of these changed materially:

- vocab size
- tokenization
- architecture shape
- label semantics

## Compatibility rules

Before loading `best.pt`, verify compatibility of at least:

- vocab size
- pad or BOS or EOS conventions
- `d_model`
- attention heads
- encoder layers
- decoder layers
- feed-forward dimension
- max source positions
- max target positions

## Why

- best known weights should not be thrown away
- stale optimizer state should not be reused blindly across materially new runs
- full resume and warm-start solve different problems
