# Training Schedule and Evaluation Practices

Generalization is the goal.

That means the trainer should run cheap signals frequently and expensive signals
only when they are justified.

## Implemented schedule scaling

The current trainer uses:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

Then:

- `EPOCHS = clamp(round(30 / (DATA_SCALE ** 2)), 10, 100)`
- `EARLY_STOPPING_PATIENCE = clamp(round(8 / DATA_SCALE), 3, 20)`
- `EARLY_STOPPING_MIN_DELTA = clamp(1e-4 * DATA_SCALE, 1e-5, 1e-3)`
- `VALIDATION_EXACT_MATCH_FREQUENCY = clamp(round(2 * DATA_SCALE), 1, 3)`

Larger datasets should run full validation exact match less often because exact
match cost scales with both sample count and decode length.

## Current cadence policy

Run every epoch:

- training loss
- validation loss

Run on the configured cadence:

- full validation exact match

Run only at the end:

- full train exact match audit

## Why this is the current best practice

- validation exact match measures the real task outcome
- validation loss is cheap and useful every epoch
- full autoregressive exact-match sweeps are expensive
- train exact match mostly tells you about memorization

## Early stopping rule

Early stopping should follow the checkpoint objective when exact match is part
of the schedule.

Current implemented behavior:

- if validation exact match is scheduled, patience is counted on those
  evaluation windows
- if exact match is disabled, patience falls back to validation loss

This avoids stopping a run just because loss moved while the true checkpoint
metric was not evaluated yet.

## Rules

- Select best checkpoints by validation exact match first.
- Use validation loss as tie-breaker or fallback only.
- Do not stop early because train exact match hit `1.0`.
- Keep final train exact match as an audit, not a checkpoint metric.

## Sources

- Transformer seq2seq reference:
  https://arxiv.org/abs/1706.03762
- Compute-optimal scaling motivation:
  https://arxiv.org/abs/2203.15556
