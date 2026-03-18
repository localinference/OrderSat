# Training Schedule and Evaluation Practices

Generalization is the goal.

That means the trainer must separate cheap signals that should run every epoch
from expensive signals that should run only when justified.

## Data-scale rule

Use `trainCount`, not total `sampleCount`.

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

## Base schedule

Use these base values at `DATA_SCALE = 1.0`:

- `EPOCHS_BASE = 30`
- `EARLY_STOPPING_PATIENCE_BASE = 8`
- `EARLY_STOPPING_MIN_DELTA_BASE = 1e-4`
- `FULL_VALIDATION_EXACT_MATCH_EVERY_BASE = 2`

## Derived schedule

Set:

- `EPOCHS = clamp(round(EPOCHS_BASE / (DATA_SCALE ** 2)), 10, 100)`
- `EARLY_STOPPING_PATIENCE = clamp(round(EARLY_STOPPING_PATIENCE_BASE / DATA_SCALE), 3, 20)`
- `EARLY_STOPPING_MIN_DELTA = clamp(EARLY_STOPPING_MIN_DELTA_BASE * DATA_SCALE, 1e-5, 1e-3)`
- `FULL_VALIDATION_EXACT_MATCH_EVERY_N_EPOCHS = clamp(round(FULL_VALIDATION_EXACT_MATCH_EVERY_BASE * DATA_SCALE), 1, 3)`

Larger datasets should run full exact match less often because the cost scales
with sample count and decode length.

## Evaluation cadence

Run every epoch:

- training loss
- validation loss
- structural-validity checks on validation outputs when available

Run periodically:

- full validation exact match

Run rarely or only at the end:

- full train exact match

Why:

- validation exact match measures generalization
- train exact match mostly measures memorization
- full autoregressive exact-match sweeps are expensive

## Selection and stopping rules

- Select the canonical best checkpoint by validation exact match first.
- Use validation loss as the tie-breaker and fallback metric.
- Do not stop early because train exact match reached `1.0`.
- Treat perfect early train exact match with weaker validation as memorization.
- Keep early stopping enabled.
