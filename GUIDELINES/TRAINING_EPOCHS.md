# Training Epoch Guidelines

Training schedule should scale with training-set size, not with raw device labels.

Device capability matters to wall-clock time, but it should not change the optimization target unless it forces a materially different effective batch.

## Data-scale rule

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

Use `trainCount`, not total `sampleCount`.

## Base schedule

Use these base values at `DATA_SCALE = 1.0`:

- `EPOCHS_BASE = 30`
- `EARLY_STOPPING_PATIENCE_BASE = 8`
- `EARLY_STOPPING_MIN_DELTA_BASE = 1e-4`
- `EXACT_MATCH_FREQUENCY_BASE = 2`

## Derived schedule

Set:

- `EPOCHS = clamp(round(EPOCHS_BASE / (DATA_SCALE ** 2)), 10, 100)`
- `EARLY_STOPPING_PATIENCE = clamp(round(EARLY_STOPPING_PATIENCE_BASE / DATA_SCALE), 3, 20)`
- `EARLY_STOPPING_MIN_DELTA = clamp(EARLY_STOPPING_MIN_DELTA_BASE * DATA_SCALE, 1e-5, 1e-3)`
- `EXACT_MATCH_FREQUENCY = clamp(round(EXACT_MATCH_FREQUENCY_BASE * DATA_SCALE), 1, 3)`

## Rules

- Do not scale epochs from device class.
- If achieved effective batch differs materially from target, revisit learning-rate policy before changing schedule policy.
- Patience is epoch-based in this trainer.
- Keep early stopping enabled.
- If train exact match reaches `1.0` early while validation remains weak, that is memorization, not success.
