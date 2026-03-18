# Model Learning Guidelines

Optimization is primarily data-scale-driven, not hardware-driven.

Device capability should affect learning settings only if it forces the achieved effective batch away from the target effective batch.

## Data-scale rule

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

Use `trainCount`, not total `sampleCount`.

## Effective-batch rule

Set:

- `BATCH_SCALE = sqrt(achieved_effective_batch / target_effective_batch)`

If accumulation keeps effective batch near target, then `BATCH_SCALE` stays near `1.0` and the optimizer does not need a device-specific adjustment.

## Base optimization

Use these base values at `DATA_SCALE = 1.0` and `BATCH_SCALE = 1.0`:

- `LEARNING_RATE_BASE = 2e-4`
- `WEIGHT_DECAY_BASE = 1e-4`
- `GRAD_CLIP = 1.0`

## Derived optimization

Set:

- `LEARNING_RATE = clamp((LEARNING_RATE_BASE / sqrt(DATA_SCALE)) * BATCH_SCALE, 1e-4, 3e-4)`
- `WEIGHT_DECAY = clamp(WEIGHT_DECAY_BASE / (DATA_SCALE ** 2), 1e-4, 5e-4)`
- `GRAD_CLIP = 1.0`

## Why

- larger data scale usually wants a more conservative peak learning rate
- smaller supervised datasets usually need stronger regularization
- learning rate should follow achieved effective batch, not raw device labels
- gradient clipping is a stability guard, not a scale knob

## Rules

- Use AdamW.
- If a scheduler is added later, treat the value above as the peak learning rate.
- Lower learning rate before touching `GRAD_CLIP`.
