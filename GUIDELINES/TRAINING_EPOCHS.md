# Training Epoch Guidelines

Epoch policy should scale with sample count because each epoch means something different at different dataset sizes.

## Small data

If `sampleCount < 10_000`, use:

- `EPOCHS = 100`
- `EARLY_STOPPING_PATIENCE = 20`
- `EARLY_STOPPING_MIN_DELTA = 1e-5`

Why: with small data, the model needs repeated exposure and validation loss is noisy.

## Medium data

If `10_000 <= sampleCount <= 100_000`, use:

- `EPOCHS = 30`
- `EARLY_STOPPING_PATIENCE = 8`
- `EARLY_STOPPING_MIN_DELTA = 1e-4`

Why: at this size, one epoch already covers meaningful variation, so training should stop earlier and ignore tiny validation noise.

## Large data

If `sampleCount > 100_000`, use:

- `EPOCHS = 10`
- `EARLY_STOPPING_PATIENCE = 3`
- `EARLY_STOPPING_MIN_DELTA = 1e-3`

Why: large datasets make each epoch expensive and small validation changes less meaningful.

## Rules

- Patience is epoch-based in this trainer.
- Keep early stopping enabled.
- If train exact match reaches `1.0` very early while validation remains weak, that is memorization, not success.
