# Model Learning Guidelines

Learning settings should scale with data size, but only for the knobs that actually change learning dynamics.

## Learning rate

- if `sampleCount < 10_000`, use `LEARNING_RATE = 3e-4`
- if `10_000 <= sampleCount <= 100_000`, use `LEARNING_RATE = 2e-4`
- if `sampleCount > 100_000`, use `LEARNING_RATE = 1e-4`

Why: this trainer does not use warmup or a decay scheduler, so the safest default is to lower the peak rate as model and dataset scale up.

## Weight decay

- if `sampleCount < 10_000`, use `WEIGHT_DECAY = 5e-4`
- if `10_000 <= sampleCount <= 100_000`, use `WEIGHT_DECAY = 1e-4`
- if `sampleCount > 100_000`, use `WEIGHT_DECAY = 1e-4`

Why: small supervised datasets benefit from stronger regularization. Once data grows, mild decay is usually enough.

## Gradient clipping

Use `GRAD_CLIP = 1.0`.

Why: for this model family, `1.0` is still the safest default and should not be tuned before learning rate and sequence lengths are sane.

## Rules

- Use AdamW, not Adam with fake L2 weight decay.
- If training is unstable, lower learning rate before changing gradient clipping.
- If the model memorizes training data early, raise regularization before making the model bigger.
- If a scheduler is added later, treat the values above as peak learning rates.
