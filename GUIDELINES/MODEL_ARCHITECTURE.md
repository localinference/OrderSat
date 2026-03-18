# Model Architecture Guidelines

Only capacity knobs should scale with dataset size.

Use one multiplier for what the data can justify, and a separate multiplier for what the current machine can realistically carry.

## Data-scale rule

Use `trainCount`, not total `sampleCount`.

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

Why:

- useful model capacity should grow sublinearly, not linearly
- fake small or medium or large buckets throw away information

## Device-scale rule

Read raw capability values from `get_device_capabilities()`.

Set:

- `DEVICE_SCALE = capabilities.device_scale`

This value is a bounded hardware and environment multiplier derived from available memory and host-side throughput.

## Final capacity rule

Set:

- `CAPACITY_SCALE = min(DATA_SCALE, DEVICE_SCALE)`

Why:

- data may justify a larger model than the machine can train
- hardware may allow a larger model than the dataset can support
- the final architecture should respect both constraints

## Base architecture

Use these base values at `CAPACITY_SCALE = 1.0`:

- `D_MODEL_BASE = 256`
- `ENCODER_LAYERS_BASE = 4`
- `DECODER_LAYERS_BASE = 4`
- `FF_RATIO = 4`
- `DROPOUT_BASE = 0.10`

## Derived architecture

Set:

- `D_MODEL = round_to_multiple(D_MODEL_BASE * CAPACITY_SCALE, 64)`
- `ATTENTION_HEADS = 4` if `D_MODEL <= 256`, else `8`
- `ENCODER_LAYERS = clamp(round(ENCODER_LAYERS_BASE * CAPACITY_SCALE), 2, 6)`
- `DECODER_LAYERS = clamp(round(DECODER_LAYERS_BASE * CAPACITY_SCALE), 2, 6)`
- `FF_DIMENSION = FF_RATIO * D_MODEL`
- `DROPOUT = clamp(DROPOUT_BASE / CAPACITY_SCALE, 0.10, 0.20)`

## Rules

- Keep `D_MODEL % ATTENTION_HEADS == 0`.
- Prefer increasing width before increasing depth.
- Do not increase heads without increasing width enough to justify them.
- Do not use `trainCount` or `DEVICE_SCALE` to set sequence lengths.
