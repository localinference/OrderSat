# Model Architecture Practices

Only capacity knobs should scale with data.

Use one multiplier for what the training set can justify, and another for what
the current machine can realistically carry.

## Data-scale rule

Use `trainCount`, not total `sampleCount`.

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

Why:

- useful capacity grows sublinearly, not linearly
- hard small or medium or large buckets throw away information
- low-data regimes usually generalize better with smaller models

## Device-scale rule

Read raw capability values from `get_device_capabilities()`.

Set:

- `DEVICE_SCALE = capabilities.device_scale`

This is a bounded hardware and environment multiplier derived from available
memory and host-side throughput.

## Final capacity rule

Set:

- `CAPACITY_SCALE = min(DATA_SCALE, DEVICE_SCALE)`

Why:

- the data may justify a larger model than the machine can train
- the machine may allow a larger model than the data can support
- the final architecture must respect both constraints

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

## Generalization rules

- Prefer the smallest model that achieves the needed validation exact match.
- Prefer increasing width before increasing depth.
- Do not increase heads without enough width to justify them.
- Keep `D_MODEL % ATTENTION_HEADS == 0`.
- Do not use `trainCount` or `DEVICE_SCALE` to set sequence lengths.
