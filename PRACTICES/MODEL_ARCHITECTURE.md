# Model Architecture Practices

Only capacity knobs should scale with data and machine capability.

The formulas below are project heuristics. They are informed by Transformer and
scaling literature, but they are not universal constants.

## Scaling inputs

Use:

- `trainCount`, not total `sampleCount`
- `device_scale` from `get_device_capabilities()`

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`
- `DEVICE_SCALE = capabilities.device_scale`
- `CAPACITY_SCALE = min(DATA_SCALE, DEVICE_SCALE)`

Why:

- useful model capacity grows sublinearly with data
- the machine can be the bottleneck even when the data could justify more
- the data can be the bottleneck even when the machine could carry more

## Implemented base architecture

At `CAPACITY_SCALE = 1.0`, the current base shape is:

- `D_MODEL_BASE = 256`
- `ENCODER_LAYERS_BASE = 4`
- `DECODER_LAYERS_BASE = 4`
- `FF_RATIO = 4`
- `DROPOUT_BASE = 0.10`

## Implemented derived architecture

The current code in `05` derives:

- `D_MODEL = clamp(round_to_multiple(256 * CAPACITY_SCALE, 64), 128, 512)`
- `ATTENTION_HEADS = 4` if `D_MODEL <= 256`, else `8`
- `ENCODER_LAYERS = clamp(round(4 * CAPACITY_SCALE), 2, 6)`
- `DECODER_LAYERS = clamp(round(4 * CAPACITY_SCALE), 2, 6)`
- `FF_DIMENSION = 4 * D_MODEL`
- `DROPOUT = clamp(0.10 / CAPACITY_SCALE, 0.10, 0.20)`

## Why this is the current best practice

- A small local seq2seq model needs width and FF capacity before it needs large
  depth.
- `D_MODEL % ATTENTION_HEADS == 0` must always hold.
- A `4x` FF expansion is standard Transformer practice.
- Lower-data regimes benefit from more regularization and smaller width.
- Local/browser-oriented deployment puts a hard ceiling on how large the model
  should be, even if the corpus eventually grows.

## Rules

- Prefer increasing width before increasing depth.
- Do not change sequence lengths to justify architecture changes.
- Do not increase heads without enough width to support them cleanly.
- Keep architecture decisions separate from deployment precision decisions.
- If the model family changes, rewrite this file instead of stretching the old
  rules.

## Sources

- Transformer architecture reference:
  https://arxiv.org/abs/1706.03762
- PyTorch Transformer API reference:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- Compute-optimal scaling motivation:
  https://arxiv.org/abs/2203.15556
