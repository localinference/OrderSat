# Model Architecture Guidelines

Architecture size should scale with the amount of supervised training data. For this repo's random-init seq2seq model, use one default per sample-count regime.

## Small data

If `sampleCount < 10_000`, use:

- `D_MODEL = 128`
- `ATTENTION_HEADS = 4`
- `ENCODER_LAYERS = 2`
- `DECODER_LAYERS = 2`
- `FF_DIMENSION = 512`
- `DROPOUT = 0.20`

Why: with small supervised datasets, the main risk is over-capacity and memorization, not lack of width.

## Medium data

If `10_000 <= sampleCount <= 100_000`, use:

- `D_MODEL = 256`
- `ATTENTION_HEADS = 4`
- `ENCODER_LAYERS = 4`
- `DECODER_LAYERS = 4`
- `FF_DIMENSION = 1024`
- `DROPOUT = 0.10`

Why: this is the point where a wider and deeper model usually starts paying for itself without becoming needlessly brittle.

## Large data

If `sampleCount > 100_000`, use:

- `D_MODEL = 512`
- `ATTENTION_HEADS = 8`
- `ENCODER_LAYERS = 6`
- `DECODER_LAYERS = 6`
- `FF_DIMENSION = 2048`
- `DROPOUT = 0.10`

Why: above this scale, the standard Transformer shape becomes a reasonable default from scratch.

## Rules

- Keep `D_MODEL % ATTENTION_HEADS == 0`.
- Prefer increasing width before increasing depth.
- Do not increase heads on a narrow model just because more heads sound better.
- Treat dropout as regularization, not as a substitute for sane model size.
