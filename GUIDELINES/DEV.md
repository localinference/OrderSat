# Development Guidelines

This file covers values that should stay fixed across normal experiments.

These are not scaling-law parameters and should not be tied to training-set size.

## Use these defaults

- `SEED = 7`
- `LOG_FREQUENCY = 1`
- `BOS_ID = 1`
- `EOS_ID = 2`
- `LABEL_PAD_ID = -100`
- `GRAD_CLIP = 1.0`

## Why

- `SEED` should stay fixed so runs are comparable.
- `LOG_FREQUENCY` is epoch-based in this trainer, so `1` is the clean default.
- `BOS_ID` and `EOS_ID` are tokenizer/control-token conventions, not tunable capacity knobs.
- `LABEL_PAD_ID = -100` matches the standard PyTorch ignore index for cross-entropy loss.
- `GRAD_CLIP = 1.0` is the safe default and should not be treated as a data-size scaling parameter.

## Rules

- Do not create fake sample-count logic for these values.
- Change them only for a concrete tokenizer, debugging, or numerical reason.
