# Development Practices

This file covers trainer invariants that should stay fixed across normal
experiments.

These values are not capacity knobs and should not be tied to sample count,
device class, or convenience heuristics.

## Fixed defaults

- `SEED = 7`
- `LOG_FREQUENCY = 1`
- `BOS_ID = 1`
- `EOS_ID = 2`
- `LABEL_PAD_ID = -100`
- `GRAD_CLIP = 1.0`

## Phase scope

Phase `05_train_pytorch_model` is an `FP32` training phase.

That means:

- train in `FP32`
- save checkpoints in `FP32`
- keep mixed precision and deployment conversion for later modules

## Canonical run policy

Use one canonical save directory per language:

- `src/05_pytorch_models/{language}`

Within that directory:

- `best.pt` is the current best validation checkpoint
- `history.json` is the epoch history for the current canonical run
- `run.json` is the current run metadata
- overwriting these files is intentional

## Logging policy

Every run should always log:

- resolved paths
- device capabilities
- data-scale and compute-scale adjusted options
- stage timings
- epoch timings
- checkpoint events

Why:

- expensive evaluation work must stay visible
- run behavior must be understandable from the terminal alone
- scaling decisions should never be hidden

## Rules

- Do not invent sample-count logic for the fixed defaults above.
- Do not treat `GRAD_CLIP` as a scale knob.
- Keep the canonical per-language overwrite policy explicit.
- When reusing checkpoints, follow the compatibility rules in `CHECKPOINTING.md`.
