# Development Guidelines

This file is for defaults that should stay stable across runs and are not mainly driven by sample count.

## Recommended defaults

- `SEED = 7`
  Use one fixed seed while the pipeline is still changing so runs stay comparable.

- `LOG_FREQUENCY = 1`
  In this trainer logging is epoch-based, so logging every epoch is the correct default.

- `BOS_ID = 1`
  Keep one explicit decoder start token and do not change it between runs.

- `EOS_ID = 2`
  Keep one explicit decoder stop token and do not change it between runs.

- `LABEL_PAD_ID = -100`
  This matches the PyTorch cross-entropy ignore index convention and keeps padded labels out of the loss.

## Exact-match evaluation

`EXACT_MATCH_FREQUENCY` is one of the few monitoring settings that should scale with data size:

- if `sampleCount < 10_000`, use `EXACT_MATCH_FREQUENCY = 1`
- if `10_000 <= sampleCount <= 100_000`, use `EXACT_MATCH_FREQUENCY = 2`
- if `sampleCount > 100_000`, use `EXACT_MATCH_FREQUENCY = 3`

Use exact match in real training. Disable it only for smoke tests and very short debug runs.
