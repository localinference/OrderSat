# Sample Length Guidelines

Sequence lengths must come from measured tokenized dataset statistics, not from training-set size and not from guessed defaults.

## Default rule

Use tokenized `max`, not `p95`.

Set:

- `MAX_INPUT_LENGTH = round_up_to_multiple(inputLengths.max, 32)`
- `MAX_LABEL_LENGTH = round_up_to_multiple(labelLengths.max, 32)`

These values must be derived from the dataset stats file written during dataset creation.

## Why

- This project is correctness-first.
- Truncation is not a harmless optimization when the goal is full structured-output fidelity.
- Rare long samples may still be important cases.
- Measured max length is more faithful than a percentile cut.

## Rules

- Measure lengths after tokenization, not from character counts.
- Read them from dataset stats, not from training-size regimes.
- Recompute them whenever the dataset changes materially.
- Round upward to a multiple of `32`.
- Do not hardcode guessed lengths when real stats already exist.
