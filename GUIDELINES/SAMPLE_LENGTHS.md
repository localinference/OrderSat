# Sample Length Guidelines

Sequence-length defaults should come from the tokenized dataset statistics produced during dataset creation.

## Default rule

Use `max`, not `p95`.

Set:

- `MAX_INPUT_LENGTH = round_up_to_multiple(input_length_max, 32)`
- `MAX_LABEL_LENGTH = round_up_to_multiple(label_length_max, 32)`

This is the default guideline for this project.

## Why

- This project is correctness-first, not throughput-first.
- Truncation is not a harmless optimization when the model is expected to reproduce full structured outputs correctly.
- Rare long samples can still be important cases, not disposable outliers.
- Using `max` keeps the training target faithful to the dataset instead of silently dropping the tail.

## Rules

- Measure lengths after tokenization, not from character counts.
- Derive both limits from the dataset stats written at dataset creation time.
- Round upward to a multiple of `32`.
- Recompute the stats whenever the dataset changes materially.
- Do not hardcode guessed lengths when real stats already exist.
- Do not use sample-count buckets here. Lengths should come from measured distributions, not from dataset size categories.

## Exception

Use a smaller percentile-based limit only if faster training is explicitly more important than preserving full sequence fidelity.
