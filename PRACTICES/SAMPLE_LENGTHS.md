# Sample Length Practices

Training lengths and evaluation decode limits are related, but they are not the
same rule.

## Training-length rule

Use measured tokenized `max`, not `p95`.

Set:

- `MAX_INPUT_LENGTH = round_up_to_multiple(inputLengths.max, 32)`
- `MAX_LABEL_LENGTH = round_up_to_multiple(labelLengths.max, 32)`

These values must be derived from the dataset stats file written during dataset
creation.

## Why

- this project is correctness-first
- truncation can destroy structured-output fidelity
- rare long samples may still matter

## Exact-match evaluation rule

Do not force every monitored sample to decode all the way to the dataset-wide
maximum label length.

When the gold target is known during evaluation:

- use target-aware stopping
- stop once exact match is impossible
- stop once the generation has exceeded the target length plus a small EOS margin

Why:

- this preserves exact-match correctness as the goal
- it removes unnecessary autoregressive decode work

## Rules

- Measure lengths after tokenization, not from characters.
- Read training lengths from dataset stats, not from guessed defaults.
- Recompute them whenever the dataset changes materially.
- Round upward to a multiple of `32`.
- Keep training caps and evaluation stop rules conceptually separate.
