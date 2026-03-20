# Validation Split Practices

The canonical pipeline should use a deterministic holdout split whose size
depends on corpus scale.

For very small experiments, cross-validation can still be useful offline, but
the reproducible artifact pipeline should keep one deterministic split.

## Implemented validation-count policy

`04` currently resolves validation size from `sampleCount` with four ranges and
continuous interpolation:

- `extra_small` up to `200`
  - `max(25, sampleCount * 0.25)`
- `small` up to `2_000`
  - linear interpolation from `50` to `200`
- `medium` up to `20_000`
  - linear interpolation from `200` to `1_000`
- `large` above `20_000`
  - log interpolation from `1_000` to `5_000` by `1_000_000`

This gives unit-level variation inside each range while still changing policy by
scale regime.

## Split rule

Current implemented behavior:

- sort samples by `sample_id`
- take the first `validationCount` samples as validation
- take the rest as train

Why:

- the split is deterministic
- reruns stay reproducible
- tokenizer formats can be compared on the same underlying validation examples

## Why this is the current best practice

- a fixed validation ratio wastes too much training data at scale
- a fixed validation count is too small for large corpora
- tiny corpora still need enough held-out examples for exact-match evaluation to
  be meaningful
- deterministic artifact generation matters for pipeline reproducibility

## Cross-validation rule

If you are comparing modeling ideas on very small data and want stronger
statistical confidence, use repeated splits or cross-validation outside the
canonical artifact pipeline.

Do not replace the canonical deterministic holdout inside `04` with ad hoc
randomness.

## Sources

- scikit-learn cross-validation guide:
  https://sklearn.org/stable/modules/cross_validation.html
