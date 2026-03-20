# Checkpointing and Reuse Practices

This project intentionally keeps one canonical model directory per
`{language, format}` in `05`.

That policy is correct, but checkpoint reuse still has to distinguish between
full resume, warm-start, and fresh start.

## Canonical files

Within `src/05_pytorch_models/{language}/{format}`:

- `best.pt` is the current best checkpoint for that tokenizer format
- `best_metrics.json` stores the best checkpoint metrics
- `history.json` stores the epoch history for the current canonical run
- `run.json` stores run metadata and runtime state

Save `history.json` and `run.json` every epoch. Overwrite `best.pt` only when a
new checkpoint actually wins.

## Best-checkpoint rule

Rank checkpoints by:

1. validation exact match
2. validation loss
3. earlier epoch as stable tie-breaker

Do not rank checkpoints by train exact match.

## Reuse modes

### Resume

Use `resume` only when:

- the previous run was interrupted
- tokenizer and vocab are unchanged
- architecture is unchanged
- dataset semantics are unchanged
- you want to continue the same optimization run

Load:

- model weights
- optimizer state
- history and runtime state

### Warm-start

Use `warm_start` when:

- the task is still the same
- tokenizer and vocab are unchanged
- architecture is unchanged
- you want to start a new run from the best known weights

Load:

- model weights only

Reset:

- optimizer state
- patience state
- run history

This is the default reuse mode for generalization-focused retraining.

### Fresh

Use `fresh` when any of these changed materially:

- vocab size
- tokenization format or semantics
- architecture shape
- label semantics

## Compatibility rules

Before loading `best.pt`, verify at least:

- vocab size
- pad or BOS or EOS conventions
- `d_model`
- attention heads
- encoder layers
- decoder layers
- FF dimension
- max source positions
- max target positions

## Relationship to `06`

`05` keeps one canonical best checkpoint per `{language, format}`.

`06` then compares the available formats for one language and exports only the
winning one as the canonical language-level ONNX bundle.

Do not collapse those two scopes together.

## Sources

This file is mostly project policy. The external rationale for validation-first
selection and deployment-specific export lives in:

- `MODEL_LEARNING.md`
- `ONNX_EXPORT.md`
