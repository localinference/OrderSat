# Model Learning Practices

Training objective and model-selection objective are not the same thing.

For this project:

- token-level cross-entropy is the training objective
- validation exact match is the primary correctness metric
- validation loss is the secondary tie-breaker metric

## Why

- seq2seq models are still trained efficiently with teacher-forced token loss
- exact match is the correct whole-output metric for schema-aligned generation
- train exact match mainly measures memorization, not generalization

## Data-scale rule

Use `trainCount`, not total `sampleCount`.

Set:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`

## Effective-batch rule

Set:

- `BATCH_SCALE = sqrt(achieved_effective_batch / target_effective_batch)`

If accumulation keeps effective batch near target, `BATCH_SCALE` stays near
`1.0` and the optimizer does not need a large device-specific correction.

## Base optimization

Use these base values at `DATA_SCALE = 1.0` and `BATCH_SCALE = 1.0`:

- `LEARNING_RATE_BASE = 2e-4`
- `WEIGHT_DECAY_BASE = 1e-4`
- `GRAD_CLIP = 1.0`

## Derived optimization

Set:

- `LEARNING_RATE = clamp((LEARNING_RATE_BASE / sqrt(DATA_SCALE)) * BATCH_SCALE, 1e-4, 3e-4)`
- `WEIGHT_DECAY = clamp(WEIGHT_DECAY_BASE / (DATA_SCALE ** 2), 1e-4, 5e-4)`
- `GRAD_CLIP = 1.0`

## Evaluation and selection rules

- Rank checkpoints by validation exact match first.
- Use validation loss only as a tie-breaker or fallback when exact match was not run on that epoch.
- Treat train exact match as a memorization signal, not as the primary model-selection score.
- Keep structural-validity and schema-validity checks in the evaluation stack.

## Rules

- Use AdamW.
- If a scheduler is added later, treat the value above as the peak learning rate.
- Lower learning rate before touching `GRAD_CLIP`.
- Do not promote a checkpoint because of train exact match alone.
