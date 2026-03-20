# Model Learning Practices

Training objective and model-selection objective are not the same thing.

For this project:

- teacher-forced token-level cross-entropy is the training objective
- validation exact match is the primary correctness metric
- validation loss is the secondary tie-breaker metric

## Optimization scaling

The current code in `05` uses:

- `DATA_SCALE = clamp((trainCount / 10_000) ** 0.25, 0.5, 2.0)`
- `BATCH_SCALE = sqrt(achieved_effective_batch_size / target_effective_batch_size)`

Then:

- `LEARNING_RATE = clamp((2e-4 / sqrt(DATA_SCALE)) * BATCH_SCALE, 1e-4, 3e-4)`
- `WEIGHT_DECAY = clamp(1e-4 / (DATA_SCALE ** 2), 1e-4, 5e-4)`
- `GRAD_CLIP = 1.0`

Optimizer:

- `AdamW`

## Why this is the current best practice

- Token-level loss is still the efficient way to train encoder-decoder models.
- Exact match is the real task metric for schema-aligned generation.
- Train exact match mainly measures memorization, not generalization.
- Decoupled weight decay is the correct default interpretation of weight decay
  with Adam-style optimizers.

## Checkpoint and stopping policy

The current best practice is:

- rank checkpoints by validation exact match first
- use validation loss as tie-breaker
- count patience on exact-match evaluation windows when exact match is part of
  the schedule
- fall back to loss-based patience only when exact match is not being run
- run full train exact match only as a final audit

This keeps checkpoint selection and early stopping aligned with the same
generalization target.

## Cross-tokenizer comparison rule

Raw token loss is not directly comparable across different tokenizers.

Why:

- `bpe` and `unigram` break the same output text into different token counts
- lower loss per token does not automatically mean better sequence modeling

So:

- compare tokenizer candidates first by validation exact match
- use validation bits per output character as the secondary quality comparison
  across tokenizers

That is why `06` does not rank export candidates by raw token loss alone.

## Rules

- Do not promote a checkpoint because of train exact match alone.
- Lower learning rate before touching `GRAD_CLIP`.
- Treat any future scheduler as shaping the base learning rate above, not
  replacing the checkpoint metric hierarchy.
- Keep structure-validity checks in the evaluation stack whenever available.

## Sources

- Transformer seq2seq training reference:
  https://arxiv.org/abs/1706.03762
- AdamW:
  https://openreview.net/forum?id=Bkg6RiCqY7
- Compute-optimal scaling motivation:
  https://arxiv.org/abs/2203.15556
