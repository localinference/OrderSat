# Sample Length Practices

Training caps, exported positional limits, and evaluation stop rules are
related, but they are not the same thing.

## Training-length rule

Use measured tokenized maxima from the dataset that will actually be trained.

Current implemented behavior:

- `04` writes tokenized length stats per `{language, format}`
- `05` recomputes the observed max input and label lengths across the concrete
  train and validation files
- `05` refuses to train if the observed maxima exceed the recorded stats

Then:

- `max_source_positions = observed_max_input_length`
- `max_target_positions = observed_max_label_length + 1`

No extra percentile trimming and no arbitrary rounding are applied in the
current trainer.

## Why this is the current best practice

- this project is correctness-first
- truncation can destroy structured-output fidelity
- long outliers may still matter
- exported ONNX model limits should reflect the real trained positional capacity

## Evaluation decode rule

Exact-match evaluation should be target-aware, not dataset-max-aware.

Current implemented behavior:

- decode only up to the target length for the current batch
- stop a sample once exact match is already impossible
- stop a sample once the gold target has reached `EOS`

This preserves exact-match correctness while removing wasted autoregressive
decode steps.

## Runtime rule

Do not confuse local execution with infinite positional capacity.

If the exported model says:

- `maxSourcePositions = N`
- `maxTargetPositions = M`

those are architecture contracts, not optional client-side hints.

## Rules

- Measure lengths after tokenization, not from character counts.
- Derive training caps from dataset stats and observed tokenized files.
- Recompute them when the dataset or tokenizer changes materially.
- Keep training caps and evaluation stopping logic conceptually separate.

## Sources

- Transformer architecture reference:
  https://arxiv.org/abs/1706.03762
