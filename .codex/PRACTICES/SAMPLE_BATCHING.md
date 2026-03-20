# Sample Batching Practices

Seq2seq training cost tracks tokens much more than raw sample count.

For this project, token-budgeted batching and length bucketing are the default
best practice.

## Per-sample cost model

Define:

- `SOURCE_TOKENS = input_length`
- `TARGET_TOKENS = label_length + 1`
- `SEQUENCE_TOKENS = SOURCE_TOKENS + TARGET_TOKENS`

Batch construction should keep total `SEQUENCE_TOKENS` per physical batch and
per optimizer step near a target budget.

## Implemented batching policy

The current trainer uses:

- length-aware token-budgeted physical batches
- gradient accumulation to reach the target effective batch
- lazy indexed datasets so the batcher can scale without loading full parsed
  corpora into memory

Current anchor values in `05`:

- `TARGET_EFFECTIVE_BATCH_SIZE = 16`
- `estimated_examples_per_batch` resolved from data length pressure and device
  capability
- `target_tokens_per_batch` and `target_tokens_per_optimizer_step` derived from
  measured average tokenized sequence length

## Why this is the current best practice

- padding waste drops when similarly sized samples are grouped together
- step time becomes more stable
- sequence length matters more than sample count for Transformer cost
- browser and local deployment goals force the project to care about token
  efficiency early

## Capability inputs

Read batching inputs from `get_device_capabilities()`:

- resolved device
- accelerator memory
- system memory
- CPU count
- pin-memory support
- recommended workers
- `device_scale`

These values decide what the machine can realize. They do not justify model
quality claims by themselves.

## Rules

- Set sequence lengths before tuning batching.
- Use token-budget batching before considering fixed sample-count batches.
- Keep lazy or indexed datasets once the corpus stops fitting comfortably in
  memory.
- Treat batching heuristics as implementation policy, not as universal theory.

## Sources

- Transformer architecture and sequence transduction cost motivation:
  https://arxiv.org/abs/1706.03762
- PyTorch DataLoader reference:
  https://docs.pytorch.org/docs/stable/data.html
