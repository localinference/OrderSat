# Sample Batching Practices

Batching cost tracks tokens much more than raw sample count.

For seq2seq training, prefer token-budgeted batching and length bucketing over
fixed sample-count batching.

## Cost rule

For each sample, define:

- `SOURCE_TOKENS = input_length`
- `TARGET_TOKENS = label_length + 1`
- `SEQUENCE_TOKENS = SOURCE_TOKENS + TARGET_TOKENS`

Batch construction should keep total `SEQUENCE_TOKENS` per optimizer step near a
target budget.

## Default strategy

Prefer:

- length bucketing
- token-budgeted physical batches
- gradient accumulation to reach the target effective batch

Why:

- padding waste drops when lengths are grouped
- step time becomes more stable
- GPU and CPU utilization both improve

## Capability inputs

Read raw capability values from `get_device_capabilities()`.

Use:

- `capabilities.accelerator_memory_gb`
- `capabilities.system_memory_gb`
- `capabilities.cpu_count`
- `capabilities.pin_memory`
- `capabilities.recommended_num_workers`
- `capabilities.device_scale`

## Fallback rule

If token-budgeted batching is not implemented yet:

- keep physical batch conservative
- use accumulation for the effective batch target
- still bucket by length if possible

## Large-data rule

Once the dataset no longer fits comfortably in memory:

- stop loading full JSONL files eagerly
- use indexed or sharded lazy loading
- keep length metadata available for the batcher

## Rules

- Set sequence lengths before tuning batching.
- Do not derive batching from sample-count buckets.
- Use device capability to realize the batch, not to justify model quality claims.
