# Sample Batching Guidelines

Batching is driven mainly by memory and sequence lengths, not by sample count.

## Physical batch size

Use the largest `BATCH_SIZE` that fits reliably after sequence lengths are finalized.

For this repo's current seq2seq setup, the safe default is:

- `BATCH_SIZE = 1`

Why: decoder-side memory grows quickly when labels are long, and long labels are normal in this task.

## Gradient accumulation

Use gradient accumulation to reach a stable effective batch instead of forcing a larger physical batch.

Best default:

- `ACCUMULATION_STEPS = 16`

This gives:

- effective batch = `BATCH_SIZE * ACCUMULATION_STEPS`

## Practical rule

Use this pattern:

- if `BATCH_SIZE = 1`, use `ACCUMULATION_STEPS = 16`
- if `BATCH_SIZE = 2`, use `ACCUMULATION_STEPS = 8`
- if `BATCH_SIZE = 4`, use `ACCUMULATION_STEPS = 4`
- if `BATCH_SIZE = 8`, use `ACCUMULATION_STEPS = 2`

The target is an effective batch close to `16`.

## Rules

- Set sequence lengths first, then tune batch size.
- Do not pretend batching is data-size driven when the real limit is VRAM.
- If accumulation is not implemented in the trainer yet, keep the physical batch small and do not write fake recommendations around it.
