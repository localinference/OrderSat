# Sample Batching Guidelines

Batching is capability-bound and sequence-length-bound. It should not be derived from training-set size.

## Capability inputs

Read raw capability values from `get_device_capabilities()`.

Use:

- `capabilities.accelerator_memory_gb`
- `capabilities.system_memory_gb`
- `capabilities.cpu_count`
- `capabilities.supports_fp16`
- `capabilities.supports_bf16`
- `capabilities.pin_memory`
- `capabilities.recommended_num_workers`
- `capabilities.device_scale`

## Default target

Use:

- `TARGET_EFFECTIVE_BATCH = 16`

This is the optimization target. Physical batch and accumulation are just the mechanism used to hit it.

## Physical batch rule

Start with:

- `BATCH_SIZE = 1`

Then raise physical batch only after final sequence lengths are known and the step fits in memory.

## Accumulation rule

Set:

- `ACCUMULATION_STEPS = max(1, round(TARGET_EFFECTIVE_BATCH / BATCH_SIZE))`

This keeps the achieved effective batch near the target even on small devices.

## Environment rule

Use:

- `PIN_MEMORY = capabilities.pin_memory`
- `NUM_WORKERS = capabilities.recommended_num_workers`

Why:

- memory capacity limits physical batch
- CPU throughput limits how aggressively the input pipeline should parallelize
- environment capability should shape data loading and batch realization, not dataset-size heuristics

## Rules

- Set sequence lengths first, then tune batching.
- Do not invent sample-count scaling for batching.
- If accumulation is not wired into the trainer yet, keep physical batch small and stable.
