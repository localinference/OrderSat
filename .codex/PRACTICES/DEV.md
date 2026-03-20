# Development Practices

This directory records project policy, not universal law.

If code and practice docs diverge, fix the divergence. Do not keep aspirational
rules that the pipeline no longer follows.

## Fixed invariants

These values are intentional fixed defaults in `05`, not scaling knobs:

- `SEED = 7`
- `LOG_FREQUENCY = 1`
- `BOS_ID = 1`
- `EOS_ID = 2`
- `LABEL_PAD_ID = -100`
- `GRAD_CLIP = 1.0`

Do not tie them to sample count, device class, or convenience heuristics.

## Canonical pipeline shape

The current pipeline is:

1. `01` prepares raw input text samples.
2. `02` builds a shared `{ input, output }` corpus where `output` is parsed JSON,
   not a quoted JSON string.
3. `03` trains both `unigram` and `bpe` tokenizers from the same prepared
   corpus.
4. `04` builds per-format tokenized datasets and deterministic train or
   validation splits.
5. `05` trains one PyTorch model per `{language, format}`.
6. `06` selects the best format for one language and exports one canonical FP32
   ONNX bundle.
7. `07` and `08` derive deployment-specific bundles from `06`.
8. `09` packages the runtime artifacts into TypeScript.

The docs in this directory should explain that pipeline as it actually exists.

## Artifact policy

Use these canonical output shapes:

- `src/03_tokenizers/{language}/{format}`
- `src/04_training_datasets/{language}/{format}`
- `src/05_pytorch_models/{language}/{format}`
- `src/06_fp32_export_onnx_models/{language}`
- `src/07_mixed-fp16_gpu_onnx_models/{language}`
- `src/08_uint8_cpu_onnx_models/{language}`

Why:

- tokenizers and training datasets are format-specific
- PyTorch checkpoints must stay format-specific
- ONNX export and deployment bundles should be language-specific canonical
  selections, not duplicated per tokenizer after `06` has already chosen the
  winner

## Precision boundaries

Keep phase boundaries explicit:

- `05` trains and saves in `FP32`
- `06` exports one canonical `FP32` ONNX model
- `07` derives a WebGPU `mixed-fp16` variant from `06`
- `08` derives a WASM `uint8` variant from `06`

Do not mix deployment compression concerns back into `05`.

## Logging and observability

All substantive stages should log:

- resolved paths
- device capabilities
- adjusted options
- stage timings
- validation cadence decisions
- checkpoint decisions
- final audit results

Why:

- expensive local training has to be observable from the terminal
- adaptive heuristics must be inspectable
- run behavior should be reconstructible without opening code first

## Rules

- Keep project-specific heuristics explicit and bounded.
- Separate fixed invariants from scalable heuristics.
- Prefer deterministic outputs over hidden randomness in artifact generation.
- Keep one canonical best artifact per intended scope:
  `05` per `{language, format}`, `06` and later per `{language}`.
- Update this directory when pipeline behavior changes materially.

## Sources

This file is mostly project policy. See the more specific practice files for the
external sources behind tokenizer, model, export, and deployment rules.
