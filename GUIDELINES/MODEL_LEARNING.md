# Model Learning Guidelines

Use the same sample-count buckets across all guideline files:

- `small`: fewer than `1,000` training pairs
- `medium`: `1,000` to `50,000` training pairs
- `large`: more than `50,000` training pairs

These defaults assume the current trainer shape in this repo: AdamW, no warmup scheduler, no LR decay scheduler, and clipped gradients.

## Recommended defaults

| Setting | Small | Medium | Large | Why |
| --- | ---: | ---: | ---: | --- |
| `LEARNING_RATE` | `3e-4` | `2e-4` | `1e-4` | Without warmup or decay, the safest default is to lower peak LR as model size and data size grow. |
| `WEIGHT_DECAY` | `5e-4` | `1e-4` | `1e-4` | Small data needs stronger regularization. Once data grows, keep decay mild rather than forcing underfit. |
| `GRAD_CLIP` | `1.0` | `1.0` | `1.0` | `1.0` is still the most reliable default for small Transformers with long decoder targets. |

## Notes

- Keep using AdamW, not Adam with L2 treated as fake weight decay. Decoupled weight decay is the right default for adaptive optimizers.
- If training becomes unstable, lower `LEARNING_RATE` before touching `GRAD_CLIP`.
- If the model clearly memorizes training data but validation stalls, raise regularization before increasing model size.
- If you later add LR warmup and decay, treat the values above as peak learning rates.
