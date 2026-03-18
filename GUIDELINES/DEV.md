# Development Guidelines

Use the same sample-count buckets across all guideline files:

- `small`: fewer than `1,000` training pairs
- `medium`: `1,000` to `50,000` training pairs
- `large`: more than `50,000` training pairs

Here `sampleCount` means the number of training pairs in the training split, not raw source files and not tokenizer tokens.

## Recommended defaults

| Setting | Small | Medium | Large | Why |
| --- | ---: | ---: | ---: | --- |
| `SEED` | `7` | `7` | `7` | Use one fixed seed so runs stay comparable while the pipeline is still changing. |
| `LOG_EVERY` | `1` | `1` | `1` | In this repo it is an epoch interval, so logging every epoch is cheap and keeps regressions visible. |
| `EXACT_MATCH_EVERY` | `1` | `2` | `5` | Run exact match more often when every epoch is highly informative, and less often when decode cost grows with dataset size. |

## Notes

- In this trainer, `LOG_EVERY` is measured in epochs, not steps.
- In this trainer, `EXACT_MATCH_EVERY` is also measured in epochs, not steps.
- Keep exact match enabled for real training. Disable it only for smoke tests or very fast debugging loops.
- Change one debugging variable at a time. If you change the seed and the training settings together, the comparison is much less useful.
