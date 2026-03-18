# Model Architecture Guidelines

Use the same sample-count buckets across all guideline files:

- `small`: fewer than `1,000` training pairs
- `medium`: `1,000` to `50,000` training pairs
- `large`: more than `50,000` training pairs

These recommendations are for a random-init encoder-decoder Transformer trained from scratch on this repo's seq2seq task. They are intentionally conservative for small data.

## Recommended defaults

| Setting | Small | Medium | Large | Why |
| --- | ---: | ---: | ---: | --- |
| `D_MODEL` | `128` | `256` | `512` | Width should grow with data. Small data does not justify a wide model. |
| `NUM_HEADS` | `4` | `4` | `8` | Keep `D_MODEL % NUM_HEADS == 0` and avoid wasting capacity on too many heads too early. |
| `NUM_ENCODER_LAYERS` | `2` | `4` | `6` | Scale depth only after the model has enough data to use it. |
| `NUM_DECODER_LAYERS` | `2` | `4` | `6` | Keep encoder and decoder depth balanced for this task. |
| `FFN_DIM` | `512` | `1024` | `2048` | `4 x D_MODEL` remains the cleanest default ratio. |
| `DROPOUT` | `0.20` | `0.10` | `0.10` | Small data needs more regularization. Larger datasets usually do not. |
| `LABEL_PAD_ID` | `-100` | `-100` | `-100` | This matches PyTorch's default ignore index for cross-entropy loss. |
| `BOS_ID` | `1` | `1` | `1` | Keep the decoder start token fixed and explicit. |
| `EOS_ID` | `2` | `2` | `2` | Keep the decoder stop token fixed and explicit. |

## Notes

- The large-bucket defaults match the classic reference Transformer shape closely: `d_model=512`, `nhead=8`, `num_encoder_layers=6`, `num_decoder_layers=6`, `dim_feedforward=2048`, `dropout=0.1`.
- For this repo's current English dataset, the `small` bucket applies.
- If the model is underfitting, grow width before depth. If it is overfitting, raise dropout before growing anything.
- Do not increase `NUM_HEADS` unless `D_MODEL` grows with it. Tiny heads are usually wasted capacity on this task.
