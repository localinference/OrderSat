# Seq2Seq Trainer Walkthrough

This module trains the project’s PyTorch encoder-decoder model from the tokenized JSONL datasets in `src/04_training_datasets/{language}` using the tokenizer vocabulary in `src/03_tokenizers/{language}`.

It is intentionally `FP32`-only in this phase. Mixed precision belongs in a later module.

## Entrypoint

Run:

```powershell
python cli/05_train_pytorch_model/__main__.py --language eng --device auto
```

The entrypoint is [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).

## High-Level Flow

The trainer does this, in this order:

1. Parse CLI args from [args/parse.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/args/parse.py).
2. Set reproducible seeds in [seed/set.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/seed/set.py).
3. Resolve all run paths in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
4. Read dataset stats from [stats/parse.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/stats/parse.py).
5. Detect environment and device capability in [device/capabilities.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/device/capabilities.py).
6. Resolve the concrete torch device in [device/build.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/device/build.py).
7. Read tokenizer vocab size from [vocab/read_size.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/vocab/read_size.py).
8. Load tokenized train and validation datasets from [TokenizedJsonlDataset/constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/TokenizedJsonlDataset/constructor.py).
9. Build the derived training config in [config/build.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/config/build.py).
10. Resolve effective sequence lengths in [sequence/get_effective_lenght.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/sequence/get_effective_lenght.py).
11. Build the collator from [Seq2SeqCollator/constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/Seq2SeqCollator/constructor.py).
12. Build train, train-eval, and validation `DataLoader`s in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
13. Build the model from [Seq2SeqTransformer/constructor.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/Seq2SeqTransformer/constructor.py).
14. Build the AdamW optimizer in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py).
15. Train epoch-by-epoch with [epoch/train.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/epoch/train.py).
16. Evaluate validation loss with [loss/evaluate.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/loss/evaluate.py).
17. Periodically run exact-match generation with [match/compute.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/match/compute.py) and [greedy/generate.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/greedy/generate.py).
18. Save best artifacts through [artifacts/save.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/artifacts/save.py).
19. Print a final run summary through [reporting/log.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/reporting/log.py).

## What It Reads

For `--language eng`, the trainer reads:

- tokenizer vocab from `src/03_tokenizers/eng/tokenizer.vocab`
- training split from `src/04_training_datasets/eng/train.jsonl`
- validation split from `src/04_training_datasets/eng/validation.jsonl`
- dataset stats from `src/04_training_datasets/eng/stats.json`

Those paths are resolved in [**main**.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/__main__.py) by `build_run_paths`.

## Why Each Step Exists

### Args and Seed

`parse_args()` decides which language directory and device policy to use.

`set_seed()` exists so runs are repeatable enough to compare changes. It seeds Python, NumPy, and PyTorch, and also disables CuDNN nondeterministic shortcuts.

### Stats and Capabilities

`parse_stats()` reads the measured dataset facts, especially:

- `trainCount`
- `validationCount`
- tokenized input length stats
- tokenized label length stats

This is important because the trainer should not guess lengths or scale from hand-written constants.

`get_device_capabilities()` reads what the machine can realistically support:

- resolved device family
- accelerator memory
- system memory
- CPU count
- a bounded `device_scale`

This is important because model capacity and physical batching should respect the current machine, not only the dataset.

### Config Resolution

`build_training_config()` combines:

- data-driven scale from `trainCount`
- machine-driven scale from `device_scale`

and derives:

- model width and depth
- dropout
- physical batch size
- accumulation steps
- effective batch size
- learning rate
- weight decay
- epoch count
- early stopping settings
- exact-match frequency

This is why the trainer adapts without using crude hard-coded size buckets in the runtime path.

### Dataset Loading

`TokenizedJsonlDataset` loads JSONL records and validates that each record has:

- `sample_id`
- `input_ids`
- `labels`

It also validates token id bounds against the tokenizer vocab size. That is there to fail early if the dataset and tokenizer are out of sync.

### Sequence Length Resolution

`get_effective_sequence_lengths()` computes the actually observed max lengths in the current train and validation splits and checks that they do not exceed the measured stats maxima.

This exists to prevent silent truncation drift and to ensure the current dataset still matches the stats file.

### Collation

The collator:

- truncates to the chosen max lengths
- pads encoder inputs with `pad_id`
- builds `attention_mask`
- prepends `BOS` to decoder inputs
- appends `EOS` to decoder targets
- pads labels with `LABEL_PAD_ID`

This is why the model gets teacher-forced decoder inputs and masked labels in the correct shape.

### Model

`Seq2SeqTransformer` is the actual encoder-decoder network. It builds:

- one shared token embedding
- learned source position embeddings
- learned target position embeddings
- a Transformer encoder
- a Transformer decoder
- an output projection to vocabulary logits

The model is parameterized by the resolved training config, not by a fixed static architecture.

### Training

`train_epoch()`:

- loops over the train loader
- computes token-normalized cross-entropy
- divides loss by accumulation steps
- backpropagates
- clips gradients
- steps the optimizer only when accumulation says so

This is why the trainer can keep a stable effective batch even when physical batch size must stay small.

### Validation

`evaluate_loss()` computes validation loss without gradients.

`compute_exact_match()` performs greedy generation and compares the generated token sequence against the full target sequence.

Loss tells you whether the model is fitting the distribution.
Exact match tells you whether the model is producing fully correct structured outputs.

### Saving

When validation loss improves, `save_artifacts()` writes:

- `best_metrics.json`
- `history.json`
- `run.json`
- `best.pt`

into `src/05_pytorch_models/{language}`.

This is why every best checkpoint carries not just weights but also the resolved run context.

## Terminal Logging

Logging is produced by [reporting/log.py](C:/Users/jorts/OrderSaT/cli/05_train_pytorch_model/reporting/log.py).

You now get stage-level messages such as:

- `args.parsed`
- `seed.set.start` / `seed.set.complete`
- `stats.parse.start` / `stats.parse.complete`
- `device.capabilities.start` / `device.capabilities.complete`
- `config.build.start` / `config.build.complete`
- `dataset.load.start` / `dataset.load.complete`
- `loader.build.start` / `loader.build.complete`
- `model.build.start` / `model.build.complete`
- `optimizer.build.start` / `optimizer.build.complete`
- `train.epoch.start` / `train.epoch.complete`
- `validation.loss.start` / `validation.loss.complete`
- `validation.exact_match.start` / `validation.exact_match.complete`
- `artifacts.save.start` / `artifacts.save.complete`
- `checkpoint.saved`
- `epoch.complete`
- `training.stop`
- `training.complete`

The logs intentionally include:

- language
- file paths
- resolved device
- parameter counts
- batch counts
- batch size and accumulation
- epoch timing
- evaluation timing
- exact-match timing
- total run time

So from the terminal you can tell both what the trainer is doing and how long each stage took.

## Outputs

Best-run artifacts are written to:

- `src/05_pytorch_models/{language}/best.pt`
- `src/05_pytorch_models/{language}/best_metrics.json`
- `src/05_pytorch_models/{language}/history.json`
- `src/05_pytorch_models/{language}/run.json`

`run.json` is the most important context file for later debugging because it includes the resolved config, device capabilities, dataset stats, and sequence lengths that produced the checkpoint.

## Design Intent

The trainer is built around four rules:

1. Read measured facts from the dataset instead of guessing.
2. Respect current machine capability instead of pretending every environment is the same.
3. Keep the runtime path semantically explicit enough that failures are easy to localize.
4. Emit enough terminal-visible state that a training run can be understood while it is happening, not only after it finishes.
