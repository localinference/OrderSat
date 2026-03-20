# Tokenized Dataset Walkthrough

This module converts the shared corpus in `src/03_tokenizers/{language}/corpus.jsonl`
into tokenized train/validation datasets for every tokenizer format discovered
under `src/03_tokenizers/{language}/{format}`.

It now produces one dataset bundle per format under:

- `src/04_training_datasets/{language}/bpe/*`
- `src/04_training_datasets/{language}/unigram/*`

## Entrypoint

Run:

```powershell
node cli/04_generate_tokenized_datasets/index.js -L eng
```

The entrypoint is [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/index.js).

## High-Level Flow

The dataset generator does this, in this order:

1. Resolve the shared corpus source under `src/03_tokenizers/{language}/corpus.jsonl`.
2. Discover tokenizer formats and `tokenizer.model` paths in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/readTokenizerFormats/index.js).
3. Load corpus lines in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/readCorpusLines/index.js).
4. Parse each JSONL row in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/parseCorpusLine/index.js).
5. For each discovered format, repeat the tokenizer-specific steps below.
6. Load that format’s tokenizer model in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/loadTokenizer/index.js).
7. Tokenize input/output strings into sample records in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/tokenizeSample/index.js).
8. Resolve the validation split policy from corpus size in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/adjustValidationRatio/index.js).
9. Split into train/validation sets in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/splitSamples/index.js).
10. Summarize input and label length stats in [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/summarizeLengths/index.js).
11. Write `train.jsonl`, `validation.jsonl`, and `stats.json` through [index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/writeJsonl/index.js).

## What It Reads

For `--language eng`, the dataset generator reads:

- `src/03_tokenizers/eng/corpus.jsonl`
- `src/03_tokenizers/eng/bpe/tokenizer.model`
- `src/03_tokenizers/eng/unigram/tokenizer.model`

## What It Writes

For `--language eng`, the dataset generator writes:

- `src/04_training_datasets/eng/bpe/train.jsonl`
- `src/04_training_datasets/eng/bpe/validation.jsonl`
- `src/04_training_datasets/eng/bpe/stats.json`
- `src/04_training_datasets/eng/unigram/train.jsonl`
- `src/04_training_datasets/eng/unigram/validation.jsonl`
- `src/04_training_datasets/eng/unigram/stats.json`

Each `stats.json` records:

- `language`
- `format`
- `corpusPath`
- `modelPath`
- `sampleCount`
- `trainCount`
- `validationCount`
- `validationRatio`
- `validationRange`
- tokenized input length stats
- tokenized label length stats

## Validation Split Policy

[adjustValidationRatio/index.js](C:/Users/jorts/OrderSaT/cli/04_generate_tokenized_datasets/adjustValidationRatio/index.js)
implements a count-driven holdout policy.

It does not use one fixed validation ratio for all corpus sizes.

Instead it:

- uses four size ranges
- grows validation count continuously within each range
- reduces validation ratio as the corpus gets larger
- records the chosen ratio and range in `stats.json`

That keeps small datasets from having a uselessly tiny holdout, while also
stopping large datasets from wasting too much data on validation.

## Why The Structure Changed

`03` now emits one tokenizer per format, so `04` must also stay format-aware.

Without that separation:

- `bpe` and `unigram` token IDs would get mixed
- token-length stats would become meaningless
- downstream `05` training could not compare models fairly

The current structure makes each format fully traceable from tokenizer through
dataset to model.

## Design Intent

This module is built around these rules:

1. Keep one shared corpus source per language.
2. Generate one tokenized dataset bundle per tokenizer format.
3. Record measured token-length facts per format.
4. Make validation sizing depend on corpus size, not a fake constant ratio.
