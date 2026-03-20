# Tokenizer Training Walkthrough

This module trains the project’s SentencePiece tokenizers from the shared
language corpus in `src/03_tokenizers/{language}/corpus.jsonl`.

It now produces one tokenizer per format under:

- `src/03_tokenizers/{language}/unigram/*`
- `src/03_tokenizers/{language}/bpe/*`

## Entrypoint

Run:

```powershell
python cli/03_train_tokenizer_vocabs/__main__.py --language eng
```

The entrypoint is [**main**.py](C:/Users/jorts/OrderSaT/cli/03_train_tokenizer_vocabs/__main__.py).

## High-Level Flow

The tokenizer trainer does this, in this order:

1. Parse CLI args from [parse.py](C:/Users/jorts/OrderSaT/cli/03_train_tokenizer_vocabs/args/parse.py).
2. Read the shared corpus source from `src/03_tokenizers/{language}/corpus.jsonl`.
3. Convert each JSONL sample into the actual model-side strings in [prepare.py](C:/Users/jorts/OrderSaT/cli/03_train_tokenizer_vocabs/corpus/prepare.py).
4. Build SentencePiece kwargs in [build.py](C:/Users/jorts/OrderSaT/cli/03_train_tokenizer_vocabs/kwargs/build.py).
5. Train `unigram`.
6. Train `bpe`.
7. Delete the temporary prepared text corpus.

## What It Reads

For `--language eng`, the tokenizer trainer reads:

- `src/03_tokenizers/eng/corpus.jsonl`

That corpus is still the shared language-level source of paired input/output
samples.

## What It Writes

For `--language eng`, the tokenizer trainer writes:

- `src/03_tokenizers/eng/unigram/tokenizer.model`
- `src/03_tokenizers/eng/unigram/tokenizer.vocab`
- `src/03_tokenizers/eng/bpe/tokenizer.model`
- `src/03_tokenizers/eng/bpe/tokenizer.vocab`

It does not permanently keep the temporary prepared training text file.

## Why The Prepared Corpus Exists

The raw `corpus.jsonl` rows are storage records shaped like:

```json
{ "input": "...", "output": "..." }
```

That wrapper is not the actual text stream the model later tokenizes.

So [prepare.py](C:/Users/jorts/OrderSaT/cli/03_train_tokenizer_vocabs/corpus/prepare.py)
extracts:

- the real `input` text
- the exact `stableStringify(output)` representation the downstream dataset
  generator uses

This makes SentencePiece learn the same kinds of strings the model actually
consumes, instead of learning wrapper keys like `"input"` and `"output"`.

## Training Policy

The current policy is:

- train both `unigram` and `bpe`
- use the same prepared corpus for both
- keep `hard_vocab_limit=False`
- use `8192` as the default max vocab-size ceiling

That means SentencePiece can realize a smaller effective vocabulary when the
current corpus does not justify the full ceiling.

## Why It Produces Both Formats

The tokenizer choice is now treated as a real pipeline decision, not a fixed
assumption.

So `03` always emits both candidates and lets later stages:

- build per-format tokenized datasets
- train per-format models
- compare downstream results before selecting a winner

## Design Intent

This module is built around these rules:

1. Train tokenizers on the text the model actually sees.
2. Keep tokenizer formats isolated under their own directories.
3. Treat `unigram` and `bpe` as candidates, not truths.
4. Keep one shared language corpus source and fan out from there.
