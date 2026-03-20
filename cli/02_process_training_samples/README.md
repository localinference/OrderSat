# Training Sample Corpus Walkthrough

This module converts already-annotated training sample pairs in
`src/02_training_samples` into the shared language corpus used by `03`.

For each language, it writes one corpus file at:

- `src/03_tokenizers/{language}/corpus.jsonl`

## Entrypoint

Run:

```powershell
node cli/02_process_training_samples/index.js -L eng
```

The entrypoint is [index.js](C:/Users/jorts/OrderSaT/cli/02_process_training_samples/index.js).

## High-Level Flow

The corpus builder does this, in this order:

1. Parse CLI languages from [index.js](C:/Users/jorts/OrderSaT/cli/utils/getArgs/index.js).
2. Discover annotated output samples under `src/02_training_samples/outputs/{language}`.
3. Sort those output paths deterministically.
4. Derive each matching input path in [index.js](C:/Users/jorts/OrderSaT/cli/02_process_training_samples/getInputPathFromOutputPath/index.js).
5. Read the input `.txt` and output `.jsonld` in parallel.
6. Normalize input whitespace through [index.js](C:/Users/jorts/OrderSaT/cli/utils/cleanWhiteSpace/index.js).
7. Parse the JSON-LD output into JSON data.
8. Write one JSONL row shaped like `{ "input": "...", "output": { ... } }` into `src/03_tokenizers/{language}/corpus.jsonl`.

## What It Reads

For `--language eng`, this module reads:

- `src/02_training_samples/inputs/eng/*.txt`
- `src/02_training_samples/outputs/eng/*.jsonld`

Only samples that already have an output annotation are included, because the
module discovers rows from `outputs/{language}` first and then resolves the
matching input path from there.

## What It Writes

For `--language eng`, this module writes:

- `src/03_tokenizers/eng/corpus.jsonl`

Each line is a compact JSON object with:

- `input`: the normalized OCR/source text
- `output`: the parsed JSON value form of the annotated JSON-LD

## Why `output` Stays JSON Here

This corpus file is a shared storage format between `02`, `03`, and `04`.

So this module intentionally writes rows like:

```json
{ "input": "...", "output": { "@context": "https://schema.org", ... } }
```

This wrapper is still not the final model-side text stream.

`03` later expands this corpus into the actual tokenizer training text, and `04`
later turns the stored `output` JSON value into the exact stable-stringified
label text used by the model.

So `02` is not trying to produce the final tokenized representation. It is
producing the shared language corpus record that downstream steps consume,
without adding an extra quoted JSON-string layer.

## Normalization Policy

The current normalization policy is intentionally small:

- inputs have whitespace collapsed
- outputs are parsed into JSON data
- sample order is deterministic because discovered output paths are sorted

This keeps the corpus stable without changing the semantic content of the
annotations.

## Design Intent

This module is built around these rules:

1. Only include samples that have already been annotated.
2. Preserve one shared corpus source per language.
3. Keep the wrapper format simple and deterministic.
4. Normalize only enough to make downstream tokenizer and dataset generation
   stable.
