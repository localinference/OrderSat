## Purpose

`synt` generates synthetic `{input -> schema.org/Order JSON-LD}` pairs directly into:

- `src/02_training_samples/inputs/{language}/*.txt`
- `src/02_training_samples/outputs/{language}/*.jsonld`

It exists to widen semantic and presentation coverage for `Order`-like documents. It is complementary to real OCR-derived samples, not a replacement for them.

## Flow

1. Load `synt/config/{language}/labels` and `synt/config/{language}/values`.
2. Build a semantic order record first.
3. Render that same record into one arbitrary order-like input presentation.
4. Build the matching JSON-LD from the semantic record.
5. Hash the `{input + stable output}` pair with `SHA-384` base64url.
6. Write the `.txt` and `.jsonld` pair only when that hash does not already exist.

The generator is coverage-driven, not blind-random. It cycles:

- blueprint
- renderer
- OCR noise profile
- error profile

and randomizes values inside each coverage cell. That gives deliberate coverage without pretending that every literal semantic combination is tractable.

## Blueprints

- `retail-receipt`
- `online-confirmation`
- `shipping-notice`
- `service-confirmation`

## Renderers

- `plain-receipt`
- `email-summary`
- `html-order`
- `json-dump`
- `xml-summary`
- `csv-export`

The same underlying order can therefore surface as line-oriented receipt text, structured machine text, or semi-structured confirmation text.

## Error Behavior

`synt` supports the same two output markers used by the real annotation pipeline:

- `[ERROR:UNCERTAIN]`
- `[ERROR:FATAL]`

Targeted fields are corrupted in the input first, then the JSON-LD keeps the raw corrupted value in the native field with the correct prefix. Global OCR noise is applied separately to teach normalization behavior.

## Performance Model

`synt` uses a worker pool like `cli/01_process_data_sources`, but keeps generation deterministic:

- each worker receives an index range
- the index maps to a stable coverage plan
- the seed and index fully determine the generated sample

That makes large runs parallel without giving up reproducibility.

## CLI

Example:

```powershell
node synt -L eng -C 5000 --batchSize 64 --concurrency 8 --seed 1
```

Options:

- `-L, --languages`
- `-C, --count`
- `--batchSize`
- `--concurrency`
- `--seed`
- `--validateMode all|sample|none`
- `--maxAttemptsFactor`

## Validation

`synt` can validate generated JSON-LD with the same validator used elsewhere:

- `all`: validate every written sample
- `sample`: validate the first sample seen for each coverage cell
- `none`: skip structural validation

`sample` is the default because the output side uses a small set of stable JSON-LD shapes while the input side carries most of the variation.

## Extension Model

The intended growth path is:

1. add richer value banks per language
2. add more semantic blueprints
3. add more renderers
4. add stronger OCR-noise and corruption profiles
5. add new field generators only when they improve downstream generalization

The important rule is semantic-first synthesis. The output JSON-LD should come from the semantic record, not be reverse-parsed from the rendered input.
