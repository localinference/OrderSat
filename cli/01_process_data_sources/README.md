# Data Source Processing Walkthrough

This module prepares raw language data sources from `src/01_data_sources` into
deduplicated plain-text input samples under `src/02_training_samples/inputs`.

It is the first fan-in stage of the pipeline. Its job is to:

- discover source archives
- unpack them safely into a temp workspace
- OCR images
- render PDFs to images and OCR those pages
- pass through real text files directly
- deduplicate normalized text into stable `.txt` samples

## Entrypoint

Run:

```powershell
node cli/01_process_data_sources/index.js -L eng
```

The entrypoint is [index.js](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/index.js).

## High-Level Flow

The processor does this, in this order:

1. Parse CLI args through [getArgs](C:/Users/jorts/OrderSaT/cli/utils/getArgs/index.js).
2. Assert that `src`, `src/01_data_sources`, and each requested language source root exist.
3. Discover archive inputs per language with `fast-glob`.
4. Create one temporary unpack root under the OS temp directory.
5. Create the worker pool in [index.js](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/runWithWorker/index.js).
6. Queue archive unpack jobs to [unpackArchive](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/runWithWorker/Worker/jobs/unpackArchive/index.js).
7. Enumerate extracted files inside each unpacked archive.
8. Detect MIME types with `wasmagic`.
9. For images, queue OCR jobs to [getTextFromImage](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/runWithWorker/Worker/jobs/getTextFromImage/index.js).
10. For PDFs, queue render jobs to [pdfToImages](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/runWithWorker/Worker/jobs/pdfToImages/index.js), then OCR each rendered page.
11. For real text-like files, write them directly without OCR.
12. Normalize and deduplicate all extracted text in [writeUniqueToDest](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/writeUniqueToDest/index.js).
13. Close the worker pool and clean the temporary unpack directory.

## What It Reads

For `--language eng`, this module reads archive sources under:

- `src/01_data_sources/eng/images/**/*`
- `src/01_data_sources/eng/pdfs/**/*`
- `src/01_data_sources/eng/texts/**/*`

Only these archive extensions are discovered:

- `.nar`
- `.zip`
- `.tar`
- `.tgz`
- `.tar.gz`
- `.gz`

## What It Writes

For `--language eng`, this module writes deduplicated text samples under:

- `src/02_training_samples/inputs/eng/*.txt`

Each output filename is a stable SHA-384 digest of the cleaned text, encoded as
Base64URL.

That means:

- duplicate samples from different sources collapse to one file
- reruns stay stable
- downstream stages do not depend on original vendor filenames

## Worker Pool

The worker pool lives in [index.js](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/runWithWorker/index.js).

It:

- creates up to `os.availableParallelism()` workers
- queues archive, PDF, and OCR jobs
- dispatches work to idle workers
- replaces failed workers automatically
- shuts everything down explicitly at the end

The main thread still controls source discovery, MIME routing, direct text
writing, and deduplication.

## MIME Routing Policy

[index.js](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/index.js)
routes extracted files by detected MIME:

- `image/*`: OCR
- `application/pdf`: render to images, then OCR
- text-like MIME types such as `text/*`, JSON, and XML: write directly
- everything else: skip

That last rule matters. The stage should not write arbitrary binary
`application/*` payloads into the text corpus just because they are not PDFs.

## Text Normalization And Deduplication

[writeUniqueToDest](C:/Users/jorts/OrderSaT/cli/01_process_data_sources/writeUniqueToDest/index.js)
does the final cleanup step:

1. decode bytes to string
2. normalize with `NFKC`
3. clean text with `@sctg/sentencepiece-js`
4. skip empty cleaned output
5. hash the cleaned text
6. write with `flag: 'wx'` so duplicates are skipped, not overwritten

This stage is intentionally lossy in one specific way: original source filenames
and archive locations are discarded once the text content has been normalized
and deduplicated.

## Temp Workspace

Archives are unpacked under one temp root like:

- `%TEMP%/.data-unpack-XXXXXX/{language}/...`

The unpack path mirrors the relative source path, with the full archive suffix
removed. That keeps `.tar.gz` archives from leaving a misleading `.tar` suffix
in the temp folder name.

## Design Intent

This module is built around these rules:

1. Treat source archives as immutable inputs.
2. Extract as much real text as possible, but do not pretend binaries are text.
3. Normalize early so duplicates collapse before annotation.
4. Use OCR only where OCR is actually required.
5. Clean up temp state and worker threads explicitly.
