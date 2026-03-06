# Models

This directory is the self-contained receipt-to-`schema.org/Order` workspace.

## Layout

- `data/`
  - `image_sources/` source manifests for image datasets
  - `pdf_sources/` source manifests for PDF datasets
  - `text_sources/` source manifests for text datasets and generators
  - `downloaded_images/` raw downloaded image assets
  - `downloaded_pdfs/` raw downloaded PDF assets
  - `downloaded_texts/` raw downloaded text assets and generated JSONL shards
- `samples/`
  - `inputs/` normalized `.txt` inputs grouped by language and source
  - `outputs/` hand-annotated JSON-LD targets
- `tokenizers/` language-specific corpora and SentencePiece artifacts
- `datasets/` tokenized training/validation JSONL
- `best_models/` PyTorch checkpoints
- `onnx_builds/` exported ONNX inference builds
- `quantized_models/` post-training quantized artifacts
- `scripts/` acquisition, extraction, corpus, tokenizer, training, and export scripts

## Input Extraction Goal

`extract:inputs` is the end-to-end ingestion command.

It is expected to:

1. download image, PDF, and text sources
2. extract all supported source material into `samples/inputs/{lang}/...`
3. write only `.txt` files as annotation inputs

The extraction pipeline is intentionally format-agnostic on the text side. CSV, JSON, YAML, XML, HTML, `.receipt`, Markdown, and other text-like files are accepted and flattened into `.txt`.

## Worker Usage

Heavy extraction scripts use worker-thread concurrency:

- `download-receipt-assets.mjs`
- `download-receipt-images.mjs`
- `download-receipt-texts.mjs`
- `extract-inputs-from-receipt-images.mjs`
- `extract-inputs-from-receipt-pdfs.mjs`
- `extract-inputs-from-receipt-texts.mjs`
- `extract-inputs.mjs`

## Main Commands

- `npm run extract:inputs`
- `npm run download:receipts`
- `npm run extract:receipt-images`
- `npm run extract:receipt-pdfs`
- `npm run extract:receipt-texts`

## Conventions

- Source manifests are language-first: `{lang}.json`
- Inputs are language-first: `samples/inputs/{lang}/{source}/...`
- Outputs are language-first: `samples/outputs/{lang}/{source}/...`
- Training and export artifacts are language-first
