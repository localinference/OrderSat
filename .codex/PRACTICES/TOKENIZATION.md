# Tokenization Practices

Tokenizer quality should be judged by downstream model quality, not by
tokenizer aesthetics alone.

## Corpus semantics

Train tokenizers on the actual text streams the model consumes.

Current implemented behavior:

- `02` writes corpus rows shaped like `{ input, output }`
- `output` is stored as parsed JSON data, not as a quoted JSON string
- `03` prepares one temporary text corpus containing:
  - raw `input` text
  - stable-stringified `output` JSON text

Do not train the tokenizer on storage wrapper noise like outer JSONL keys or on
an extra quoted JSON-string layer.

## Tokenizer formats

Current project policy:

- always train both `unigram` and `bpe`
- train them from the same prepared corpus
- write them to `src/03_tokenizers/{language}/{format}`

Why:

- this project has both noisy OCR input and rigid structured output
- neither `unigram` nor `bpe` should be assumed best in advance

## Vocab budget rule

Treat `max_vocab_size` as a deployment budget, not as a guarantee that the
realized vocabulary will use the whole budget.

Current project default:

- `max_vocab_size = 8192`
- `hard_vocab_limit = False`

Why:

- the browser/local target cares about vocab-driven embedding and logits cost
- SentencePiece can realize fewer useful pieces on small corpora
- the ceiling should leave headroom for corpus growth without forcing a large
  realized vocab on day one

## Selection rule

Choose tokenizer format by downstream results, not by shorter sequences alone.

Project ranking order:

1. validation exact match
2. character-normalized quality comparison when exact match ties
3. deployment/runtime cost

That is why `06` compares format candidates using validation exact match first
and only then uses secondary metrics like bits per output character and runtime
proxy.

## Runtime rule

For one exported language model:

- keep one `tokenizer.model`
- copy that same tokenizer unchanged into `06`, `07`, and `08`

Do not build different tokenizers per deployment target. If tokenization needs
improvement, retrain it upstream in `03` and rerun the pipeline.

## Sources

- SentencePiece project:
  https://github.com/google/sentencepiece
- SentencePiece paper:
  https://arxiv.org/abs/1808.06226
- Subword regularization:
  https://arxiv.org/abs/1804.10959
