#!/usr/bin/env python3
import pathlib

import sentencepiece

from args.parse import parse_args
from corpus.prepare import prepare_corpus
from kwargs.build import build_kwargs

TOKENIZERS_DIR = pathlib.Path("src/03_tokenizers")
CORPUS_NAME = "corpus.jsonl"
MODEL_PREFIX = "tokenizer"
TOKENIZER_FORMATS = ("unigram", "bpe")


def main() -> None:
    args = parse_args()
    language_dir = TOKENIZERS_DIR / args.language
    source_corpus_path = language_dir / CORPUS_NAME

    prepared_corpus = prepare_corpus(source_corpus_path, language_dir)

    try:
        for model_type in TOKENIZER_FORMATS:
            output_dir = language_dir / model_type
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / MODEL_PREFIX
            trainer_kwargs = build_kwargs(
                prepared_corpus.prepared_path,
                output_path,
                model_type,
                args,
            )
            sentencepiece.SentencePieceTrainer.train(**trainer_kwargs)

            print(f"Trained tokenizer for language: {args.language}")
            print(f"Format: {model_type}")
            print(f"Corpus source: {source_corpus_path}")
            print(f"Prepared texts: {prepared_corpus.text_count}")
            print(f"Prepared samples: {prepared_corpus.sample_count}")
            print(f"Model: {output_path}.model")
            print(f"Vocab: {output_path}.vocab")
            print(f"Max vocab size: {args.max_vocab_size}")
            print("Hard vocab limit: False")
            print()
    finally:
        if prepared_corpus.prepared_path.exists():
            prepared_corpus.prepared_path.unlink()


if __name__ == "__main__":
    main()
