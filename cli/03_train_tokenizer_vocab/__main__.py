#!/usr/bin/env python3
import pathlib

from args.parse import parse_args

from kwargs.build import build_kwargs

try:
    import sentencepiece as spm
except ImportError as error:
    raise SystemExit(
        "sentencepiece is required to train tokenizer vocabs. "
        "Install it in your Python environment first."
    ) from error


TOKENIZERS_DIR = pathlib.Path("src/03_tokenizers")
CORPUS_NAME = "corpus.jsonl"
MODEL_PREFIX = "tokenizer"


def main() -> None:
    args = parse_args()
    language_dir = TOKENIZERS_DIR / args.language
    input_path = language_dir / CORPUS_NAME
    output_path = language_dir / MODEL_PREFIX

    trainer_kwargs = build_kwargs(input_path, output_path, args)

    spm.SentencePieceTrainer.train(**trainer_kwargs)

    print(f"Trained tokenizer for language: {args.language}")
    print(f"Corpus: {input_path}")
    print(f"Model: {output_path}.model")
    print(f"Vocab: {output_path}.vocab")


if __name__ == "__main__":
    main()


