#!/usr/bin/env python3
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a SentencePiece tokenizer for one language corpus under "
            "src/03_tokenizers/{lang}/corpus.jsonl."
        )
    )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        default="eng",
        help="Language directory name under src/03_tokenizers, for example: eng",
    )
    parser.add_argument(
        "-V",
        "--vocab-size",
        type=int,
        default=8000,
        help="SentencePiece vocab size (default: 8000)",
    )
    parser.add_argument(
        "-M",
        "--model-type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="SentencePiece model type (default: unigram)",
    )
    parser.add_argument(
        "-C",
        "--character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage (default: 1.0)",
    )
    parser.add_argument(
        "-L",
        "max-sentence-length",
        type=int,
        default=16384,
        help="SentencePiece character coverage (default: 1.0)",
    )
    return parser.parse_args()
