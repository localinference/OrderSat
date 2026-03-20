import argparse
import pathlib


def build_kwargs(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    model_type: str,
    args: argparse.Namespace,
) -> dict:
    return {
        "input": str(input_path),
        "model_prefix": str(output_path),
        "model_type": model_type,
        "vocab_size": args.max_vocab_size,
        "character_coverage": args.character_coverage,
        "max_sentence_length": args.max_sentence_length,
        "hard_vocab_limit": False,
    }

