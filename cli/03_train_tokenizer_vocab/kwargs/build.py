import argparse
import pathlib

def build_kwargs(
    input_path: pathlib.Path,  output_path: pathlib.Path, args: argparse.Namespace, 
) -> dict:
    return {
        "input": str(input_path),
        "model_prefix": str(output_path),
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "max_sentence_length": args.max_sentence_length,
        "hard_vocab_limit": args.hard_vocab_limit,
    }

