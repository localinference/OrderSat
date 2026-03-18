import argparse
import pathlib

def build_kwargs(
    input_path: pathlib.Path,  output_path: pathlib.Path, args: argparse.Namespace, 
) -> dict:
    return {
        "input": str(input_path),
        "model_prefix": str(output_path),
        "model_type": args.model_type,
        "vocab_size": args.vocab_size,
        "character_coverage": args.character_coverage,
        "max_sentence_length": args.max_sentence_length,
    }

