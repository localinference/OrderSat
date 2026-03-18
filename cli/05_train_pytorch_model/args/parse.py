import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the seq2seq Transformer on tokenized JSONL datasets."
        )
    )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        default="eng",
        help=(
            "Language directory under src/03_tokenizers, "
            "src/04_training_datasets, and src/05_pytorch_models."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()
