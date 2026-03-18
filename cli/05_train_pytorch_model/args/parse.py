import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load tokenized seq2seq datasets, pad them with a collator, and "
            "run a tiny random-init Transformer overfit experiment."
        )
    )
    parser.add_argument(
        "-L"
        "--language",
        type=str,
        default="eng",
        help="Language directory under src/03_tokenizers, src/04_training_datasets, and src/05_pytorch_models.",
    )
    return parser.parse_args()
