import argparse

from checkpoint.load import CHECKPOINT_MODES
from reporting.log import log_event


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
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        choices=CHECKPOINT_MODES,
        default="auto",
        help=(
            "Checkpoint reuse policy: auto warm-starts from the current best.pt "
            "when compatible, fresh ignores it, warm_start requires compatible "
            "weights, and resume restores optimizer and runtime state."
        ),
    )
    args = parser.parse_args()
    log_event(
        "args.parsed",
        language=args.language,
        requested_device=args.device,
        checkpoint_mode=args.checkpoint_mode,
    )
    return args
