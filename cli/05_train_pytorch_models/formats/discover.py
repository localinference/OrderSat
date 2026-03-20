from __future__ import annotations

import pathlib


TOKENIZER_VOCAB_FILE = "tokenizer.vocab"
TRAINING_DATA_FILE = "train.jsonl"
VALIDATION_DATA_FILE = "validation.jsonl"
STATS_FILE = "stats.json"


def discover_available_formats(
    *,
    language: str,
    tokenizers_root: pathlib.Path,
    datasets_root: pathlib.Path,
) -> list[str]:
    tokenizer_language_dir = tokenizers_root / language
    dataset_language_dir = datasets_root / language

    if not tokenizer_language_dir.exists():
        raise SystemExit(
            f"Tokenizer language directory does not exist: {tokenizer_language_dir}"
        )
    if not dataset_language_dir.exists():
        raise SystemExit(
            f"Dataset language directory does not exist: {dataset_language_dir}"
        )

    formats: list[str] = []
    for entry in sorted(
        tokenizer_language_dir.iterdir(),
        key=lambda path: path.name,
    ):
        if not entry.is_dir():
            continue

        format_name = entry.name
        vocab_path = entry / TOKENIZER_VOCAB_FILE
        dataset_dir = dataset_language_dir / format_name
        required_dataset_paths = (
            dataset_dir / TRAINING_DATA_FILE,
            dataset_dir / VALIDATION_DATA_FILE,
            dataset_dir / STATS_FILE,
        )

        if not vocab_path.is_file():
            continue
        if not dataset_dir.is_dir():
            continue
        if not all(path.is_file() for path in required_dataset_paths):
            continue

        formats.append(format_name)

    if not formats:
        raise SystemExit(
            "No training formats discovered under "
            f"{tokenizer_language_dir} and {dataset_language_dir}"
        )

    return formats
