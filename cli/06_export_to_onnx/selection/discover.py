from __future__ import annotations

import pathlib
from dataclasses import dataclass


CHECKPOINT_NAME = "best.pt"
BEST_METRICS_NAME = "best_metrics.json"
RUN_NAME = "run.json"
TOKENIZER_MODEL_NAME = "tokenizer.model"
STATS_NAME = "stats.json"
VALIDATION_DATASET_NAME = "validation.jsonl"


@dataclass(frozen=True)
class CandidatePaths:
    language: str
    format: str
    checkpoint_path: pathlib.Path
    best_metrics_path: pathlib.Path
    run_path: pathlib.Path
    tokenizer_model_path: pathlib.Path
    stats_path: pathlib.Path
    validation_dataset_path: pathlib.Path


def discover_export_candidates(
    *,
    language: str,
    tokenizers_root: pathlib.Path,
    datasets_root: pathlib.Path,
    pytorch_models_root: pathlib.Path,
) -> list[CandidatePaths]:
    language_models_dir = pytorch_models_root / language
    if not language_models_dir.exists():
        raise SystemExit(
            f"PyTorch model language directory does not exist: {language_models_dir}"
        )

    candidates: list[CandidatePaths] = []
    for entry in sorted(language_models_dir.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue

        format_name = entry.name
        checkpoint_path = entry / CHECKPOINT_NAME
        best_metrics_path = entry / BEST_METRICS_NAME
        run_path = entry / RUN_NAME
        tokenizer_model_path = tokenizers_root / language / format_name / TOKENIZER_MODEL_NAME
        stats_path = datasets_root / language / format_name / STATS_NAME
        validation_dataset_path = (
            datasets_root / language / format_name / VALIDATION_DATASET_NAME
        )

        required_paths = (
            checkpoint_path,
            best_metrics_path,
            run_path,
            tokenizer_model_path,
            stats_path,
            validation_dataset_path,
        )
        if not all(path.is_file() for path in required_paths):
            continue

        candidates.append(
            CandidatePaths(
                language=language,
                format=format_name,
                checkpoint_path=checkpoint_path,
                best_metrics_path=best_metrics_path,
                run_path=run_path,
                tokenizer_model_path=tokenizer_model_path,
                stats_path=stats_path,
                validation_dataset_path=validation_dataset_path,
            )
        )

    if not candidates:
        raise SystemExit(
            f"No export candidates discovered for language '{language}' under "
            f"{language_models_dir}"
        )

    return candidates
