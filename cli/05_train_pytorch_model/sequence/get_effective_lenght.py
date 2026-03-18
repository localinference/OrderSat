from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from TokenizedJsonlDataset.constructor import TokenizedJsonlDataset
from reporting.log import log_stage_complete, log_stage_start


@dataclass(frozen=True)
class EffectiveSequenceLengths:
    max_input_length: int
    max_label_length: int
    max_source_positions: int
    max_target_positions: int

    def to_dict(self) -> dict:
        return asdict(self)


def _get_observed_max(dataset: TokenizedJsonlDataset, field_name: str) -> int:
    return max(len(record[field_name]) for record in dataset.records)


def get_effective_sequence_lengths(
    *,
    train_dataset: TokenizedJsonlDataset,
    validation_dataset: TokenizedJsonlDataset,
    max_input_length: int,
    max_label_length: int,
) -> EffectiveSequenceLengths:
    started_at = time.perf_counter()
    log_stage_start(
        "sequence.lengths",
        train_path=str(train_dataset.file_path),
        validation_path=str(validation_dataset.file_path),
    )

    observed_max_input_length = max(
        _get_observed_max(train_dataset, "input_ids"),
        _get_observed_max(validation_dataset, "input_ids"),
    )
    observed_max_label_length = max(
        _get_observed_max(train_dataset, "labels"),
        _get_observed_max(validation_dataset, "labels"),
    )

    if observed_max_input_length > max_input_length:
        raise SystemExit(
            "Observed input length exceeds dataset stats max. "
            "Regenerate the dataset stats before training."
        )

    if observed_max_label_length > max_label_length:
        raise SystemExit(
            "Observed label length exceeds dataset stats max. "
            "Regenerate the dataset stats before training."
        )

    sequence_lengths = EffectiveSequenceLengths(
        max_input_length=observed_max_input_length,
        max_label_length=observed_max_label_length,
        max_source_positions=observed_max_input_length,
        max_target_positions=observed_max_label_length + 1,
    )

    log_stage_complete(
        "sequence.lengths",
        duration_seconds=time.perf_counter() - started_at,
        max_input_length=sequence_lengths.max_input_length,
        max_label_length=sequence_lengths.max_label_length,
        max_source_positions=sequence_lengths.max_source_positions,
        max_target_positions=sequence_lengths.max_target_positions,
    )
    return sequence_lengths
