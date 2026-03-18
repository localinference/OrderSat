from __future__ import annotations

import json
import pathlib
import time

from torch.utils.data import Dataset

from reporting.log import log_stage_complete, log_stage_start


class TokenizedJsonlDataset(Dataset):
    def __init__(self, file_path: pathlib.Path, vocab_size: int) -> None:
        self.file_path = file_path
        self.records = self._load_records(file_path, vocab_size)
        self.input_lengths = [len(record["input_ids"]) for record in self.records]
        self.target_lengths = [len(record["labels"]) + 1 for record in self.records]
        self.sequence_token_counts = [
            input_length + target_length
            for input_length, target_length in zip(
                self.input_lengths,
                self.target_lengths,
            )
        ]

    @staticmethod
    def _load_records(file_path: pathlib.Path, vocab_size: int) -> list[dict]:
        started_at = time.perf_counter()
        log_stage_start(
            "dataset.load",
            path=str(file_path),
            vocab_size=vocab_size,
        )

        if not file_path.exists():
            raise SystemExit(f"Dataset file does not exist: {file_path}")
        if not file_path.is_file():
            raise SystemExit(f"Dataset path is not a file: {file_path}")

        records: list[dict] = []

        with file_path.open("r", encoding="utf8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(
                        f"Invalid JSONL in {file_path} at line {line_number}: {error}"
                    ) from error

                TokenizedJsonlDataset._validate_record(
                    parsed=parsed,
                    vocab_size=vocab_size,
                    file_path=file_path,
                    line_number=line_number,
                )
                records.append(parsed)

        if not records:
            raise SystemExit(f"Dataset file is empty: {file_path}")

        input_token_count = sum(len(record["input_ids"]) for record in records)
        label_token_count = sum(len(record["labels"]) for record in records)
        log_stage_complete(
            "dataset.load",
            duration_seconds=time.perf_counter() - started_at,
            path=str(file_path),
            sample_count=len(records),
            input_token_count=input_token_count,
            label_token_count=label_token_count,
        )

        return records

    @staticmethod
    def _validate_record(
        parsed: object,
        vocab_size: int,
        file_path: pathlib.Path,
        line_number: int,
    ) -> None:
        if not isinstance(parsed, dict):
            raise SystemExit(
                f"Expected JSON object in {file_path} at line {line_number}"
            )

        required_keys = ("sample_id", "input_ids", "labels")
        for key in required_keys:
            if key not in parsed:
                raise SystemExit(
                    f"Missing key '{key}' in {file_path} at line {line_number}"
                )

        for key in ("input_ids", "labels"):
            value = parsed[key]
            if not isinstance(value, list) or not value:
                raise SystemExit(
                    f"Expected non-empty list '{key}' in {file_path} at line {line_number}"
                )
            if not all(isinstance(token_id, int) for token_id in value):
                raise SystemExit(
                    f"Expected integer token ids in '{key}' for {file_path} at line {line_number}"
                )
            max_token_id = max(value)
            min_token_id = min(value)
            if min_token_id < 0 or max_token_id >= vocab_size:
                raise SystemExit(
                    f"Token id out of range in '{key}' for {file_path} at line {line_number}: "
                    f"expected 0 <= id < {vocab_size}, got range [{min_token_id}, {max_token_id}]"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def get_input_length(self, index: int) -> int:
        return self.input_lengths[index]

    def get_target_length(self, index: int) -> int:
        return self.target_lengths[index]

    def get_sequence_token_count(self, index: int) -> int:
        return self.sequence_token_counts[index]
