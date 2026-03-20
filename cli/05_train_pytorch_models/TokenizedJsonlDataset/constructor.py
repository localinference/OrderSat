from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import BinaryIO

from torch.utils.data import Dataset

from reporting.log import log_stage_complete, log_stage_start


@dataclass(frozen=True)
class DatasetIndex:
    line_offsets: list[int]
    input_lengths: list[int]
    label_lengths: list[int]
    input_token_count: int
    label_token_count: int


class TokenizedJsonlDataset(Dataset):
    def __init__(self, file_path: pathlib.Path, vocab_size: int) -> None:
        self.file_path = file_path
        self.vocab_size = vocab_size
        self._handle: BinaryIO | None = None

        dataset_index = self._scan_records(file_path, vocab_size)
        self.line_offsets = dataset_index.line_offsets
        self.input_lengths = dataset_index.input_lengths
        self.label_lengths = dataset_index.label_lengths
        self.target_lengths = [length + 1 for length in self.label_lengths]
        self.sequence_token_counts = [
            input_length + target_length
            for input_length, target_length in zip(
                self.input_lengths,
                self.target_lengths,
            )
        ]

    @staticmethod
    def _scan_records(file_path: pathlib.Path, vocab_size: int) -> DatasetIndex:
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

        line_offsets: list[int] = []
        input_lengths: list[int] = []
        label_lengths: list[int] = []
        input_token_count = 0
        label_token_count = 0

        with file_path.open("rb") as handle:
            line_number = 0
            while True:
                line_offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break

                line_number += 1
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    decoded_line = line.decode("utf8")
                except UnicodeDecodeError as error:
                    raise SystemExit(
                        f"Invalid UTF-8 in {file_path} at line {line_number}: {error}"
                    ) from error

                try:
                    parsed = json.loads(decoded_line)
                except json.JSONDecodeError as error:
                    raise SystemExit(
                        f"Invalid JSONL in {file_path} at line {line_number}: {error}"
                    ) from error

                input_length, label_length = TokenizedJsonlDataset._validate_record(
                    parsed=parsed,
                    vocab_size=vocab_size,
                    file_path=file_path,
                    line_number=line_number,
                )

                line_offsets.append(line_offset)
                input_lengths.append(input_length)
                label_lengths.append(label_length)
                input_token_count += input_length
                label_token_count += label_length

        if not line_offsets:
            raise SystemExit(f"Dataset file is empty: {file_path}")

        log_stage_complete(
            "dataset.load",
            duration_seconds=time.perf_counter() - started_at,
            path=str(file_path),
            sample_count=len(line_offsets),
            input_token_count=input_token_count,
            label_token_count=label_token_count,
            storage_mode="indexed_lazy",
            index_entry_count=len(line_offsets),
        )

        return DatasetIndex(
            line_offsets=line_offsets,
            input_lengths=input_lengths,
            label_lengths=label_lengths,
            input_token_count=input_token_count,
            label_token_count=label_token_count,
        )

    @staticmethod
    def _validate_record(
        parsed: object,
        vocab_size: int,
        file_path: pathlib.Path,
        line_number: int,
    ) -> tuple[int, int]:
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

        lengths: dict[str, int] = {}
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
            lengths[key] = len(value)

        return lengths["input_ids"], lengths["labels"]

    def __len__(self) -> int:
        return len(self.line_offsets)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self.line_offsets):
            raise IndexError(index)

        handle = self._get_handle()
        handle.seek(self.line_offsets[index])
        raw_line = handle.readline()
        if not raw_line:
            raise SystemExit(
                f"Failed to read dataset record at index {index} from {self.file_path}"
            )

        line = raw_line.strip()
        try:
            parsed = json.loads(line.decode("utf8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"Failed to parse dataset record at index {index} from {self.file_path}: {error}"
            ) from error

        if not isinstance(parsed, dict):
            raise SystemExit(
                f"Expected JSON object when reading dataset index {index} from {self.file_path}"
            )
        return parsed

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self._close_handle()

    def _get_handle(self) -> BinaryIO:
        if self._handle is None:
            self._handle = self.file_path.open("rb")
        return self._handle

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def get_input_length(self, index: int) -> int:
        return self.input_lengths[index]

    def get_label_length(self, index: int) -> int:
        return self.label_lengths[index]

    def get_target_length(self, index: int) -> int:
        return self.target_lengths[index]

    def get_sequence_token_count(self, index: int) -> int:
        return self.sequence_token_counts[index]

    def get_max_input_length(self) -> int:
        return max(self.input_lengths)

    def get_max_label_length(self) -> int:
        return max(self.label_lengths)
