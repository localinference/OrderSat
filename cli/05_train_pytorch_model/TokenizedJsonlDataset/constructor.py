import pathlib
import json
from torch.utils.data import Dataset


class TokenizedJsonlDataset(Dataset):
    def __init__(self, file_path: pathlib.Path, vocab_size: int) -> None:
        self.file_path = file_path
        self.records = self._load_records(file_path, vocab_size)

    @staticmethod
    def _load_records(file_path: pathlib.Path, vocab_size: int) -> list[dict]:
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
