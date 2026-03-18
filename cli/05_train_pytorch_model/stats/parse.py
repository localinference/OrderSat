from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass

from reporting.log import log_stage_complete, log_stage_start


@dataclass(frozen=True)
class LengthStats:
    count: int
    min: int
    max: int
    avg: float
    p50: int
    p95: int


@dataclass(frozen=True)
class DatasetStats:
    language: str | None
    corpus_path: str | None
    model_path: str | None
    sample_count: int
    train_count: int
    validation_count: int
    input_lengths: LengthStats
    label_lengths: LengthStats

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_length_stats(
    raw_value: object,
    *,
    path: pathlib.Path,
    field_name: str,
) -> LengthStats:
    if not isinstance(raw_value, dict):
        raise SystemExit(f"Expected object '{field_name}' in {path}")

    required_keys = ("count", "min", "max", "avg", "p50", "p95")
    missing_keys = [key for key in required_keys if key not in raw_value]
    if missing_keys:
        raise SystemExit(
            f"Missing keys in '{field_name}' for {path}: {', '.join(missing_keys)}"
        )

    count = raw_value["count"]
    minimum = raw_value["min"]
    maximum = raw_value["max"]
    average = raw_value["avg"]
    p50 = raw_value["p50"]
    p95 = raw_value["p95"]

    integer_values = {
        "count": count,
        "min": minimum,
        "max": maximum,
        "p50": p50,
        "p95": p95,
    }
    for key, value in integer_values.items():
        if not isinstance(value, int) or value < 0:
            raise SystemExit(
                f"Expected non-negative integer '{field_name}.{key}' in {path}"
            )

    if not isinstance(average, (int, float)) or average < 0:
        raise SystemExit(
            f"Expected non-negative number '{field_name}.avg' in {path}"
        )

    if not (minimum <= p50 <= maximum):
        raise SystemExit(
            f"Inconsistent percentile ordering in '{field_name}' for {path}"
        )
    if not (minimum <= p95 <= maximum):
        raise SystemExit(
            f"Inconsistent percentile ordering in '{field_name}' for {path}"
        )

    return LengthStats(
        count=count,
        min=minimum,
        max=maximum,
        avg=float(average),
        p50=p50,
        p95=p95,
    )


def parse_stats(path: pathlib.Path) -> DatasetStats:
    started_at = time.perf_counter()
    log_stage_start("stats.parse", path=str(path))

    if not path.exists():
        raise SystemExit(f"Stats file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Stats path is not a file: {path}")

    try:
        parsed = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in stats file {path}: {error}") from error

    if not isinstance(parsed, dict):
        raise SystemExit(f"Expected JSON object in stats file {path}")

    required_top_level_keys = (
        "sampleCount",
        "trainCount",
        "validationCount",
        "inputLengths",
        "labelLengths",
    )
    missing_keys = [key for key in required_top_level_keys if key not in parsed]
    if missing_keys:
        raise SystemExit(
            f"Missing keys in stats file {path}: {', '.join(missing_keys)}"
        )

    sample_count = parsed["sampleCount"]
    train_count = parsed["trainCount"]
    validation_count = parsed["validationCount"]

    for key, value in {
        "sampleCount": sample_count,
        "trainCount": train_count,
        "validationCount": validation_count,
    }.items():
        if not isinstance(value, int) or value <= 0:
            raise SystemExit(f"Expected positive integer '{key}' in {path}")

    if train_count + validation_count != sample_count:
        raise SystemExit(
            f"Inconsistent split counts in {path}: "
            f"trainCount + validationCount != sampleCount"
        )

    input_lengths = _parse_length_stats(
        parsed["inputLengths"],
        path=path,
        field_name="inputLengths",
    )
    label_lengths = _parse_length_stats(
        parsed["labelLengths"],
        path=path,
        field_name="labelLengths",
    )

    stats = DatasetStats(
        language=parsed.get("language"),
        corpus_path=parsed.get("corpusPath"),
        model_path=parsed.get("modelPath"),
        sample_count=sample_count,
        train_count=train_count,
        validation_count=validation_count,
        input_lengths=input_lengths,
        label_lengths=label_lengths,
    )

    log_stage_complete(
        "stats.parse",
        duration_seconds=time.perf_counter() - started_at,
        path=str(path),
        sample_count=stats.sample_count,
        train_count=stats.train_count,
        validation_count=stats.validation_count,
        max_input_length=stats.input_lengths.max,
        max_label_length=stats.label_lengths.max,
    )
    return stats
