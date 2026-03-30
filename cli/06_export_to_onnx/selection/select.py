from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass

from selection.discover import CandidatePaths


@dataclass(frozen=True)
class CandidateEvaluation:
    language: str
    format: str
    checkpoint_path: str
    tokenizer_model_path: str
    run_path: str
    stats_path: str
    validation_dataset_path: str
    validation_count: int
    validation_exact_match: float
    validation_exact_match_count: int
    validation_exact_match_wilson_lb: float
    validation_loss: float
    validation_token_count: int
    validation_output_char_count: int
    validation_bits_per_output_char: float
    validation_greedy_exact_match_rate: float | None
    validation_valid_json_rate: float | None
    validation_valid_structure_rate: float | None
    validation_field_macro_match_rate: float | None
    validation_structure_validation_available: bool
    parameter_count: int
    runtime_proxy: float
    label_p95: int
    best_epoch: int
    latest_epoch_completed: int
    edge_of_training: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionResult:
    language: str
    selected_format: str
    confidence: str
    reason: str
    selected: CandidateEvaluation
    candidates: list[CandidateEvaluation]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "selected_format": self.selected_format,
            "confidence": self.confidence,
            "reason": self.reason,
            "selected": self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def select_best_candidate(candidates: list[CandidatePaths]) -> SelectionResult:
    evaluations = [_evaluate_candidate(candidate) for candidate in candidates]
    _assert_shared_validation_split(evaluations)

    ranked = sorted(
        evaluations,
        key=lambda candidate: (
            -candidate.validation_exact_match_wilson_lb,
            -candidate.validation_exact_match,
            *_sort_rate_desc(candidate.validation_valid_structure_rate),
            *_sort_rate_desc(candidate.validation_field_macro_match_rate),
            *_sort_rate_desc(candidate.validation_valid_json_rate),
            *_sort_rate_desc(candidate.validation_greedy_exact_match_rate),
            candidate.validation_bits_per_output_char,
            candidate.runtime_proxy,
            candidate.parameter_count,
            candidate.label_p95,
            candidate.format,
        ),
    )
    selected = ranked[0]
    confidence = _build_confidence(selected, ranked)
    reason = _build_reason(selected, ranked)

    return SelectionResult(
        language=selected.language,
        selected_format=selected.format,
        confidence=confidence,
        reason=reason,
        selected=selected,
        candidates=ranked,
    )


def _evaluate_candidate(candidate: CandidatePaths) -> CandidateEvaluation:
    best_metrics = _read_json(candidate.best_metrics_path)
    run_payload = _read_json(candidate.run_path)
    stats_payload = _read_json(candidate.stats_path)
    validation_dataset_info = _read_validation_dataset(candidate.validation_dataset_path)
    audit_metrics_payload = (
        _read_json(candidate.audit_metrics_path)
        if candidate.audit_metrics_path is not None
        else None
    )
    run_paths = _require_dict(run_payload, "run_paths", candidate.run_path)

    _require_string(run_paths, "language", candidate.run_path, candidate.language)
    _require_string(run_paths, "format", candidate.run_path, candidate.format)
    _require_string(stats_payload, "language", candidate.stats_path, candidate.language)
    _require_string(stats_payload, "format", candidate.stats_path, candidate.format)
    _require_bool(
        best_metrics,
        "validation_exact_match_ran",
        candidate.best_metrics_path,
        expected=True,
    )

    validation_count = _require_int(stats_payload, "validationCount", candidate.stats_path)
    validation_exact_match = _require_float(
        best_metrics,
        "validation_exact_match",
        candidate.best_metrics_path,
    )
    validation_exact_match_count = round(validation_exact_match * validation_count)
    validation_loss = _require_float(
        best_metrics,
        "validation_loss",
        candidate.best_metrics_path,
    )
    validation_token_count = _require_int(
        best_metrics,
        "validation_token_count",
        candidate.best_metrics_path,
    )
    parameter_count = _require_int(run_payload, "parameter_count", candidate.run_path)
    dataset_stats = _require_dict(run_payload, "dataset_stats", candidate.run_path)
    label_lengths = _require_dict(dataset_stats, "label_lengths", candidate.run_path)
    input_lengths = _require_dict(dataset_stats, "input_lengths", candidate.run_path)
    runtime_proxy = parameter_count * (
        _require_number(input_lengths, "p95", candidate.run_path)
        + 2 * _require_number(label_lengths, "p95", candidate.run_path)
    )
    runtime_state = _require_dict(run_payload, "runtime_state", candidate.run_path)
    best_epoch = _require_int(best_metrics, "epoch", candidate.best_metrics_path)
    latest_epoch_completed = _require_int(
        runtime_state,
        "latest_epoch_completed",
        candidate.run_path,
    )

    total_validation_nats = validation_loss * validation_token_count
    validation_bits_per_output_char = (
        total_validation_nats
        / validation_dataset_info["output_char_count"]
        / math.log(2.0)
    )
    validation_audit = _read_validation_audit_summary(
        audit_metrics_payload,
        candidate,
    )

    return CandidateEvaluation(
        language=candidate.language,
        format=candidate.format,
        checkpoint_path=str(candidate.checkpoint_path),
        tokenizer_model_path=str(candidate.tokenizer_model_path),
        run_path=str(candidate.run_path),
        stats_path=str(candidate.stats_path),
        validation_dataset_path=str(candidate.validation_dataset_path),
        validation_count=validation_count,
        validation_exact_match=validation_exact_match,
        validation_exact_match_count=validation_exact_match_count,
        validation_exact_match_wilson_lb=_wilson_lower_bound(
            validation_exact_match_count,
            validation_count,
        ),
        validation_loss=validation_loss,
        validation_token_count=validation_token_count,
        validation_output_char_count=validation_dataset_info["output_char_count"],
        validation_bits_per_output_char=validation_bits_per_output_char,
        validation_greedy_exact_match_rate=validation_audit["greedy_exact_match_rate"],
        validation_valid_json_rate=validation_audit["valid_json_rate"],
        validation_valid_structure_rate=validation_audit["valid_structure_rate"],
        validation_field_macro_match_rate=validation_audit["field_macro_match_rate"],
        validation_structure_validation_available=validation_audit[
            "structure_validation_available"
        ],
        parameter_count=parameter_count,
        runtime_proxy=runtime_proxy,
        label_p95=int(_require_number(label_lengths, "p95", candidate.run_path)),
        best_epoch=best_epoch,
        latest_epoch_completed=latest_epoch_completed,
        edge_of_training=best_epoch >= latest_epoch_completed - 1,
    )


def _read_json(path) -> dict:
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _read_validation_dataset(path) -> dict:
    path = pathlib.Path(path)
    rows = []
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSONL line in {path}: {error}") from error
        if not isinstance(payload, dict):
            raise SystemExit(f"Expected JSON object row in {path}")
        sample_id = payload.get("sample_id")
        output_text = payload.get("output_text")
        if not isinstance(sample_id, str):
            raise SystemExit(f"Missing sample_id in {path}")
        if not isinstance(output_text, str):
            raise SystemExit(f"Missing output_text in {path}")
        rows.append((sample_id, len(output_text)))

    if not rows:
        raise SystemExit(f"Validation dataset is empty: {path}")

    rows.sort(key=lambda item: item[0])
    return {
        "sample_ids": tuple(sample_id for sample_id, _ in rows),
        "output_char_count": sum(length for _, length in rows),
    }


def _assert_shared_validation_split(candidates: list[CandidateEvaluation]) -> None:
    if not candidates:
        raise SystemExit("No candidates to compare")

    first_info = _read_validation_dataset(path=candidates[0].validation_dataset_path)
    baseline_ids = first_info["sample_ids"]
    baseline_chars = first_info["output_char_count"]
    for candidate in candidates[1:]:
        candidate_info = _read_validation_dataset(path=candidate.validation_dataset_path)
        if candidate_info["sample_ids"] != baseline_ids:
            raise SystemExit(
                "Validation sample-id sets do not match across candidates, "
                f"cannot compare tokenizers fairly: {candidate.validation_dataset_path}"
            )
        if candidate_info["output_char_count"] != baseline_chars:
            raise SystemExit(
                "Validation output character counts do not match across candidates, "
                f"cannot compare tokenizers fairly: {candidate.validation_dataset_path}"
            )


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + (z * z) / total
    center = proportion + (z * z) / (2.0 * total)
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) / total)
        + ((z * z) / (4.0 * total * total))
    )
    return (center - margin) / denominator


def _build_confidence(
    selected: CandidateEvaluation,
    ranked: list[CandidateEvaluation],
) -> str:
    if selected.validation_count < 100:
        return "low"
    if selected.validation_exact_match <= 0.0:
        return "low"
    if len(ranked) > 1:
        runner_up = ranked[1]
        if abs(selected.validation_exact_match - runner_up.validation_exact_match) < 0.02:
            return "medium"
    if selected.edge_of_training:
        return "medium"
    return "high"


def _build_reason(
    selected: CandidateEvaluation,
    ranked: list[CandidateEvaluation],
) -> str:
    if len(ranked) == 1:
        return f"only candidate available: {selected.format}"

    runner_up = ranked[1]
    if (
        selected.validation_exact_match_wilson_lb
        > runner_up.validation_exact_match_wilson_lb
    ):
        return (
            "higher validation exact-match lower bound: "
            f"{selected.validation_exact_match_wilson_lb:.6f} > "
            f"{runner_up.validation_exact_match_wilson_lb:.6f}"
        )
    if selected.validation_exact_match > runner_up.validation_exact_match:
        return (
            "higher validation exact match: "
            f"{selected.validation_exact_match:.6f} > "
            f"{runner_up.validation_exact_match:.6f}"
        )
    if _strictly_greater(
        selected.validation_valid_structure_rate,
        runner_up.validation_valid_structure_rate,
    ):
        return (
            "higher validation structure-valid rate: "
            f"{_format_rate(selected.validation_valid_structure_rate)} > "
            f"{_format_rate(runner_up.validation_valid_structure_rate)}"
        )
    if _strictly_greater(
        selected.validation_field_macro_match_rate,
        runner_up.validation_field_macro_match_rate,
    ):
        return (
            "higher validation field macro match rate: "
            f"{_format_rate(selected.validation_field_macro_match_rate)} > "
            f"{_format_rate(runner_up.validation_field_macro_match_rate)}"
        )
    if _strictly_greater(
        selected.validation_valid_json_rate,
        runner_up.validation_valid_json_rate,
    ):
        return (
            "higher validation JSON-valid rate: "
            f"{_format_rate(selected.validation_valid_json_rate)} > "
            f"{_format_rate(runner_up.validation_valid_json_rate)}"
        )
    if _strictly_greater(
        selected.validation_greedy_exact_match_rate,
        runner_up.validation_greedy_exact_match_rate,
    ):
        return (
            "higher validation greedy exact match: "
            f"{_format_rate(selected.validation_greedy_exact_match_rate)} > "
            f"{_format_rate(runner_up.validation_greedy_exact_match_rate)}"
        )
    if (
        selected.validation_bits_per_output_char
        < runner_up.validation_bits_per_output_char
    ):
        return (
            "lower validation bits per output char: "
            f"{selected.validation_bits_per_output_char:.6f} < "
            f"{runner_up.validation_bits_per_output_char:.6f}"
        )
    if selected.runtime_proxy < runner_up.runtime_proxy:
        return (
            "lower deployment runtime proxy: "
            f"{selected.runtime_proxy:.0f} < {runner_up.runtime_proxy:.0f}"
        )
    if selected.parameter_count < runner_up.parameter_count:
        return (
            "lower parameter count: "
            f"{selected.parameter_count} < {runner_up.parameter_count}"
        )
    return f"stable lexical tiebreaker: {selected.format}"


def _require_dict(payload: dict, key: str, path) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing object '{key}' in {path}")
    return value


def _require_int(payload: dict, key: str, path) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise SystemExit(f"Missing integer '{key}' in {path}")
    return value


def _require_float(payload: dict, key: str, path) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"Missing number '{key}' in {path}")
    return float(value)


def _require_number(payload: dict, key: str, path) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"Missing numeric '{key}' in {path}")
    return float(value)


def _require_bool(
    payload: dict,
    key: str,
    path,
    *,
    expected: bool | None = None,
) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SystemExit(f"Missing boolean '{key}' in {path}")
    if expected is not None and value is not expected:
        raise SystemExit(f"Unexpected value for '{key}' in {path}: {value}")
    return value


def _require_string(
    payload: dict,
    key: str,
    path,
    expected: str | None = None,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SystemExit(f"Missing string '{key}' in {path}")
    if expected is not None and value != expected:
        raise SystemExit(
            f"Unexpected value for '{key}' in {path}: expected '{expected}', got '{value}'"
        )
    return value


def _read_validation_audit_summary(
    payload: dict | None,
    candidate: CandidatePaths,
) -> dict[str, object]:
    if payload is None:
        return {
            "greedy_exact_match_rate": None,
            "valid_json_rate": None,
            "valid_structure_rate": None,
            "field_macro_match_rate": None,
            "structure_validation_available": False,
        }

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise SystemExit(
            f"Missing validation audit summary in {candidate.audit_metrics_path}"
        )

    return {
        "greedy_exact_match_rate": _optional_rate(
            validation,
            "greedy_exact_match_rate",
            candidate.audit_metrics_path,
        ),
        "valid_json_rate": _optional_rate(
            validation,
            "valid_json_rate",
            candidate.audit_metrics_path,
        ),
        "valid_structure_rate": _optional_rate(
            validation,
            "valid_structure_rate",
            candidate.audit_metrics_path,
        ),
        "field_macro_match_rate": _optional_rate(
            validation,
            "field_macro_match_rate",
            candidate.audit_metrics_path,
        ),
        "structure_validation_available": _optional_bool(
            validation,
            "structure_validation_available",
        ),
    }


def _optional_rate(payload: dict, key: str, path) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise SystemExit(f"Missing numeric '{key}' in {path}")
    return float(value)


def _optional_bool(payload: dict, key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    return False


def _sort_rate_desc(value: float | None) -> tuple[bool, float]:
    if value is None:
        return (True, 0.0)
    return (False, -value)


def _strictly_greater(left: float | None, right: float | None) -> bool:
    if left is None:
        return False
    if right is None:
        return True
    return left > right


def _format_rate(value: float | None) -> str:
    if value is None:
        return "<unavailable>"
    return f"{value:.6f}"
