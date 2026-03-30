from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import sentencepiece as spm
import torch

from greedy.generate import greedy_generate
from reporting.log import log_event, log_stage_complete, log_stage_start


FIELD_NAMES = (
    "orderNumber",
    "orderDate",
    "totalPaymentDue.price",
    "seller.name",
    "customer.name",
    "trackingNumber",
    "deliveryAddress",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STRUCTURE_VALIDATOR_PATH = REPO_ROOT / "cli" / "validateStructure" / "batch.js"


@dataclass(frozen=True)
class FieldAuditMetric:
    relevant_count: int
    match_count: int
    match_rate: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "relevant_count": self.relevant_count,
            "match_count": self.match_count,
            "match_rate": self.match_rate,
        }


@dataclass(frozen=True)
class DecodeAuditResult:
    split_name: str
    sample_count: int
    greedy_exact_match_count: int
    greedy_exact_match_rate: float
    valid_json_count: int
    valid_json_rate: float
    valid_structure_count: int | None
    valid_structure_rate: float | None
    structure_validation_available: bool
    structure_validation_error: str | None
    field_metrics: dict[str, FieldAuditMetric]
    field_macro_match_rate: float | None
    duration_seconds: float
    records: list[dict]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "sample_count": self.sample_count,
            "greedy_exact_match_count": self.greedy_exact_match_count,
            "greedy_exact_match_rate": self.greedy_exact_match_rate,
            "valid_json_count": self.valid_json_count,
            "valid_json_rate": self.valid_json_rate,
            "valid_structure_count": self.valid_structure_count,
            "valid_structure_rate": self.valid_structure_rate,
            "structure_validation_available": self.structure_validation_available,
            "structure_validation_error": self.structure_validation_error,
            "field_metrics": {
                name: metric.to_dict()
                for name, metric in self.field_metrics.items()
            },
            "field_macro_match_rate": self.field_macro_match_rate,
            "duration_seconds": self.duration_seconds,
        }


def compute_decode_audit(
    model: torch.nn.Module,
    batches: Iterable[dict],
    *,
    split_name: str,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
    tokenizer_model_path: pathlib.Path,
    batch_count_hint: int | None = None,
) -> DecodeAuditResult:
    stage_name = f"{split_name}.decode_audit"
    log_stage_start(
        stage_name,
        device=str(device),
        batch_count=batch_count_hint,
        max_generation_length=max_generation_length,
    )
    started_at = time.perf_counter()

    processor = spm.SentencePieceProcessor()
    if not processor.load(str(tokenizer_model_path)):
        raise SystemExit(f"Failed to load tokenizer model: {tokenizer_model_path}")

    records: list[dict] = []
    sample_count = 0
    greedy_exact_match_count = 0
    valid_json_count = 0
    field_counts = {
        field_name: {"relevant_count": 0, "match_count": 0}
        for field_name in FIELD_NAMES
    }

    model.eval()
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for batch in batches:
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=non_blocking,
            )
            predicted_token_ids = greedy_generate(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                bos_id=bos_id,
                eos_id=eos_id,
                max_generation_length=max_generation_length,
            )

            sample_ids = batch["sample_ids"]
            source_lines = batch.get("source_lines") or [None] * len(sample_ids)
            input_texts = batch.get("input_texts") or [""] * len(sample_ids)
            target_texts = batch.get("output_texts") or [""] * len(sample_ids)

            for index, sample_id in enumerate(sample_ids):
                predicted_text = processor.decode_ids(predicted_token_ids[index])
                target_text = target_texts[index]
                greedy_exact_match = predicted_text == target_text
                if greedy_exact_match:
                    greedy_exact_match_count += 1

                predicted_json = _parse_json(predicted_text)
                target_json = _parse_json(target_text)
                valid_json = predicted_json is not None
                if valid_json:
                    valid_json_count += 1

                target_fields = _extract_grounding_fields(target_json)
                predicted_fields = _extract_grounding_fields(predicted_json)
                per_field: dict[str, dict[str, object]] = {}
                for field_name in FIELD_NAMES:
                    target_value = target_fields[field_name]
                    predicted_value = predicted_fields[field_name]
                    relevant = target_value is not None
                    match = bool(relevant and predicted_value == target_value)
                    if relevant:
                        field_counts[field_name]["relevant_count"] += 1
                        if match:
                            field_counts[field_name]["match_count"] += 1
                    per_field[field_name] = {
                        "relevant": relevant,
                        "match": match if relevant else None,
                        "target": target_value,
                        "predicted": predicted_value,
                    }

                records.append(
                    {
                        "sample_id": sample_id,
                        "source_line": source_lines[index],
                        "input_text": input_texts[index],
                        "target_text": target_text,
                        "predicted_text": predicted_text,
                        "greedy_exact_match": greedy_exact_match,
                        "valid_json": valid_json,
                        "valid_structure": None,
                        "structure_issues": None,
                        "fields": per_field,
                    }
                )
                sample_count += 1

    structure_validation = _apply_structure_validation(records)
    field_metrics = _build_field_metrics(field_counts)

    result = DecodeAuditResult(
        split_name=split_name,
        sample_count=sample_count,
        greedy_exact_match_count=greedy_exact_match_count,
        greedy_exact_match_rate=_safe_rate(greedy_exact_match_count, sample_count),
        valid_json_count=valid_json_count,
        valid_json_rate=_safe_rate(valid_json_count, sample_count),
        valid_structure_count=structure_validation["valid_structure_count"],
        valid_structure_rate=structure_validation["valid_structure_rate"],
        structure_validation_available=structure_validation["available"],
        structure_validation_error=structure_validation["error"],
        field_metrics=field_metrics,
        field_macro_match_rate=_build_field_macro_match_rate(field_metrics),
        duration_seconds=time.perf_counter() - started_at,
        records=records,
    )

    log_stage_complete(
        stage_name,
        duration_seconds=result.duration_seconds,
        sample_count=result.sample_count,
        greedy_exact_match=f"{result.greedy_exact_match_rate:.4f}",
        valid_json_rate=f"{result.valid_json_rate:.4f}",
        valid_structure_rate=(
            f"{result.valid_structure_rate:.4f}"
            if result.valid_structure_rate is not None
            else "<unavailable>"
        ),
        field_macro_match_rate=(
            f"{result.field_macro_match_rate:.4f}"
            if result.field_macro_match_rate is not None
            else "<unavailable>"
        ),
    )
    if result.structure_validation_error:
        log_event(
            f"{stage_name}.structure_validation_unavailable",
            error=result.structure_validation_error,
        )
    return result


def _parse_json(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _apply_structure_validation(records: list[dict]) -> dict[str, object]:
    valid_json_rows = [
        {
            "sample_id": record["sample_id"],
            "output_text": record["predicted_text"],
        }
        for record in records
        if record["valid_json"]
    ]

    if not valid_json_rows:
        for record in records:
            record["valid_structure"] = False
            record["structure_issues"] = ["prediction is not valid JSON"]
        return {
            "available": True,
            "error": None,
            "valid_structure_count": 0,
            "valid_structure_rate": 0.0,
        }

    temp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf8",
            suffix=".jsonl",
            prefix="decode-structure-audit-",
            dir=REPO_ROOT,
            delete=False,
        ) as handle:
            temp_path = pathlib.Path(handle.name)
            for row in valid_json_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        completed = subprocess.run(
            ["node", str(STRUCTURE_VALIDATOR_PATH), str(temp_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        return _mark_structure_unavailable(records, f"Node is unavailable: {error}")
    except OSError as error:
        return _mark_structure_unavailable(records, str(error))
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except Exception:
                pass

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return _mark_structure_unavailable(
            records,
            f"batch validator failed with exit code {completed.returncode}: {stderr}",
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return _mark_structure_unavailable(
            records,
            f"batch validator returned invalid JSON: {error}",
        )

    results = payload.get("results")
    if not isinstance(results, list):
        return _mark_structure_unavailable(
            records,
            "batch validator result payload is missing 'results'",
        )

    result_map: dict[str, dict] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id")
        if isinstance(sample_id, str):
            result_map[sample_id] = item

    valid_structure_count = 0
    for record in records:
        if not record["valid_json"]:
            record["valid_structure"] = False
            record["structure_issues"] = ["prediction is not valid JSON"]
            continue

        validation = result_map.get(record["sample_id"])
        if validation is None:
            record["valid_structure"] = False
            record["structure_issues"] = ["missing structure validation result"]
            continue

        valid_structure = bool(validation.get("valid_structure"))
        issues = validation.get("issues")
        record["valid_structure"] = valid_structure
        record["structure_issues"] = issues if isinstance(issues, list) else []
        if valid_structure:
            valid_structure_count += 1

    return {
        "available": True,
        "error": None,
        "valid_structure_count": valid_structure_count,
        "valid_structure_rate": _safe_rate(valid_structure_count, len(records)),
    }


def _mark_structure_unavailable(
    records: list[dict],
    error_message: str,
) -> dict[str, object]:
    for record in records:
        record["valid_structure"] = None
        record["structure_issues"] = [error_message]
    return {
        "available": False,
        "error": error_message,
        "valid_structure_count": None,
        "valid_structure_rate": None,
    }


def _build_field_metrics(
    field_counts: dict[str, dict[str, int]],
) -> dict[str, FieldAuditMetric]:
    metrics: dict[str, FieldAuditMetric] = {}
    for field_name, counts in field_counts.items():
        relevant_count = counts["relevant_count"]
        match_count = counts["match_count"]
        metrics[field_name] = FieldAuditMetric(
            relevant_count=relevant_count,
            match_count=match_count,
            match_rate=(
                _safe_rate(match_count, relevant_count)
                if relevant_count > 0
                else None
            ),
        )
    return metrics


def _build_field_macro_match_rate(
    field_metrics: dict[str, FieldAuditMetric],
) -> float | None:
    rates = [
        metric.match_rate
        for metric in field_metrics.values()
        if metric.match_rate is not None
    ]
    if not rates:
        return None
    return sum(rates) / len(rates)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _stable_json_value(value):
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key in sorted(value):
            normalized[key] = _stable_json_value(value[key])
        return normalized
    return value


def _stable_stringify(value) -> str:
    return json.dumps(
        _stable_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _type_matches(node: dict, type_name: str) -> bool:
    node_type = node.get("@type")
    if isinstance(node_type, str):
        return node_type == type_name
    if isinstance(node_type, list):
        return any(isinstance(item, str) and item == type_name for item in node_type)
    return False


def _resolve_ref(value, nodes_by_id: dict[str, dict]) -> dict | None:
    if isinstance(value, dict):
        ref_id = value.get("@id")
        if isinstance(ref_id, str) and ref_id in nodes_by_id:
            return nodes_by_id[ref_id]
        return value
    if isinstance(value, str):
        return nodes_by_id.get(value)
    return None


def _find_order_node(graph: list[dict]) -> dict | None:
    for node in graph:
        if isinstance(node, dict) and _type_matches(node, "Order"):
            return node
    return None


def _find_named_property(properties, name: str) -> str | None:
    if not isinstance(properties, list):
        return None
    for item in properties:
        if not isinstance(item, dict):
            continue
        if item.get("name") != name:
            continue
        return _normalize_scalar(item.get("value"))
    return None


def _normalize_scalar(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _normalize_address(value) -> str | None:
    if isinstance(value, dict):
        return _stable_stringify(value)
    return None


def _extract_grounding_fields(parsed: dict | None) -> dict[str, str | None]:
    fields = {field_name: None for field_name in FIELD_NAMES}
    if not isinstance(parsed, dict):
        return fields

    graph = parsed.get("@graph")
    if not isinstance(graph, list):
        return fields

    nodes_by_id = {
        node_id: node
        for node in graph
        if isinstance(node, dict)
        and isinstance((node_id := node.get("@id")), str)
    }

    order_node = _find_order_node(graph)
    if order_node is None:
        return fields

    seller_node = _resolve_ref(order_node.get("seller"), nodes_by_id) or nodes_by_id.get(
        "#seller"
    )
    customer_node = _resolve_ref(
        order_node.get("customer"),
        nodes_by_id,
    ) or nodes_by_id.get("#customer")
    delivery_node = _resolve_ref(
        order_node.get("orderDelivery"),
        nodes_by_id,
    ) or nodes_by_id.get("#delivery")

    total_payment_due = order_node.get("totalPaymentDue")
    delivery_address = (
        delivery_node.get("deliveryAddress")
        if isinstance(delivery_node, dict)
        else None
    )

    fields["orderNumber"] = _normalize_scalar(order_node.get("orderNumber"))
    fields["orderDate"] = _normalize_scalar(order_node.get("orderDate"))
    fields["totalPaymentDue.price"] = (
        _normalize_scalar(total_payment_due.get("price"))
        if isinstance(total_payment_due, dict)
        else None
    )
    fields["seller.name"] = (
        _normalize_scalar(seller_node.get("name"))
        if isinstance(seller_node, dict)
        else None
    )
    fields["customer.name"] = (
        _normalize_scalar(customer_node.get("name"))
        if isinstance(customer_node, dict)
        else None
    )
    fields["trackingNumber"] = _find_named_property(
        order_node.get("additionalProperty"),
        "Tracking Number",
    )
    fields["deliveryAddress"] = _normalize_address(delivery_address)
    return fields
