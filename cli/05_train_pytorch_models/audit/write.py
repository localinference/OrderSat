from __future__ import annotations

import json
import pathlib

from audit.compute import DecodeAuditResult


def write_decode_audit_artifacts(
    *,
    save_dir: pathlib.Path,
    validation_audit: DecodeAuditResult,
    train_audit: DecodeAuditResult,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    audit_metrics_payload = {
        "validation": validation_audit.to_summary_dict(),
        "train_audit": train_audit.to_summary_dict(),
    }
    (save_dir / "audit_metrics.json").write_text(
        f"{json.dumps(audit_metrics_payload, indent=2)}\n",
        encoding="utf8",
    )
    _write_jsonl(
        save_dir / "decoded_validation.jsonl",
        validation_audit.records,
    )
    _write_jsonl(
        save_dir / "decoded_train_audit.jsonl",
        train_audit.records,
    )


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
