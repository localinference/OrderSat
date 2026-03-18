from __future__ import annotations

import json
import pathlib
import time

import torch

from reporting.log import log_stage_complete, log_stage_start


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf8")


def save_artifacts(
    *,
    save_dir: pathlib.Path,
    metrics: dict,
    history: list[dict],
    metadata: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    started_at = time.perf_counter()
    log_stage_start(
        "artifacts.save",
        save_dir=str(save_dir),
        history_length=len(history),
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    best_metrics_path = save_dir / "best_metrics.json"
    history_path = save_dir / "history.json"
    metadata_path = save_dir / "run.json"
    checkpoint_path = save_dir / "best.pt"

    _write_json(best_metrics_path, metrics)
    _write_json(history_path, history)
    _write_json(metadata_path, metadata)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "history": history,
            "metadata": metadata,
        },
        checkpoint_path,
    )

    log_stage_complete(
        "artifacts.save",
        duration_seconds=time.perf_counter() - started_at,
        save_dir=str(save_dir),
        checkpoint_path=str(checkpoint_path),
        best_metrics_path=str(best_metrics_path),
        history_path=str(history_path),
        metadata_path=str(metadata_path),
    )
