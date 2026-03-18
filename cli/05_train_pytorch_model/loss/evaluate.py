from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from loss.compute import compute_loss


@dataclass(frozen=True)
class LossEvaluationResult:
    average_loss: float
    token_count: int
    sample_count: int
    batch_count: int
    duration_seconds: float


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    label_pad_id: int,
) -> LossEvaluationResult:
    model.eval()
    started_at = time.perf_counter()
    total_loss_sum = 0.0
    total_token_count = 0
    total_sample_count = 0
    batch_count = 0

    with torch.inference_mode():
        for batch in loader:
            loss_result = compute_loss(
                model,
                batch,
                device=device,
                label_pad_id=label_pad_id,
            )
            total_loss_sum += float(loss_result.loss.item()) * loss_result.token_count
            total_token_count += loss_result.token_count
            total_sample_count += loss_result.sample_count
            batch_count += 1

    average_loss = 0.0
    if total_token_count > 0:
        average_loss = total_loss_sum / total_token_count

    return LossEvaluationResult(
        average_loss=average_loss,
        token_count=total_token_count,
        sample_count=total_sample_count,
        batch_count=batch_count,
        duration_seconds=time.perf_counter() - started_at,
    )
