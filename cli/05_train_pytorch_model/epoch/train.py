from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from loss.compute import compute_loss
from reporting.log import log_stage_complete, log_stage_start


@dataclass(frozen=True)
class TrainEpochResult:
    average_loss: float
    token_count: int
    sample_count: int
    batch_count: int
    optimizer_steps: int
    duration_seconds: float


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    label_pad_id: int,
    grad_clip: float,
    accumulation_steps: int,
) -> TrainEpochResult:
    log_stage_start(
        "train.epoch",
        device=str(device),
        batch_count=len(loader),
        accumulation_steps=accumulation_steps,
    )
    model.train()
    started_at = time.perf_counter()
    total_loss_sum = 0.0
    total_token_count = 0
    total_sample_count = 0
    batch_count = 0
    optimizer_steps = 0
    expected_batch_count = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        loss_result = compute_loss(
            model,
            batch,
            device=device,
            label_pad_id=label_pad_id,
        )

        scaled_loss = loss_result.loss / accumulation_steps
        scaled_loss.backward()

        should_step = (
            batch_index % accumulation_steps == 0
            or batch_index == expected_batch_count
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        total_loss_sum += float(loss_result.loss.item()) * loss_result.token_count
        total_token_count += loss_result.token_count
        total_sample_count += loss_result.sample_count
        batch_count += 1

    average_loss = 0.0
    if total_token_count > 0:
        average_loss = total_loss_sum / total_token_count

    result = TrainEpochResult(
        average_loss=average_loss,
        token_count=total_token_count,
        sample_count=total_sample_count,
        batch_count=batch_count,
        optimizer_steps=optimizer_steps,
        duration_seconds=time.perf_counter() - started_at,
    )
    log_stage_complete(
        "train.epoch",
        duration_seconds=result.duration_seconds,
        average_loss=f"{result.average_loss:.6f}",
        token_count=result.token_count,
        sample_count=result.sample_count,
        batch_count=result.batch_count,
        optimizer_steps=result.optimizer_steps,
    )
    return result
