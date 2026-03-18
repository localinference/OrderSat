from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from greedy.generate import greedy_generate
from reporting.log import log_stage_complete, log_stage_start


@dataclass(frozen=True)
class ExactMatchEvaluation:
    exact_match: float
    match_count: int
    sample_count: int
    duration_seconds: float


def compute_exact_match(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
) -> ExactMatchEvaluation:
    log_stage_start(
        "validation.exact_match",
        device=str(device),
        batch_count=len(loader),
        max_generation_length=max_generation_length,
    )
    started_at = time.perf_counter()
    match_count = 0
    sample_count = 0

    with torch.inference_mode():
        for batch in loader:
            predictions = greedy_generate(
                model,
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                bos_id=bos_id,
                eos_id=eos_id,
                max_generation_length=max_generation_length,
            )
            targets = batch["target_token_ids"]

            for predicted, target in zip(predictions, targets):
                sample_count += 1
                if predicted == target:
                    match_count += 1

    exact_match = 0.0
    if sample_count > 0:
        exact_match = match_count / sample_count

    result = ExactMatchEvaluation(
        exact_match=exact_match,
        match_count=match_count,
        sample_count=sample_count,
        duration_seconds=time.perf_counter() - started_at,
    )
    log_stage_complete(
        "validation.exact_match",
        duration_seconds=result.duration_seconds,
        sample_count=result.sample_count,
        match_count=result.match_count,
        exact_match=f"{result.exact_match:.4f}",
    )
    return result
