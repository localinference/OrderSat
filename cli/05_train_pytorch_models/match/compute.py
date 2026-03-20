from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

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
    split_name: str,
    device: torch.device,
    bos_id: int,
    eos_id: int,
) -> ExactMatchEvaluation:
    stage_name = f"{split_name}.exact_match"
    model.eval()
    log_stage_start(
        stage_name,
        device=str(device),
        batch_count=len(loader),
        decode_policy="target_aware",
    )
    started_at = time.perf_counter()
    match_count = 0
    sample_count = 0
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=non_blocking,
            )
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            label_lengths = batch["label_lengths"].to(
                device,
                non_blocking=non_blocking,
            )

            memory, source_padding_mask = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            batch_size = input_ids.size(0)
            generated = torch.full(
                (batch_size, 1),
                fill_value=bos_id,
                dtype=input_ids.dtype,
                device=device,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            exact_matches = torch.ones(batch_size, dtype=torch.bool, device=device)
            max_target_steps = int(label_lengths.max().item())

            for step in range(max_target_steps):
                if bool(finished.all()):
                    break

                logits = model.decode_step(
                    decoder_input_ids=generated,
                    memory=memory,
                    source_padding_mask=source_padding_mask,
                )
                next_token = logits[:, -1, :].argmax(dim=-1)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

                active = ~finished
                expected_token = labels[:, step]
                matches = next_token.eq(expected_token)
                exact_matches = torch.where(
                    active,
                    exact_matches & matches,
                    exact_matches,
                )

                should_finish = (~matches) | expected_token.eq(eos_id)
                finished = finished | (active & should_finish)

            sample_count += batch_size
            match_count += int(exact_matches.sum().item())

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
        stage_name,
        duration_seconds=result.duration_seconds,
        sample_count=result.sample_count,
        match_count=result.match_count,
        exact_match=f"{result.exact_match:.4f}",
    )
    return result
