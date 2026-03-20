from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossComputation:
    loss: torch.Tensor
    token_count: int
    sample_count: int


def compute_loss(
    model: torch.nn.Module,
    batch: dict,
    *,
    device: torch.device,
    label_pad_id: int,
) -> LossComputation:
    non_blocking = device.type == "cuda"
    input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
    attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
    decoder_input_ids = batch["decoder_input_ids"].to(
        device,
        non_blocking=non_blocking,
    )
    labels = batch["labels"].to(device, non_blocking=non_blocking)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

    token_count = int(labels.ne(label_pad_id).sum().item())
    if token_count <= 0:
        raise SystemExit("Encountered batch without any supervised label tokens.")

    # CrossEntropyLoss still requires int64 class indices even though the
    # training pipeline stores token ids as int32.
    labels_for_loss = labels.to(dtype=torch.long)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels_for_loss.reshape(-1),
        ignore_index=label_pad_id,
        reduction="sum",
    )
    loss = loss_sum / token_count

    return LossComputation(
        loss=loss,
        token_count=token_count,
        sample_count=int(labels.size(0)),
    )
