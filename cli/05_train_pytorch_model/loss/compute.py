import Seq2SeqTransformer
import torch
from torch import F
def compute_loss(
    model: Seq2SeqTransformer,
    batch: dict,
    *,
    device: str,
    label_pad_id: int,
) -> torch.Tensor:
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        decoder_input_ids=batch["decoder_input_ids"].to(device),
    )
    labels = batch["labels"].to(device)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=label_pad_id,
    )
