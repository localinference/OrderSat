import Seq2SeqTransformer
import torch
from torch.utils.data import DataLoader
from loss.compute import compute_loss
def train_epoch(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    label_pad_id: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(
            model,
            batch,
            device=device,
            label_pad_id=label_pad_id,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1

    return total_loss / batch_count

