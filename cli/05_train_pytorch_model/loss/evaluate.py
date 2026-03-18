import Seq2SeqTransformer
import torch
from torch.utils.data import DataLoader
from loss.compute import compute_loss
def evaluate_loss(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    *,
    device: str,
    label_pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            loss = compute_loss(
                model,
                batch,
                device=device,
                label_pad_id=label_pad_id,
            )
            total_loss += float(loss.item())
            batch_count += 1

    if batch_count == 0:
        return 0.0

    return total_loss / batch_count