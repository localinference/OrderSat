import torch
import Seq2SeqTransformer
from torch.utils.data import DataLoader
from greedy.generate import greedy_generate

def compute_exact_match(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    *,
    device: str,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
) -> float:
    matches = 0
    total = 0

    with torch.no_grad():
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
                total += 1
                if predicted == target:
                    matches += 1

    if total == 0:
        return 0.0

    return matches / total