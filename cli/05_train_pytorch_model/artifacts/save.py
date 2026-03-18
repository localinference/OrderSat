import Seq2SeqTransformer
import torch
import pathlib
import json

def save_artifacts(
    *,
    save_dir: pathlib.Path,
    metrics: dict,
    model: Seq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.json"
    checkpoint_path = save_dir / "best.pt"

    metrics_path.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )