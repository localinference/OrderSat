from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(duration_seconds: float) -> str:
    return f"{duration_seconds:.2f}s"


def _print_block(title: str, values: dict) -> None:
    print(f"[{_timestamp()}] {title}", flush=True)
    for key, value in values.items():
        print(f"  {key}: {value}", flush=True)
    print(flush=True)


def log_event(event: str, **values) -> None:
    _print_block(event, values)


def log_stage_start(stage: str, **values) -> None:
    _print_block(f"{stage}.start", values)


def log_stage_complete(stage: str, duration_seconds: float | None = None, **values) -> None:
    if duration_seconds is not None:
        values["duration"] = format_seconds(duration_seconds)
    _print_block(f"{stage}.complete", values)


def log_json_block(title: str, payload: dict) -> None:
    print(f"[{_timestamp()}] {title}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print(flush=True)


def log_adjusted_options(*, adjusted_options: dict) -> None:
    log_json_block("config.adjusted_options", adjusted_options)


def log_run_overview(
    *,
    language: str,
    run_paths: dict,
    dataset_stats: dict,
    device_capabilities: dict,
    training_config: dict,
    sequence_lengths: dict,
    parameter_count: int,
) -> None:
    _print_block(
        "run.start",
        {
            "language": language,
            "train_samples": dataset_stats["train_count"],
            "validation_samples": dataset_stats["validation_count"],
            "sample_count": dataset_stats["sample_count"],
            "resolved_device": device_capabilities["resolved_device"],
            "accelerator_name": device_capabilities["accelerator_name"],
            "data_scale": f"{training_config['data_scale']:.4f}",
            "device_scale": f"{training_config['device_scale']:.4f}",
            "capacity_scale": f"{training_config['capacity_scale']:.4f}",
            "d_model": training_config["d_model"],
            "attention_heads": training_config["attention_heads"],
            "encoder_layers": training_config["encoder_layers"],
            "decoder_layers": training_config["decoder_layers"],
            "ff_dimension": training_config["ff_dimension"],
            "dropout": f"{training_config['dropout']:.2f}",
            "batch_size": training_config["batch_size"],
            "accumulation_steps": training_config["accumulation_steps"],
            "effective_batch_size": training_config["achieved_effective_batch_size"],
            "num_workers": training_config["num_workers"],
            "pin_memory": training_config["pin_memory"],
            "learning_rate": f"{training_config['learning_rate']:.6f}",
            "weight_decay": f"{training_config['weight_decay']:.6f}",
            "epochs": training_config["epochs"],
            "patience": training_config["early_stopping_patience"],
            "exact_match_frequency": training_config["exact_match_frequency"],
            "max_input_length": sequence_lengths["max_input_length"],
            "max_label_length": sequence_lengths["max_label_length"],
            "parameter_count": parameter_count,
            "save_dir": run_paths["save_dir"],
        },
    )
    _print_block(
        "run.paths",
        {
            "tokenizer_vocab_path": run_paths["vocab_path"],
            "train_dataset_path": run_paths["training_dataset_path"],
            "validation_dataset_path": run_paths["validation_dataset_path"],
            "dataset_stats_path": run_paths["dataset_stats_path"],
        },
    )


def log_epoch_metrics(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    validation_loss: float,
    train_duration_seconds: float,
    validation_duration_seconds: float,
    optimizer_steps: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    early_stopping_patience: int,
    train_exact_match: float | None,
    validation_exact_match: float | None,
    exact_match_duration_seconds: float | None,
) -> None:
    values = {
        "epoch": f"{epoch}/{total_epochs}",
        "train_loss": f"{train_loss:.6f}",
        "validation_loss": f"{validation_loss:.6f}",
        "best_validation_loss": f"{best_validation_loss:.6f}",
        "optimizer_steps": optimizer_steps,
        "train_duration": format_seconds(train_duration_seconds),
        "validation_duration": format_seconds(validation_duration_seconds),
        "patience_state": f"{epochs_without_improvement}/{early_stopping_patience}",
    }
    if train_exact_match is not None and validation_exact_match is not None:
        values["train_exact_match"] = f"{train_exact_match:.4f}"
        values["validation_exact_match"] = f"{validation_exact_match:.4f}"
        values["exact_match_duration"] = format_seconds(
            exact_match_duration_seconds or 0.0
        )

    _print_block("epoch.complete", values)


def log_checkpoint_saved(*, epoch: int, save_dir: Path, validation_loss: float) -> None:
    _print_block(
        "checkpoint.saved",
        {
            "epoch": epoch,
            "validation_loss": f"{validation_loss:.6f}",
            "save_dir": str(save_dir),
        },
    )


def log_early_stop(*, reason: str, epoch: int) -> None:
    _print_block(
        "training.stop",
        {
            "epoch": epoch,
            "reason": reason,
        },
    )


def log_training_complete(
    *,
    best_metrics: dict | None,
    save_dir: Path,
    duration_seconds: float,
) -> None:
    values = {
        "duration": format_seconds(duration_seconds),
        "save_dir": str(save_dir),
        "best_metrics": json.dumps(best_metrics, indent=2) if best_metrics else "null",
    }
    _print_block("training.complete", values)
