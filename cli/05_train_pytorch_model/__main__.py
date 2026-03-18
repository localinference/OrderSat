#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from Seq2SeqCollator.constructor import Seq2SeqCollator
from Seq2SeqTransformer.constructor import Seq2SeqTransformer
from TokenizedJsonlDataset.constructor import TokenizedJsonlDataset
from args.parse import parse_args
from artifacts.save import save_artifacts
from config.build import build_training_config
from device.build import build_device
from device.capabilities import get_device_capabilities
from epoch.train import train_epoch
from loss.evaluate import evaluate_loss
from match.compute import compute_exact_match
from parameters.count import count_parameters
from reporting.log import (
    log_adjusted_options,
    log_checkpoint_saved,
    log_early_stop,
    log_epoch_metrics,
    log_event,
    log_run_overview,
    log_stage_complete,
    log_stage_start,
    log_training_complete,
)
from seed.set import set_seed
from sequence.get_effective_lenght import get_effective_sequence_lengths
from stats.parse import parse_stats
from vocab.read_size import read_vocab_size


TOKENIZERS_ROOT = pathlib.Path("src/03_tokenizers")
DATASETS_ROOT = pathlib.Path("src/04_training_datasets")
PYTORCH_MODELS_ROOT = pathlib.Path("src/05_pytorch_models")

TOKENIZER_VOCAB_FILE = "tokenizer.vocab"
TRAINING_DATA_FILE = "train.jsonl"
VALIDATION_DATA_FILE = "validation.jsonl"
STATS_FILE = "stats.json"

SEED = 7
LOG_FREQUENCY = 1
BOS_ID = 1
EOS_ID = 2
LABEL_PAD_ID = -100
GRAD_CLIP = 1.0


@dataclass(frozen=True)
class RunPaths:
    language: str
    vocab_path: pathlib.Path
    training_dataset_path: pathlib.Path
    validation_dataset_path: pathlib.Path
    dataset_stats_path: pathlib.Path
    save_dir: pathlib.Path

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "vocab_path": str(self.vocab_path),
            "training_dataset_path": str(self.training_dataset_path),
            "validation_dataset_path": str(self.validation_dataset_path),
            "dataset_stats_path": str(self.dataset_stats_path),
            "save_dir": str(self.save_dir),
        }


def build_run_paths(language: str) -> RunPaths:
    return RunPaths(
        language=language,
        vocab_path=TOKENIZERS_ROOT / language / TOKENIZER_VOCAB_FILE,
        training_dataset_path=DATASETS_ROOT / language / TRAINING_DATA_FILE,
        validation_dataset_path=DATASETS_ROOT / language / VALIDATION_DATA_FILE,
        dataset_stats_path=DATASETS_ROOT / language / STATS_FILE,
        save_dir=PYTORCH_MODELS_ROOT / language,
    )


def build_loader(
    *,
    loader_name: str,
    dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    started_at = time.perf_counter()
    dataset_path = getattr(dataset, "file_path", None)
    log_stage_start(
        "loader.build",
        loader_name=loader_name,
        dataset_path=str(dataset_path) if dataset_path is not None else "<unknown>",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(**loader_kwargs)
    log_stage_complete(
        "loader.build",
        duration_seconds=time.perf_counter() - started_at,
        loader_name=loader_name,
        dataset_path=str(dataset_path) if dataset_path is not None else "<unknown>",
        batch_count=len(loader),
        batch_size=batch_size,
    )
    return loader


def main() -> None:
    run_started_at = time.perf_counter()
    args = parse_args()
    set_seed(SEED)

    run_paths = build_run_paths(args.language)
    log_event(
        "run.paths.resolved",
        language=run_paths.language,
        vocab_path=str(run_paths.vocab_path),
        training_dataset_path=str(run_paths.training_dataset_path),
        validation_dataset_path=str(run_paths.validation_dataset_path),
        dataset_stats_path=str(run_paths.dataset_stats_path),
        save_dir=str(run_paths.save_dir),
    )

    dataset_stats = parse_stats(run_paths.dataset_stats_path)
    device_capabilities = get_device_capabilities(args.device)
    training_config = build_training_config(
        dataset_stats=dataset_stats,
        device_capabilities=device_capabilities,
    )
    log_adjusted_options(
        adjusted_options=training_config.to_adjusted_options_dict(),
    )
    device = build_device(device_capabilities.resolved_device)

    vocab_size = read_vocab_size(run_paths.vocab_path)
    pad_id = vocab_size

    training_dataset = TokenizedJsonlDataset(
        run_paths.training_dataset_path,
        vocab_size,
    )
    validation_dataset = TokenizedJsonlDataset(
        run_paths.validation_dataset_path,
        vocab_size,
    )

    sequence_lengths = get_effective_sequence_lengths(
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        max_input_length=dataset_stats.input_lengths.max,
        max_label_length=dataset_stats.label_lengths.max,
    )

    log_stage_start(
        "collator.build",
        max_input_length=sequence_lengths.max_input_length,
        max_label_length=sequence_lengths.max_label_length,
        pad_id=pad_id,
    )
    collator_started_at = time.perf_counter()
    collator = Seq2SeqCollator(
        pad_id=pad_id,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        label_pad_id=LABEL_PAD_ID,
        max_input_length=sequence_lengths.max_input_length,
        max_label_length=sequence_lengths.max_label_length,
    )
    log_stage_complete(
        "collator.build",
        duration_seconds=time.perf_counter() - collator_started_at,
        pad_id=pad_id,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        label_pad_id=LABEL_PAD_ID,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = build_loader(
        loader_name="train",
        dataset=training_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
        generator=train_generator,
    )
    evaluation_train_loader = build_loader(
        loader_name="train_eval",
        dataset=training_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
    )
    validation_loader = build_loader(
        loader_name="validation",
        dataset=validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
    )

    log_stage_start(
        "model.build",
        language=run_paths.language,
        d_model=training_config.d_model,
        attention_heads=training_config.attention_heads,
        encoder_layers=training_config.encoder_layers,
        decoder_layers=training_config.decoder_layers,
    )
    model_started_at = time.perf_counter()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=training_config.d_model,
        num_heads=training_config.attention_heads,
        num_encoder_layers=training_config.encoder_layers,
        num_decoder_layers=training_config.decoder_layers,
        ffn_dim=training_config.ff_dimension,
        dropout=training_config.dropout,
        max_source_positions=sequence_lengths.max_source_positions,
        max_target_positions=sequence_lengths.max_target_positions,
    ).to(device)
    parameter_count = count_parameters(model)
    log_stage_complete(
        "model.build",
        duration_seconds=time.perf_counter() - model_started_at,
        language=run_paths.language,
        parameter_count=parameter_count,
        device=str(device),
    )

    log_stage_start(
        "optimizer.build",
        learning_rate=f"{training_config.learning_rate:.6f}",
        weight_decay=f"{training_config.weight_decay:.6f}",
    )
    optimizer_started_at = time.perf_counter()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    log_stage_complete(
        "optimizer.build",
        duration_seconds=time.perf_counter() - optimizer_started_at,
        optimizer="AdamW",
        learning_rate=f"{training_config.learning_rate:.6f}",
        weight_decay=f"{training_config.weight_decay:.6f}",
    )

    log_run_overview(
        language=run_paths.language,
        run_paths=run_paths.to_dict(),
        dataset_stats=dataset_stats.to_dict(),
        device_capabilities=device_capabilities.to_dict(),
        training_config=training_config.to_dict(),
        sequence_lengths=sequence_lengths.to_dict(),
        parameter_count=parameter_count,
    )

    run_metadata = {
        "static_config": {
            "seed": SEED,
            "log_frequency": LOG_FREQUENCY,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "label_pad_id": LABEL_PAD_ID,
            "grad_clip": GRAD_CLIP,
        },
        "run_paths": run_paths.to_dict(),
        "dataset_stats": dataset_stats.to_dict(),
        "device_capabilities": device_capabilities.to_dict(),
        "training_config": training_config.to_dict(),
        "sequence_lengths": sequence_lengths.to_dict(),
        "parameter_count": parameter_count,
    }

    max_generation_length = sequence_lengths.max_target_positions
    best_metrics: dict | None = None
    best_validation_loss: float | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, training_config.epochs + 1):
        log_event(
            "epoch.start",
            language=run_paths.language,
            epoch=epoch,
            total_epochs=training_config.epochs,
        )
        training_result = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            label_pad_id=LABEL_PAD_ID,
            grad_clip=GRAD_CLIP,
            accumulation_steps=training_config.accumulation_steps,
        )
        validation_result = evaluate_loss(
            model,
            validation_loader,
            device=device,
            label_pad_id=LABEL_PAD_ID,
        )

        should_run_exact_match = (
            training_config.exact_match_frequency > 0
            and (
                epoch % training_config.exact_match_frequency == 0
                or epoch == training_config.epochs
            )
        )

        train_exact_match_result = None
        validation_exact_match_result = None
        if should_run_exact_match:
            train_exact_match_result = compute_exact_match(
                model,
                evaluation_train_loader,
                split_name="train",
                device=device,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                max_generation_length=max_generation_length,
            )
            validation_exact_match_result = compute_exact_match(
                model,
                validation_loader,
                split_name="validation",
                device=device,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                max_generation_length=max_generation_length,
            )
        else:
            log_event(
                "validation.exact_match.skipped",
                epoch=epoch,
                exact_match_frequency=training_config.exact_match_frequency,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": training_result.average_loss,
            "validation_loss": validation_result.average_loss,
            "train_exact_match": (
                train_exact_match_result.exact_match
                if train_exact_match_result is not None
                else None
            ),
            "validation_exact_match": (
                validation_exact_match_result.exact_match
                if validation_exact_match_result is not None
                else None
            ),
            "train_duration_seconds": training_result.duration_seconds,
            "validation_duration_seconds": validation_result.duration_seconds,
            "exact_match_duration_seconds": (
                (
                    train_exact_match_result.duration_seconds
                    + validation_exact_match_result.duration_seconds
                )
                if train_exact_match_result is not None
                and validation_exact_match_result is not None
                else None
            ),
            "optimizer_steps": training_result.optimizer_steps,
            "train_token_count": training_result.token_count,
            "validation_token_count": validation_result.token_count,
        }
        history.append(epoch_metrics)

        improved_validation = (
            best_validation_loss is None
            or validation_result.average_loss
            < best_validation_loss - training_config.early_stopping_min_delta
        )

        if improved_validation:
            best_validation_loss = validation_result.average_loss
            best_metrics = epoch_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_run_metadata = {
            **run_metadata,
            "runtime_state": {
                "latest_epoch_completed": epoch,
                "best_validation_loss": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "latest_metrics": epoch_metrics,
                "best_metrics": best_metrics,
            },
        }
        save_artifacts(
            save_dir=run_paths.save_dir,
            best_metrics=best_metrics,
            history=history,
            metadata=epoch_run_metadata,
            model=model,
            optimizer=optimizer,
            save_checkpoint=improved_validation,
        )
        if improved_validation:
            log_checkpoint_saved(
                epoch=epoch,
                save_dir=run_paths.save_dir,
                validation_loss=validation_result.average_loss,
            )

        if epoch == 1 or epoch % LOG_FREQUENCY == 0 or epoch == training_config.epochs:
            log_epoch_metrics(
                epoch=epoch,
                total_epochs=training_config.epochs,
                train_loss=training_result.average_loss,
                validation_loss=validation_result.average_loss,
                train_duration_seconds=training_result.duration_seconds,
                validation_duration_seconds=validation_result.duration_seconds,
                optimizer_steps=training_result.optimizer_steps,
                best_validation_loss=(
                    best_validation_loss
                    if best_validation_loss is not None
                    else validation_result.average_loss
                ),
                epochs_without_improvement=epochs_without_improvement,
                early_stopping_patience=training_config.early_stopping_patience,
                train_exact_match=(
                    train_exact_match_result.exact_match
                    if train_exact_match_result is not None
                    else None
                ),
                validation_exact_match=(
                    validation_exact_match_result.exact_match
                    if validation_exact_match_result is not None
                    else None
                ),
                exact_match_duration_seconds=epoch_metrics["exact_match_duration_seconds"],
            )

        if (
            train_exact_match_result is not None
            and train_exact_match_result.exact_match >= 1.0
        ):
            log_event(
                "training.memorization_signal",
                epoch=epoch,
                train_exact_match=f"{train_exact_match_result.exact_match:.4f}",
                validation_exact_match=(
                    f"{validation_exact_match_result.exact_match:.4f}"
                    if validation_exact_match_result is not None
                    else "<not_run>"
                ),
            )

        if epochs_without_improvement >= training_config.early_stopping_patience:
            log_early_stop(
                reason=(
                    "validation loss did not improve within patience "
                    f"({training_config.early_stopping_patience} epochs)"
                ),
                epoch=epoch,
            )
            break

    log_training_complete(
        best_metrics=best_metrics,
        save_dir=run_paths.save_dir,
        duration_seconds=time.perf_counter() - run_started_at,
    )


if __name__ == "__main__":
    main()
