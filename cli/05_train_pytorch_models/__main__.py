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
from batching.sampler import TokenBudgetBatchSampler
from checkpoint.load import build_model_signature, load_checkpoint
from config.build import build_training_config
from device.build import build_device
from device.capabilities import get_device_capabilities
from epoch.train import train_epoch
from formats.discover import discover_available_formats
from loss.evaluate import evaluate_loss
from match.compute import compute_exact_match
from orchestration.run_formats import run_formats_in_parallel
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
from selection.rank import build_checkpoint_score, is_better_checkpoint
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
    format: str
    vocab_path: pathlib.Path
    training_dataset_path: pathlib.Path
    validation_dataset_path: pathlib.Path
    dataset_stats_path: pathlib.Path
    save_dir: pathlib.Path

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "format": self.format,
            "vocab_path": str(self.vocab_path),
            "training_dataset_path": str(self.training_dataset_path),
            "validation_dataset_path": str(self.validation_dataset_path),
            "dataset_stats_path": str(self.dataset_stats_path),
            "save_dir": str(self.save_dir),
        }


def build_run_paths(language: str, format_name: str) -> RunPaths:
    return RunPaths(
        language=language,
        format=format_name,
        vocab_path=TOKENIZERS_ROOT / language / format_name / TOKENIZER_VOCAB_FILE,
        training_dataset_path=DATASETS_ROOT / language / format_name / TRAINING_DATA_FILE,
        validation_dataset_path=DATASETS_ROOT / language / format_name / VALIDATION_DATA_FILE,
        dataset_stats_path=DATASETS_ROOT / language / format_name / STATS_FILE,
        save_dir=PYTORCH_MODELS_ROOT / language / format_name,
    )


def build_loader(
    *,
    loader_name: str,
    dataset,
    target_tokens_per_batch: int,
    max_batch_size: int,
    shuffle: bool,
    collate_fn,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
) -> DataLoader:
    started_at = time.perf_counter()
    dataset_path = getattr(dataset, "file_path", None)
    batch_sampler = TokenBudgetBatchSampler(
        dataset,
        target_tokens_per_batch=target_tokens_per_batch,
        max_batch_size=max_batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    batch_plan = batch_sampler.describe_current_plan()
    log_stage_start(
        "loader.build",
        loader_name=loader_name,
        dataset_path=str(dataset_path) if dataset_path is not None else "<unknown>",
        batch_strategy="token_budget",
        shuffle=shuffle,
        target_tokens_per_batch=target_tokens_per_batch,
        max_batch_size=max_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(**loader_kwargs)
    log_stage_complete(
        "loader.build",
        duration_seconds=time.perf_counter() - started_at,
        loader_name=loader_name,
        dataset_path=str(dataset_path) if dataset_path is not None else "<unknown>",
        batch_count=batch_plan.batch_count,
        average_batch_size=f"{batch_plan.average_batch_size:.2f}",
        max_batch_size_observed=batch_plan.max_batch_size_observed,
        average_batch_tokens=f"{batch_plan.average_batch_tokens:.2f}",
        max_batch_tokens_observed=batch_plan.max_batch_tokens_observed,
    )
    return loader


def should_run_validation_exact_match(
    *,
    epoch: int,
    total_epochs: int,
    frequency: int,
) -> bool:
    if frequency <= 0:
        return epoch == total_epochs
    return (
        epoch == 1
        or epoch == total_epochs
        or epoch % frequency == 0
    )


def checkpoint_score_from_metrics(metrics: dict | None):
    if not isinstance(metrics, dict):
        return None

    epoch = metrics.get("epoch")
    validation_loss = metrics.get("validation_loss")
    validation_exact_match = metrics.get("validation_exact_match")

    if not isinstance(epoch, int):
        return None
    if not isinstance(validation_loss, (int, float)):
        return None
    if validation_exact_match is not None and not isinstance(
        validation_exact_match,
        (int, float),
    ):
        validation_exact_match = None

    return build_checkpoint_score(
        epoch=epoch,
        validation_loss=float(validation_loss),
        validation_exact_match=(
            float(validation_exact_match)
            if validation_exact_match is not None
            else None
        ),
    )


def train_format(
    *,
    language: str,
    format_name: str,
    requested_device: str,
    checkpoint_mode: str,
) -> None:
    run_started_at = time.perf_counter()
    set_seed(SEED)

    run_paths = build_run_paths(language, format_name)
    log_event(
        "run.paths.resolved",
        language=run_paths.language,
        format=run_paths.format,
        vocab_path=str(run_paths.vocab_path),
        training_dataset_path=str(run_paths.training_dataset_path),
        validation_dataset_path=str(run_paths.validation_dataset_path),
        dataset_stats_path=str(run_paths.dataset_stats_path),
        save_dir=str(run_paths.save_dir),
    )

    dataset_stats = parse_stats(run_paths.dataset_stats_path)
    device_capabilities = get_device_capabilities(requested_device)
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

    train_loader = build_loader(
        loader_name="train",
        dataset=training_dataset,
        target_tokens_per_batch=training_config.target_tokens_per_batch,
        max_batch_size=training_config.max_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
        seed=SEED,
    )
    evaluation_train_loader = build_loader(
        loader_name="train_audit",
        dataset=training_dataset,
        target_tokens_per_batch=training_config.target_tokens_per_batch,
        max_batch_size=training_config.max_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
    )
    validation_loader = build_loader(
        loader_name="validation",
        dataset=validation_dataset,
        target_tokens_per_batch=training_config.target_tokens_per_batch,
        max_batch_size=training_config.max_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
    )

    log_stage_start(
        "model.build",
        language=run_paths.language,
        format=run_paths.format,
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
        format=run_paths.format,
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

    model_signature = build_model_signature(
        language=run_paths.language,
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        label_pad_id=LABEL_PAD_ID,
        training_config=training_config,
        sequence_lengths=sequence_lengths,
    )
    checkpoint_path = run_paths.save_dir / "best.pt"
    checkpoint_result = load_checkpoint(
        checkpoint_mode=checkpoint_mode,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        model_signature=model_signature,
    )

    log_run_overview(
        language=run_paths.language,
        format_name=run_paths.format,
        checkpoint_mode=checkpoint_mode,
        checkpoint_applied_mode=checkpoint_result.applied_mode,
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
            "pad_id": pad_id,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "label_pad_id": LABEL_PAD_ID,
            "grad_clip": GRAD_CLIP,
        },
        "checkpoint_policy": {
            "requested_mode": checkpoint_mode,
            "applied_mode": checkpoint_result.applied_mode,
            "checkpoint_path": str(checkpoint_path),
        },
        "model_signature": model_signature,
        "run_paths": run_paths.to_dict(),
        "dataset_stats": dataset_stats.to_dict(),
        "device_capabilities": device_capabilities.to_dict(),
        "training_config": training_config.to_dict(),
        "sequence_lengths": sequence_lengths.to_dict(),
        "parameter_count": parameter_count,
    }

    best_metrics: dict | None = checkpoint_result.best_metrics
    best_validation_loss: float | None = checkpoint_result.best_validation_loss
    best_checkpoint_score = checkpoint_score_from_metrics(best_metrics)
    epochs_without_improvement = checkpoint_result.epochs_without_improvement
    history: list[dict] = list(checkpoint_result.history)
    latest_epoch_metrics = history[-1] if history else None
    last_completed_epoch = (
        latest_epoch_metrics["epoch"]
        if isinstance(latest_epoch_metrics, dict)
        and isinstance(latest_epoch_metrics.get("epoch"), int)
        else 0
    )

    for epoch in range(checkpoint_result.start_epoch, training_config.epochs + 1):
        log_event(
            "epoch.start",
            language=run_paths.language,
            format=run_paths.format,
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

        should_run_validation_exact = should_run_validation_exact_match(
            epoch=epoch,
            total_epochs=training_config.epochs,
            frequency=training_config.validation_exact_match_frequency,
        )

        validation_exact_match_result = None
        if should_run_validation_exact:
            validation_exact_match_result = compute_exact_match(
                model,
                validation_loader,
                split_name="validation",
                device=device,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
            )
        else:
            log_event(
                "validation.exact_match.skipped",
                epoch=epoch,
                validation_exact_match_frequency=(
                    training_config.validation_exact_match_frequency
                ),
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": training_result.average_loss,
            "validation_loss": validation_result.average_loss,
            "validation_exact_match": (
                validation_exact_match_result.exact_match
                if validation_exact_match_result is not None
                else None
            ),
            "train_duration_seconds": training_result.duration_seconds,
            "validation_duration_seconds": validation_result.duration_seconds,
            "exact_match_duration_seconds": (
                validation_exact_match_result.duration_seconds
                if validation_exact_match_result is not None
                else None
            ),
            "validation_exact_match_ran": validation_exact_match_result is not None,
            "checkpoint_mode": checkpoint_result.applied_mode,
            "optimizer_steps": training_result.optimizer_steps,
            "train_token_count": training_result.token_count,
            "validation_token_count": validation_result.token_count,
        }
        history.append(epoch_metrics)
        latest_epoch_metrics = epoch_metrics
        last_completed_epoch = epoch

        improved_validation_loss = (
            best_validation_loss is None
            or validation_result.average_loss
            < best_validation_loss - training_config.early_stopping_min_delta
        )

        if improved_validation_loss:
            best_validation_loss = validation_result.average_loss

        candidate_checkpoint_score = checkpoint_score_from_metrics(epoch_metrics)
        improved_checkpoint = False
        if candidate_checkpoint_score is not None:
            improved_checkpoint = is_better_checkpoint(
                candidate_checkpoint_score,
                best_checkpoint_score,
            )

        if improved_checkpoint:
            best_checkpoint_score = candidate_checkpoint_score
            best_metrics = epoch_metrics

        should_track_patience_from_checkpoint_metric = (
            training_config.validation_exact_match_frequency > 0
        )

        if should_track_patience_from_checkpoint_metric:
            if validation_exact_match_result is not None:
                if improved_checkpoint:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
        else:
            if improved_validation_loss:
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
            save_checkpoint=improved_checkpoint,
        )
        if improved_checkpoint:
            log_checkpoint_saved(
                epoch=epoch,
                save_dir=run_paths.save_dir,
                validation_loss=validation_result.average_loss,
                validation_exact_match=(
                    validation_exact_match_result.exact_match
                    if validation_exact_match_result is not None
                    else None
                ),
            )

        if epoch == checkpoint_result.start_epoch or epoch % LOG_FREQUENCY == 0 or epoch == training_config.epochs:
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
                train_exact_match=None,
                validation_exact_match=(
                    validation_exact_match_result.exact_match
                    if validation_exact_match_result is not None
                    else None
                ),
                exact_match_duration_seconds=epoch_metrics["exact_match_duration_seconds"],
            )

        if epochs_without_improvement >= training_config.early_stopping_patience:
            log_early_stop(
                reason=(
                    "checkpoint selection metric did not improve within patience "
                    f"({training_config.early_stopping_patience} evaluation windows)"
                    if should_track_patience_from_checkpoint_metric
                    else
                    "validation loss did not improve within patience "
                    f"({training_config.early_stopping_patience} epochs)"
                ),
                epoch=epoch,
            )
            break

    final_train_exact_match_result = None
    if (
        training_config.run_train_exact_match_at_end
        and checkpoint_path.exists()
        and best_metrics is not None
    ):
        audit_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(audit_checkpoint["model_state_dict"], strict=True)
        final_train_exact_match_result = compute_exact_match(
            model,
            evaluation_train_loader,
            split_name="train_final_audit",
            device=device,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
        )

        if (
            final_train_exact_match_result.exact_match >= 1.0
            and isinstance(best_metrics.get("validation_exact_match"), (int, float))
            and float(best_metrics["validation_exact_match"]) < 1.0
        ):
            log_event(
                "training.memorization_signal",
                epoch=best_metrics.get("epoch"),
                format=run_paths.format,
                train_exact_match=f"{final_train_exact_match_result.exact_match:.4f}",
                validation_exact_match=(
                    f"{float(best_metrics['validation_exact_match']):.4f}"
                ),
            )

    final_run_metadata = {
        **run_metadata,
        "runtime_state": {
            "latest_epoch_completed": last_completed_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "latest_metrics": latest_epoch_metrics,
            "best_metrics": best_metrics,
        },
        "final_audit": {
            "train_exact_match": (
                final_train_exact_match_result.exact_match
                if final_train_exact_match_result is not None
                else None
            ),
            "train_exact_match_duration_seconds": (
                final_train_exact_match_result.duration_seconds
                if final_train_exact_match_result is not None
                else None
            ),
        },
    }
    save_artifacts(
        save_dir=run_paths.save_dir,
        best_metrics=best_metrics,
        history=history,
        metadata=final_run_metadata,
        model=model,
        optimizer=optimizer,
        save_checkpoint=False,
    )

    log_training_complete(
        best_metrics=best_metrics,
        save_dir=run_paths.save_dir,
        duration_seconds=time.perf_counter() - run_started_at,
    )


def main() -> None:
    args = parse_args()

    if args.format == "all":
        formats = discover_available_formats(
            language=args.language,
            tokenizers_root=TOKENIZERS_ROOT,
            datasets_root=DATASETS_ROOT,
        )
        log_event(
            "formats.discovered",
            language=args.language,
            formats=", ".join(formats),
            count=len(formats),
            parallel=not args.sequential_formats,
        )

        if len(formats) == 1:
            train_format(
                language=args.language,
                format_name=formats[0],
                requested_device=args.device,
                checkpoint_mode=args.checkpoint_mode,
            )
            return

        if args.sequential_formats:
            for format_name in formats:
                log_event(
                    "format.dispatch",
                    language=args.language,
                    format=format_name,
                    mode="sequential",
                )
                train_format(
                    language=args.language,
                    format_name=format_name,
                    requested_device=args.device,
                    checkpoint_mode=args.checkpoint_mode,
                )
            return

        run_formats_in_parallel(
            entrypoint_path=pathlib.Path(__file__).resolve(),
            language=args.language,
            formats=formats,
            requested_device=args.device,
            checkpoint_mode=args.checkpoint_mode,
        )
        return

    train_format(
        language=args.language,
        format_name=args.format,
        requested_device=args.device,
        checkpoint_mode=args.checkpoint_mode,
    )


if __name__ == "__main__":
    main()
