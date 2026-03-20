from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import torch

from reporting.log import log_stage_complete, log_stage_start


CHECKPOINT_MODES = ("auto", "fresh", "warm_start", "resume")


@dataclass(frozen=True)
class CheckpointLoadResult:
    applied_mode: str
    checkpoint_path: pathlib.Path
    loaded: bool
    reason: str
    start_epoch: int
    history: list[dict]
    best_metrics: dict | None
    best_validation_loss: float | None
    epochs_without_improvement: int


def build_model_signature(
    *,
    language: str,
    vocab_size: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    label_pad_id: int,
    training_config,
    sequence_lengths,
) -> dict:
    return {
        "language": language,
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "label_pad_id": label_pad_id,
        "d_model": training_config.d_model,
        "attention_heads": training_config.attention_heads,
        "encoder_layers": training_config.encoder_layers,
        "decoder_layers": training_config.decoder_layers,
        "ff_dimension": training_config.ff_dimension,
        "max_source_positions": sequence_lengths.max_source_positions,
        "max_target_positions": sequence_lengths.max_target_positions,
    }


def _extract_checkpoint_signature(payload: dict) -> dict | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    explicit_signature = metadata.get("model_signature")
    if isinstance(explicit_signature, dict):
        return explicit_signature

    static_config = metadata.get("static_config")
    training_config = metadata.get("training_config")
    sequence_lengths = metadata.get("sequence_lengths")

    if not isinstance(static_config, dict):
        return None
    if not isinstance(training_config, dict):
        return None
    if not isinstance(sequence_lengths, dict):
        return None

    language = metadata.get("run_paths", {}).get("language")
    vocab_size = training_config.get("vocab_size")
    if vocab_size is None:
        output_projection = payload.get("model_state_dict", {}).get(
            "output_projection.weight"
        )
        if isinstance(output_projection, torch.Tensor):
            vocab_size = int(output_projection.size(0))

    pad_id = static_config.get("pad_id")
    if pad_id is None and vocab_size is not None:
        pad_id = int(vocab_size)

    return {
        "language": language,
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "bos_id": static_config.get("bos_id"),
        "eos_id": static_config.get("eos_id"),
        "label_pad_id": static_config.get("label_pad_id"),
        "d_model": training_config.get("d_model"),
        "attention_heads": training_config.get("attention_heads"),
        "encoder_layers": training_config.get("encoder_layers"),
        "decoder_layers": training_config.get("decoder_layers"),
        "ff_dimension": training_config.get("ff_dimension"),
        "max_source_positions": sequence_lengths.get("max_source_positions"),
        "max_target_positions": sequence_lengths.get("max_target_positions"),
    }


def _extract_legacy_checkpoint_signature(
    *,
    payload: dict,
    current_signature: dict,
) -> dict | None:
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return None

    token_embedding = state_dict.get("token_embedding.weight")
    output_projection = state_dict.get("output_projection.weight")
    source_position_embedding = state_dict.get("source_position_embedding.weight")
    target_position_embedding = state_dict.get("target_position_embedding.weight")
    encoder_ff = state_dict.get("encoder.layers.0.linear1.weight")

    required_tensors = (
        token_embedding,
        output_projection,
        source_position_embedding,
        target_position_embedding,
        encoder_ff,
    )
    if not all(isinstance(value, torch.Tensor) for value in required_tensors):
        return None

    encoder_layer_indices: set[int] = set()
    decoder_layer_indices: set[int] = set()
    for key in state_dict:
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                encoder_layer_indices.add(int(parts[2]))
        if key.startswith("decoder.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                decoder_layer_indices.add(int(parts[2]))

    if not encoder_layer_indices or not decoder_layer_indices:
        return None

    return {
        "language": current_signature["language"],
        "vocab_size": int(output_projection.size(0)),
        "pad_id": int(token_embedding.size(0)) - 1,
        "bos_id": current_signature["bos_id"],
        "eos_id": current_signature["eos_id"],
        "label_pad_id": current_signature["label_pad_id"],
        "d_model": int(token_embedding.size(1)),
        "attention_heads": current_signature["attention_heads"],
        "encoder_layers": len(encoder_layer_indices),
        "decoder_layers": len(decoder_layer_indices),
        "ff_dimension": int(encoder_ff.size(0)),
        "max_source_positions": int(source_position_embedding.size(0)),
        "max_target_positions": int(target_position_embedding.size(0)),
    }


def _get_signature_mismatches(
    *,
    current_signature: dict,
    checkpoint_signature: dict | None,
) -> list[str]:
    if checkpoint_signature is None:
        return ["checkpoint metadata missing compatible model signature"]

    mismatches: list[str] = []
    for key, current_value in current_signature.items():
        checkpoint_value = checkpoint_signature.get(key)
        if checkpoint_value != current_value:
            mismatches.append(
                f"{key}: current={current_value!r}, checkpoint={checkpoint_value!r}"
            )
    return mismatches


def _extract_runtime_state(payload: dict) -> tuple[list[dict], dict | None, float | None, int, int]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit("Checkpoint metadata is missing.")

    runtime_state = metadata.get("runtime_state")
    if not isinstance(runtime_state, dict):
        raise SystemExit("Checkpoint runtime_state is missing.")

    latest_epoch_completed = runtime_state.get("latest_epoch_completed")
    if not isinstance(latest_epoch_completed, int) or latest_epoch_completed < 0:
        raise SystemExit("Checkpoint latest_epoch_completed is invalid.")

    history = payload.get("history")
    if not isinstance(history, list):
        raise SystemExit("Checkpoint history is missing.")

    best_metrics = runtime_state.get("best_metrics", payload.get("metrics"))
    best_validation_loss = runtime_state.get("best_validation_loss")
    if best_validation_loss is not None and not isinstance(
        best_validation_loss,
        (int, float),
    ):
        raise SystemExit("Checkpoint best_validation_loss is invalid.")

    epochs_without_improvement = runtime_state.get("epochs_without_improvement")
    if not isinstance(epochs_without_improvement, int) or epochs_without_improvement < 0:
        raise SystemExit("Checkpoint epochs_without_improvement is invalid.")

    return (
        history,
        best_metrics if isinstance(best_metrics, dict) else None,
        float(best_validation_loss) if best_validation_loss is not None else None,
        epochs_without_improvement,
        latest_epoch_completed + 1,
    )


def _extract_best_metrics(payload: dict) -> dict | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    runtime_state = metadata.get("runtime_state")
    if not isinstance(runtime_state, dict):
        return None

    best_metrics = runtime_state.get("best_metrics")
    if isinstance(best_metrics, dict):
        return best_metrics
    return None


def load_checkpoint(
    *,
    checkpoint_mode: str,
    checkpoint_path: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_signature: dict,
) -> CheckpointLoadResult:
    started_at = time.perf_counter()
    log_stage_start(
        "checkpoint.load",
        checkpoint_mode=checkpoint_mode,
        checkpoint_path=str(checkpoint_path),
    )

    if checkpoint_mode == "fresh":
        result = CheckpointLoadResult(
            applied_mode="fresh",
            checkpoint_path=checkpoint_path,
            loaded=False,
            reason="fresh mode requested",
            start_epoch=1,
            history=[],
            best_metrics=None,
            best_validation_loss=None,
            epochs_without_improvement=0,
        )
        log_stage_complete(
            "checkpoint.load",
            duration_seconds=time.perf_counter() - started_at,
            applied_mode=result.applied_mode,
            loaded=result.loaded,
            reason=result.reason,
            start_epoch=result.start_epoch,
        )
        return result

    if not checkpoint_path.exists():
        if checkpoint_mode in {"warm_start", "resume"}:
            raise SystemExit(f"Checkpoint file does not exist: {checkpoint_path}")

        result = CheckpointLoadResult(
            applied_mode="fresh",
            checkpoint_path=checkpoint_path,
            loaded=False,
            reason="checkpoint missing",
            start_epoch=1,
            history=[],
            best_metrics=None,
            best_validation_loss=None,
            epochs_without_improvement=0,
        )
        log_stage_complete(
            "checkpoint.load",
            duration_seconds=time.perf_counter() - started_at,
            applied_mode=result.applied_mode,
            loaded=result.loaded,
            reason=result.reason,
            start_epoch=result.start_epoch,
        )
        return result

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid checkpoint payload in {checkpoint_path}")

    checkpoint_signature = _extract_checkpoint_signature(payload)
    used_legacy_signature = False
    if checkpoint_signature is None:
        checkpoint_signature = _extract_legacy_checkpoint_signature(
            payload=payload,
            current_signature=model_signature,
        )
        used_legacy_signature = checkpoint_signature is not None
    mismatches = _get_signature_mismatches(
        current_signature=model_signature,
        checkpoint_signature=checkpoint_signature,
    )
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        if checkpoint_mode in {"warm_start", "resume"}:
            raise SystemExit(
                f"Incompatible checkpoint at {checkpoint_path}: {mismatch_text}"
            )

        result = CheckpointLoadResult(
            applied_mode="fresh",
            checkpoint_path=checkpoint_path,
            loaded=False,
            reason=f"incompatible checkpoint: {mismatch_text}",
            start_epoch=1,
            history=[],
            best_metrics=None,
            best_validation_loss=None,
            epochs_without_improvement=0,
        )
        log_stage_complete(
            "checkpoint.load",
            duration_seconds=time.perf_counter() - started_at,
            applied_mode=result.applied_mode,
            loaded=result.loaded,
            reason=result.reason,
            start_epoch=result.start_epoch,
        )
        return result

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Checkpoint is missing model_state_dict: {checkpoint_path}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise SystemExit(
            f"Failed to load checkpoint model weights from {checkpoint_path}: {error}"
        ) from error

    if checkpoint_mode in {"auto", "warm_start"}:
        best_metrics = _extract_best_metrics(payload)
        result = CheckpointLoadResult(
            applied_mode="warm_start",
            checkpoint_path=checkpoint_path,
            loaded=True,
            reason=(
                "loaded compatible legacy checkpoint weights"
                if used_legacy_signature
                else "loaded compatible best checkpoint weights"
            ),
            start_epoch=1,
            history=[],
            best_metrics=best_metrics,
            best_validation_loss=None,
            epochs_without_improvement=0,
        )
        log_stage_complete(
            "checkpoint.load",
            duration_seconds=time.perf_counter() - started_at,
            applied_mode=result.applied_mode,
            loaded=result.loaded,
            reason=result.reason,
            start_epoch=result.start_epoch,
        )
        return result

    optimizer_state_dict = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        raise SystemExit(
            f"Checkpoint is missing optimizer_state_dict for resume: {checkpoint_path}"
        )

    try:
        optimizer.load_state_dict(optimizer_state_dict)
    except ValueError as error:
        raise SystemExit(
            f"Failed to load checkpoint optimizer state from {checkpoint_path}: {error}"
        ) from error

    (
        history,
        best_metrics,
        best_validation_loss,
        epochs_without_improvement,
        start_epoch,
    ) = _extract_runtime_state(payload)

    result = CheckpointLoadResult(
        applied_mode="resume",
        checkpoint_path=checkpoint_path,
        loaded=True,
        reason=(
            "resumed compatible legacy checkpoint state"
            if used_legacy_signature
            else "resumed compatible checkpoint state"
        ),
        start_epoch=start_epoch,
        history=history,
        best_metrics=best_metrics,
        best_validation_loss=best_validation_loss,
        epochs_without_improvement=epochs_without_improvement,
    )
    log_stage_complete(
        "checkpoint.load",
        duration_seconds=time.perf_counter() - started_at,
        applied_mode=result.applied_mode,
        loaded=result.loaded,
        reason=result.reason,
        start_epoch=result.start_epoch,
        history_length=len(result.history),
        epochs_without_improvement=result.epochs_without_improvement,
    )
    return result
