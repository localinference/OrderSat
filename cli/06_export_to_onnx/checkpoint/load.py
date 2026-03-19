from __future__ import annotations

import pathlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExportModelConfig:
    language: str
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    d_model: int
    attention_heads: int
    encoder_layers: int
    decoder_layers: int
    ff_dimension: int
    dropout: float
    max_source_positions: int
    max_target_positions: int


def load_export_checkpoint(
    checkpoint_path: pathlib.Path,
) -> tuple[dict, ExportModelConfig]:
    if not checkpoint_path.exists():
        raise SystemExit(f"PyTorch checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"PyTorch checkpoint path is not a file: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint payload is not a dictionary: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise SystemExit(
            f"Checkpoint is missing model_state_dict: {checkpoint_path}"
        )

    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"Checkpoint metadata is missing: {checkpoint_path}")

    model_signature = _require_mapping(metadata, "model_signature", checkpoint_path)
    static_config = _require_mapping(metadata, "static_config", checkpoint_path)
    training_config = _require_mapping(metadata, "training_config", checkpoint_path)
    sequence_lengths = _require_mapping(metadata, "sequence_lengths", checkpoint_path)

    model_config = ExportModelConfig(
        language=str(model_signature.get("language") or "unknown"),
        vocab_size=_require_int(model_signature, "vocab_size", checkpoint_path),
        pad_id=_read_int(
            primary=model_signature,
            primary_key="pad_id",
            fallback=static_config,
            fallback_key="pad_id",
            checkpoint_path=checkpoint_path,
        ),
        bos_id=_read_int(
            primary=model_signature,
            primary_key="bos_id",
            fallback=static_config,
            fallback_key="bos_id",
            checkpoint_path=checkpoint_path,
        ),
        eos_id=_read_int(
            primary=model_signature,
            primary_key="eos_id",
            fallback=static_config,
            fallback_key="eos_id",
            checkpoint_path=checkpoint_path,
        ),
        d_model=_require_int(model_signature, "d_model", checkpoint_path),
        attention_heads=_require_int(
            model_signature,
            "attention_heads",
            checkpoint_path,
        ),
        encoder_layers=_require_int(
            model_signature,
            "encoder_layers",
            checkpoint_path,
        ),
        decoder_layers=_require_int(
            model_signature,
            "decoder_layers",
            checkpoint_path,
        ),
        ff_dimension=_require_int(model_signature, "ff_dimension", checkpoint_path),
        dropout=_require_float(training_config, "dropout", checkpoint_path),
        max_source_positions=_read_int(
            primary=model_signature,
            primary_key="max_source_positions",
            fallback=sequence_lengths,
            fallback_key="max_source_positions",
            checkpoint_path=checkpoint_path,
        ),
        max_target_positions=_read_int(
            primary=model_signature,
            primary_key="max_target_positions",
            fallback=sequence_lengths,
            fallback_key="max_target_positions",
            checkpoint_path=checkpoint_path,
        ),
    )
    return checkpoint, model_config


def _require_mapping(
    metadata: dict,
    key: str,
    checkpoint_path: pathlib.Path,
) -> dict:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise SystemExit(
            f"Checkpoint metadata is missing {key}: {checkpoint_path}"
        )
    return value


def _read_int(
    *,
    primary: dict,
    primary_key: str,
    fallback: dict,
    fallback_key: str,
    checkpoint_path: pathlib.Path,
) -> int:
    value = primary.get(primary_key, fallback.get(fallback_key))
    if not isinstance(value, int):
        raise SystemExit(
            f"Checkpoint metadata field is missing or invalid: "
            f"{primary_key}/{fallback_key} in {checkpoint_path}"
        )
    return value


def _require_int(
    payload: dict,
    key: str,
    checkpoint_path: pathlib.Path,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise SystemExit(
            f"Checkpoint metadata field is missing or invalid: "
            f"{key} in {checkpoint_path}"
        )
    return value


def _require_float(
    payload: dict,
    key: str,
    checkpoint_path: pathlib.Path,
) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise SystemExit(
            f"Checkpoint metadata field is missing or invalid: "
            f"{key} in {checkpoint_path}"
        )
    return float(value)
