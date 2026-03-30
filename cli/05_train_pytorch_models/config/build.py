from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from device.capabilities import DeviceCapabilities
from reporting.log import log_stage_complete, log_stage_start
from stats.parse import DatasetStats


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _round_to_multiple(value: float, multiple: int) -> int:
    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


def _resolve_estimated_examples_per_batch(
    *,
    dataset_stats: DatasetStats,
    device_capabilities: DeviceCapabilities,
    capacity_scale: float,
) -> int:
    if device_capabilities.resolved_device == "cpu":
        return 1

    total_sequence_length = (
        dataset_stats.input_lengths.max + dataset_stats.label_lengths.max + 1
    )
    length_pressure = max(1.0, total_sequence_length / 1024.0)
    model_pressure = max(0.5, capacity_scale**1.5)
    memory_budget = device_capabilities.device_scale / (
        length_pressure * model_pressure
    )

    if device_capabilities.resolved_device == "mps":
        if (
            (device_capabilities.system_memory_gb or 0.0) >= 24
            and memory_budget >= 1.2
        ):
            return 2
        return 1

    accelerator_memory_gb = device_capabilities.accelerator_memory_gb or 0.0
    if memory_budget >= 1.6 and accelerator_memory_gb >= 16:
        return 4
    if memory_budget >= 0.9 and accelerator_memory_gb >= 8:
        return 2
    return 1


def _resolve_num_workers(
    *,
    dataset_stats: DatasetStats,
    device_capabilities: DeviceCapabilities,
) -> int:
    if dataset_stats.train_count < 256:
        return 0
    return min(device_capabilities.recommended_num_workers, 4)


@dataclass(frozen=True)
class TrainingConfig:
    data_scale: float
    device_scale: float
    capacity_scale: float
    d_model: int
    attention_heads: int
    encoder_layers: int
    decoder_layers: int
    ff_dimension: int
    dropout: float
    average_sequence_tokens: int
    target_effective_batch_size: int
    estimated_examples_per_batch: int
    max_batch_size: int
    target_tokens_per_batch: int
    target_tokens_per_optimizer_step: int
    accumulation_steps: int
    achieved_effective_batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    validation_exact_match_frequency: int
    run_train_exact_match_at_end: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_adjusted_options_dict(self) -> dict:
        return {
            "scale_inputs": {
                "data_scale": self.data_scale,
                "device_scale": self.device_scale,
                "capacity_scale": self.capacity_scale,
            },
            "model": {
                "d_model": self.d_model,
                "attention_heads": self.attention_heads,
                "encoder_layers": self.encoder_layers,
                "decoder_layers": self.decoder_layers,
                "ff_dimension": self.ff_dimension,
                "dropout": self.dropout,
            },
            "batching": {
                "average_sequence_tokens": self.average_sequence_tokens,
                "target_effective_batch_size": self.target_effective_batch_size,
                "estimated_examples_per_batch": self.estimated_examples_per_batch,
                "max_batch_size": self.max_batch_size,
                "target_tokens_per_batch": self.target_tokens_per_batch,
                "target_tokens_per_optimizer_step": self.target_tokens_per_optimizer_step,
                "accumulation_steps": self.accumulation_steps,
                "achieved_effective_batch_size": self.achieved_effective_batch_size,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
            },
            "optimization": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "schedule": {
                "epochs": self.epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "validation_exact_match_frequency": self.validation_exact_match_frequency,
                "run_train_exact_match_at_end": self.run_train_exact_match_at_end,
            },
        }


def build_training_config(
    *,
    dataset_stats: DatasetStats,
    device_capabilities: DeviceCapabilities,
) -> TrainingConfig:
    started_at = time.perf_counter()
    log_stage_start(
        "config.build",
        language=dataset_stats.language or "unknown",
        train_count=dataset_stats.train_count,
        requested_device=device_capabilities.requested_device,
        resolved_device=device_capabilities.resolved_device,
    )

    raw_data_scale = (dataset_stats.train_count / 10_000) ** 0.25
    data_scale = _clamp_float(raw_data_scale, 0.5, 2.0)
    device_scale = device_capabilities.device_scale
    capacity_scale = min(data_scale, device_scale)

    d_model = _clamp_int(
        _round_to_multiple(512 * capacity_scale, 128),
        256,
        1024,
    )
    attention_heads = 8 if d_model <= 512 else 16
    encoder_layers = _clamp_int(round(4 * capacity_scale), 2, 6)
    decoder_layers = _clamp_int(round(4 * capacity_scale), 2, 6)
    ff_dimension = d_model * 4
    dropout = _clamp_float(0.10 / capacity_scale, 0.10, 0.20)

    average_sequence_tokens = _round_to_multiple(
        dataset_stats.input_lengths.avg + dataset_stats.label_lengths.avg + 1.0,
        32,
    )
    target_effective_batch_size = 16
    estimated_examples_per_batch = _resolve_estimated_examples_per_batch(
        dataset_stats=dataset_stats,
        device_capabilities=device_capabilities,
        capacity_scale=capacity_scale,
    )
    max_batch_size = _clamp_int(
        estimated_examples_per_batch * 4,
        estimated_examples_per_batch,
        target_effective_batch_size,
    )
    target_tokens_per_batch = _round_to_multiple(
        average_sequence_tokens * estimated_examples_per_batch,
        64,
    )
    target_tokens_per_optimizer_step = _round_to_multiple(
        average_sequence_tokens * target_effective_batch_size,
        64,
    )
    accumulation_steps = max(
        1,
        round(target_effective_batch_size / estimated_examples_per_batch),
    )
    achieved_effective_batch_size = (
        estimated_examples_per_batch * accumulation_steps
    )
    batch_scale = math.sqrt(
        achieved_effective_batch_size / target_effective_batch_size
    )

    learning_rate = _clamp_float(
        (2e-4 / math.sqrt(data_scale)) * batch_scale,
        1e-4,
        3e-4,
    )
    weight_decay = _clamp_float(1e-4 / (data_scale**2), 1e-4, 5e-4)
    epochs = _clamp_int(round(30 / (data_scale**2)), 10, 100)
    early_stopping_patience = _clamp_int(round(8 / data_scale), 3, 20)
    early_stopping_min_delta = _clamp_float(1e-4 * data_scale, 1e-5, 1e-3)
    validation_exact_match_frequency = _clamp_int(round(2 * data_scale), 1, 3)

    num_workers = _resolve_num_workers(
        dataset_stats=dataset_stats,
        device_capabilities=device_capabilities,
    )

    config = TrainingConfig(
        data_scale=data_scale,
        device_scale=device_scale,
        capacity_scale=capacity_scale,
        d_model=d_model,
        attention_heads=attention_heads,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        ff_dimension=ff_dimension,
        dropout=dropout,
        average_sequence_tokens=average_sequence_tokens,
        target_effective_batch_size=target_effective_batch_size,
        estimated_examples_per_batch=estimated_examples_per_batch,
        max_batch_size=max_batch_size,
        target_tokens_per_batch=target_tokens_per_batch,
        target_tokens_per_optimizer_step=target_tokens_per_optimizer_step,
        accumulation_steps=accumulation_steps,
        achieved_effective_batch_size=achieved_effective_batch_size,
        num_workers=num_workers,
        pin_memory=device_capabilities.pin_memory,
        persistent_workers=num_workers > 0,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        validation_exact_match_frequency=validation_exact_match_frequency,
        run_train_exact_match_at_end=True,
    )

    log_stage_complete(
        "config.build",
        duration_seconds=time.perf_counter() - started_at,
        language=dataset_stats.language or "unknown",
        data_scale=f"{config.data_scale:.4f}",
        device_scale=f"{config.device_scale:.4f}",
        capacity_scale=f"{config.capacity_scale:.4f}",
        d_model=config.d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        estimated_examples_per_batch=config.estimated_examples_per_batch,
        max_batch_size=config.max_batch_size,
        target_tokens_per_batch=config.target_tokens_per_batch,
        accumulation_steps=config.accumulation_steps,
        effective_batch_size=config.achieved_effective_batch_size,
        learning_rate=f"{config.learning_rate:.6f}",
        epochs=config.epochs,
        validation_exact_match_frequency=config.validation_exact_match_frequency,
    )
    return config
