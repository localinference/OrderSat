from __future__ import annotations

import pathlib
from dataclasses import dataclass

FP32_EXPORT_ROOT = pathlib.Path("src/06_FP32_export_onnx_models")
WEBGPU_MIXED_ROOT = pathlib.Path("src/07_mixed-fp16_gpu_onnx_models")

MODEL_NAME = "model.onnx"
MIXED_MODEL_NAME = "model.mixed-fp16.onnx"
CONFIG_NAME = "config.json"
TOKENIZER_MODEL_NAME = "tokenizer.model"


@dataclass(frozen=True)
class MixedPaths:
    language: str
    source_model_path: pathlib.Path
    source_config_path: pathlib.Path
    source_tokenizer_model_path: pathlib.Path
    mixed_dir: pathlib.Path
    mixed_model_path: pathlib.Path
    mixed_model_external_data_path: pathlib.Path
    mixed_config_path: pathlib.Path
    mixed_tokenizer_model_path: pathlib.Path


def build_mixed_paths(language: str) -> MixedPaths:
    mixed_dir = WEBGPU_MIXED_ROOT / language
    mixed_model_path = mixed_dir / MIXED_MODEL_NAME
    return MixedPaths(
        language=language,
        source_model_path=FP32_EXPORT_ROOT / language / MODEL_NAME,
        source_config_path=FP32_EXPORT_ROOT / language / CONFIG_NAME,
        source_tokenizer_model_path=FP32_EXPORT_ROOT / language / TOKENIZER_MODEL_NAME,
        mixed_dir=mixed_dir,
        mixed_model_path=mixed_model_path,
        mixed_model_external_data_path=mixed_dir / f"{MIXED_MODEL_NAME}.data",
        mixed_config_path=mixed_dir / CONFIG_NAME,
        mixed_tokenizer_model_path=mixed_dir / TOKENIZER_MODEL_NAME,
    )
