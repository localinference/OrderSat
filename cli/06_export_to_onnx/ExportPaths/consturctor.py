from __future__ import annotations

import pathlib
from dataclasses import dataclass

CHECKPOINT_NAME = "best.pt"
TOKENIZER_MODEL_NAME = "tokenizer.model"
ONNX_MODEL_NAME = "model.onnx"
CONFIG_NAME = "config.json"

TOKENIZERS_ROOT = pathlib.Path("src/03_tokenizers")
PYTORCH_MODELS_ROOT = pathlib.Path("src/05_pytorch_models")
ONNX_EXPORT_ROOT = pathlib.Path("src/06_FP32_export_onnx_models")


@dataclass(frozen=True)
class ExportPaths:
    language: str
    checkpoint_path: pathlib.Path
    tokenizer_model_path: pathlib.Path
    export_dir: pathlib.Path
    onnx_model_path: pathlib.Path
    config_path: pathlib.Path
    exported_tokenizer_model_path: pathlib.Path


def build_export_paths(language: str) -> ExportPaths:
    export_dir = ONNX_EXPORT_ROOT / language
    return ExportPaths(
        language=language,
        checkpoint_path=PYTORCH_MODELS_ROOT / language / CHECKPOINT_NAME,
        tokenizer_model_path=TOKENIZERS_ROOT / language / TOKENIZER_MODEL_NAME,
        export_dir=export_dir,
        onnx_model_path=export_dir / ONNX_MODEL_NAME,
        config_path=export_dir / CONFIG_NAME,
        exported_tokenizer_model_path=export_dir / TOKENIZER_MODEL_NAME,
    )
