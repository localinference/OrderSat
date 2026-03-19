from __future__ import annotations

import pathlib
from dataclasses import dataclass

FP32_EXPORT_ROOT = pathlib.Path("src/06_fp32_export_onnx_models")
WASM_QUANTIZED_ROOT = pathlib.Path("src/08_uint8_cpu_onnx_models")

MODEL_NAME = "model.onnx"
QUANTIZED_MODEL_NAME = "model.uint8.onnx"
CONFIG_NAME = "config.json"
TOKENIZER_MODEL_NAME = "tokenizer.model"


@dataclass(frozen=True)
class QuantizationPaths:
    language: str
    source_model_path: pathlib.Path
    source_config_path: pathlib.Path
    source_tokenizer_model_path: pathlib.Path
    quantized_dir: pathlib.Path
    quantized_model_path: pathlib.Path
    quantized_model_external_data_path: pathlib.Path
    quantized_config_path: pathlib.Path
    quantized_tokenizer_model_path: pathlib.Path


def build_quantization_paths(language: str) -> QuantizationPaths:
    quantized_dir = WASM_QUANTIZED_ROOT / language
    quantized_model_path = quantized_dir / QUANTIZED_MODEL_NAME
    return QuantizationPaths(
        language=language,
        source_model_path=FP32_EXPORT_ROOT / language / MODEL_NAME,
        source_config_path=FP32_EXPORT_ROOT / language / CONFIG_NAME,
        source_tokenizer_model_path=FP32_EXPORT_ROOT / language / TOKENIZER_MODEL_NAME,
        quantized_dir=quantized_dir,
        quantized_model_path=quantized_model_path,
        quantized_model_external_data_path=quantized_dir
        / f"{QUANTIZED_MODEL_NAME}.data",
        quantized_config_path=quantized_dir / CONFIG_NAME,
        quantized_tokenizer_model_path=quantized_dir / TOKENIZER_MODEL_NAME,
    )
