from __future__ import annotations

import json

from QuantizationPaths.consturctor import QuantizationPaths


def write_quantized_config(
    *,
    paths: QuantizationPaths,
    source_config: dict,
    quantization: dict[str, object],
    validation: dict[str, object],
) -> None:
    quantized_config = dict(source_config)
    quantized_config["format"] = "onnx-wasm-uint8"
    quantized_config["runtime_target"] = "wasm"
    quantized_config["precision"] = "uint8"
    quantized_config["onnx_model_filename"] = paths.quantized_model_path.name
    quantized_config["tokenizer_model_filename"] = (
        paths.quantized_tokenizer_model_path.name
    )
    quantized_config["source_onnx_model_filename"] = paths.source_model_path.name
    quantized_config["quantization"] = quantization
    quantized_config["validation"] = validation

    paths.quantized_config_path.write_text(
        f"{json.dumps(quantized_config, indent=2)}\n",
        encoding="utf8",
    )
