from __future__ import annotations

import json

from MixedPaths.consturctor import MixedPaths


def write_mixed_config(
    *,
    paths: MixedPaths,
    source_config: dict,
    mixed_precision: dict[str, object],
    validation: dict[str, object],
) -> None:
    mixed_config = dict(source_config)
    mixed_config["format"] = "onnx-webgpu-mixed-fp16"
    mixed_config["runtime_target"] = "webgpu"
    mixed_config["precision"] = "mixed-fp16"
    mixed_config["onnx_model_filename"] = paths.mixed_model_path.name
    mixed_config["tokenizer_model_filename"] = paths.mixed_tokenizer_model_path.name
    mixed_config["source_onnx_model_filename"] = paths.source_model_path.name
    mixed_config["mixed_precision"] = mixed_precision
    mixed_config["validation"] = validation

    paths.mixed_config_path.write_text(
        f"{json.dumps(mixed_config, indent=2)}\n",
        encoding="utf8",
    )
