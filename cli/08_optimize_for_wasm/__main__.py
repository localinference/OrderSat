#!/usr/bin/env python3
from __future__ import annotations

import json

from args.parse import parse_args
from file.copy import copy_support_artifacts
from file.require import require_file
from QuantizationPaths.consturctor import build_quantization_paths
from quantize.uint8 import quantize_with_uint8
from quantize.validate import validate_quantized_model
from quantize.write import write_quantized_config


def main() -> None:
    args = parse_args()
    paths = build_quantization_paths(args.language)

    require_file(paths.source_model_path, "Source fp32 ONNX model")
    require_file(paths.source_config_path, "Source fp32 ONNX config")
    require_file(paths.source_tokenizer_model_path, "Source tokenizer model")

    paths.quantized_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        paths.quantized_model_path,
        paths.quantized_model_external_data_path,
        paths.quantized_config_path,
        paths.quantized_tokenizer_model_path,
        paths.quantized_dir / "tokenizer.vocab",
        paths.quantized_dir / "metrics.json",
    ):
        if stale_path.exists():
            stale_path.unlink()

    source_config = json.loads(paths.source_config_path.read_text(encoding="utf8"))

    quantization = quantize_with_uint8(paths=paths)
    validation = validate_quantized_model(
        source_model_path=paths.source_model_path,
        quantized_model_path=paths.quantized_model_path,
        source_config=source_config,
    )
    copy_support_artifacts(paths=paths)
    write_quantized_config(
        paths=paths,
        source_config=source_config,
        quantization=quantization,
        validation=validation,
    )

    print(f"language: {paths.language}")
    print(f"source_model: {paths.source_model_path}")
    print(f"quantized_model: {paths.quantized_model_path}")
    print(f"tokenizer_model: {paths.quantized_tokenizer_model_path}")
    print(f"config: {paths.quantized_config_path}")
    print(
        "quantization: "
        f"algorithm={quantization['algorithm']} "
        f"weight_type={quantization['weight_type']}"
    )
    print(
        "validation: "
        f"max_abs_diff={validation['max_abs_diff']:.8f} "
        f"argmax_match_rate={validation['argmax_match_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
