#!/usr/bin/env python3
from __future__ import annotations

import json

from args.parse import parse_args
from file.copy import copy_support_artifacts
from file.require import require_file
from MixedPaths.consturctor import build_mixed_paths
from mix.fp16 import mix_with_fp16
from mix.validate import validate_mixed_model
from mix.write import write_mixed_config


def main() -> None:
    args = parse_args()
    paths = build_mixed_paths(args.language)

    require_file(paths.source_model_path, "Source fp32 ONNX model")
    require_file(paths.source_config_path, "Source fp32 ONNX config")
    require_file(paths.source_tokenizer_model_path, "Source tokenizer model")

    paths.mixed_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        paths.mixed_model_path,
        paths.mixed_model_external_data_path,
        paths.mixed_config_path,
        paths.mixed_tokenizer_model_path,
        paths.mixed_dir / "tokenizer.vocab",
        paths.mixed_dir / "metrics.json",
    ):
        if stale_path.exists():
            stale_path.unlink()

    source_config = json.loads(paths.source_config_path.read_text(encoding="utf8"))

    mixed_precision = mix_with_fp16(paths=paths)
    validation = validate_mixed_model(
        source_model_path=paths.source_model_path,
        mixed_model_path=paths.mixed_model_path,
        source_config=source_config,
    )
    copy_support_artifacts(paths=paths)
    write_mixed_config(
        paths=paths,
        source_config=source_config,
        mixed_precision=mixed_precision,
        validation=validation,
    )

    print(f"language: {paths.language}")
    print(f"source_model: {paths.source_model_path}")
    print(f"mixed_model: {paths.mixed_model_path}")
    print(f"tokenizer_model: {paths.mixed_tokenizer_model_path}")
    print(f"config: {paths.mixed_config_path}")
    print(
        "mixed_precision: "
        f"keep_io_types={mixed_precision['keep_io_types']} "
        f"disable_shape_infer={mixed_precision['disable_shape_infer']}"
    )
    print(
        "validation: "
        f"max_abs_diff={validation['max_abs_diff']:.8f} "
        f"argmax_match_rate={validation['argmax_match_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
