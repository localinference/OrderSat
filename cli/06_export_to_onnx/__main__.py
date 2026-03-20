#!/usr/bin/env python3
from __future__ import annotations

import sys

from args.parse import parse_args
from artifacts.write import write_export_bundle
from checkpoint.load import load_export_checkpoint
from ExportPaths.consturctor import (
    ONNX_EXPORT_ROOT,
    PYTORCH_MODELS_ROOT,
    TOKENIZERS_ROOT,
    build_export_paths,
)
from onnx_model.export import export_onnx_model, validate_exported_onnx_model
from pytorch_model.build import build_pytorch_model
from selection.discover import discover_export_candidates
from selection.select import select_best_candidate


def configure_text_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue


def main() -> None:
    configure_text_streams()
    args = parse_args()
    selection = select_best_candidate(
        discover_export_candidates(
            language=args.language,
            tokenizers_root=TOKENIZERS_ROOT,
            datasets_root=PYTORCH_MODELS_ROOT.parent / "04_training_datasets",
            pytorch_models_root=PYTORCH_MODELS_ROOT,
        )
    )
    paths = build_export_paths(args.language, selection.selected_format)

    checkpoint, model_config = load_export_checkpoint(paths.checkpoint_path)
    model = build_pytorch_model(
        checkpoint=checkpoint,
        model_config=model_config,
    )

    export_onnx_model(
        model=model,
        model_config=model_config,
        onnx_model_path=paths.onnx_model_path,
        opset_version=args.opset_version,
    )
    validation = validate_exported_onnx_model(
        model=model,
        model_config=model_config,
        onnx_model_path=paths.onnx_model_path,
    )

    write_export_bundle(
        paths=paths,
        checkpoint=checkpoint,
        model_config=model_config,
        opset_version=args.opset_version,
        validation=validation,
        selection=selection.to_dict(),
    )

    print(f"language: {paths.language}")
    print(f"selected_format: {paths.selected_format}")
    print(f"checkpoint: {paths.checkpoint_path}")
    print(f"onnx_model: {paths.onnx_model_path}")
    print(f"tokenizer_model: {paths.exported_tokenizer_model_path}")
    print(f"config: {paths.config_path}")
    print(f"selection_confidence: {selection.confidence}")
    print(f"selection_reason: {selection.reason}")
    print(
        "validation: "
        f"shape={validation['logits_shape']} "
        f"max_abs_diff={validation['max_abs_diff']:.8f}"
    )
    print(f"validated_cases: {validation['validated_case_count']}")


if __name__ == "__main__":
    main()
