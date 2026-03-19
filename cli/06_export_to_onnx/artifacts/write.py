from __future__ import annotations

import json
import shutil
from dataclasses import asdict

from checkpoint.load import ExportModelConfig
from ExportPaths.consturctor import ExportPaths


def write_export_bundle(
    *,
    paths: ExportPaths,
    checkpoint: dict,
    model_config: ExportModelConfig,
    opset_version: int,
    validation: dict[str, object],
) -> None:
    if not paths.tokenizer_model_path.exists():
        raise SystemExit(
            f"Tokenizer model does not exist: {paths.tokenizer_model_path}"
        )
    if not paths.tokenizer_vocab_path.exists():
        raise SystemExit(
            f"Tokenizer vocab does not exist: {paths.tokenizer_vocab_path}"
        )

    paths.export_dir.mkdir(parents=True, exist_ok=True)

    legacy_metrics_path = paths.export_dir / "metrics.json"
    if legacy_metrics_path.exists():
        legacy_metrics_path.unlink()

    shutil.copy2(paths.tokenizer_model_path, paths.exported_tokenizer_model_path)
    shutil.copy2(paths.tokenizer_vocab_path, paths.exported_tokenizer_vocab_path)

    config_payload = {
        "language": paths.language,
        "format": "onnx",
        "precision": "fp32",
        "onnx_model_filename": paths.onnx_model_path.name,
        "tokenizer_model_filename": paths.exported_tokenizer_model_path.name,
        "tokenizer_vocab_filename": paths.exported_tokenizer_vocab_path.name,
        "source_checkpoint_path": str(paths.checkpoint_path),
        "source_metrics": checkpoint.get("metrics"),
        "model_config": asdict(model_config),
        "validation": validation,
        "export": {
            "opset_version": opset_version,
            "input_names": [
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
            ],
            "output_names": ["logits"],
        },
    }
    paths.config_path.write_text(
        f"{json.dumps(config_payload, indent=2)}\n",
        encoding="utf8",
    )
