from __future__ import annotations

import pathlib

import numpy as np
import onnx
import onnxruntime as ort


def validate_mixed_model(
    *,
    source_model_path: pathlib.Path,
    mixed_model_path: pathlib.Path,
    source_config: dict,
) -> dict[str, object]:
    source_model = onnx.load(str(source_model_path), load_external_data=False)
    mixed_model = onnx.load(str(mixed_model_path), load_external_data=False)
    onnx.checker.check_model(source_model)
    onnx.checker.check_model(mixed_model)

    external_initializer_count = sum(
        1
        for initializer in mixed_model.graph.initializer
        if initializer.external_data
    )
    if external_initializer_count:
        raise SystemExit(
            "Mixed-fp16 WebGPU model unexpectedly uses ONNX external data."
        )

    source_session = ort.InferenceSession(
        str(source_model_path),
        providers=["CPUExecutionProvider"],
    )
    mixed_session = ort.InferenceSession(
        str(mixed_model_path),
        providers=["CPUExecutionProvider"],
    )

    export_config = source_config.get("export") or {}
    input_names = export_config.get(
        "input_names",
        ["input_ids", "attention_mask", "decoder_input_ids"],
    )
    output_names = export_config.get("output_names", ["logits"])

    max_abs_diff = 0.0
    argmax_match_count = 0
    argmax_total = 0
    validation_cases: list[dict[str, object]] = []

    for case in build_validation_cases(source_config):
        ort_inputs = build_validation_inputs(case=case, source_config=source_config)
        source_logits = source_session.run(output_names, ort_inputs)[0]
        mixed_logits = mixed_session.run(output_names, ort_inputs)[0]

        if source_logits.shape != mixed_logits.shape:
            raise SystemExit(
                f"Mixed model output shape mismatch for case '{case['name']}': "
                f"{source_logits.shape} vs {mixed_logits.shape}"
            )

        case_max_abs_diff = float(np.abs(source_logits - mixed_logits).max())
        source_argmax = source_logits.argmax(axis=-1)
        mixed_argmax = mixed_logits.argmax(axis=-1)
        case_argmax_match_count = int((source_argmax == mixed_argmax).sum())
        case_argmax_total = int(source_argmax.size)

        max_abs_diff = max(max_abs_diff, case_max_abs_diff)
        argmax_match_count += case_argmax_match_count
        argmax_total += case_argmax_total

        validation_cases.append(
            {
                "name": case["name"],
                "input_shape": {
                    input_names[0]: list(ort_inputs[input_names[0]].shape),
                    input_names[1]: list(ort_inputs[input_names[1]].shape),
                    input_names[2]: list(ort_inputs[input_names[2]].shape),
                },
                "output_shape": list(mixed_logits.shape),
                "max_abs_diff": case_max_abs_diff,
                "argmax_match_rate": (
                    case_argmax_match_count / case_argmax_total
                    if case_argmax_total
                    else 1.0
                ),
            }
        )

    source_model_bytes = source_model_path.stat().st_size
    mixed_model_bytes = mixed_model_path.stat().st_size

    return {
        "inputs": [item.name for item in mixed_session.get_inputs()],
        "outputs": [item.name for item in mixed_session.get_outputs()],
        "providers": mixed_session.get_providers(),
        "external_initializer_count": external_initializer_count,
        "source_model_bytes": source_model_bytes,
        "mixed_model_bytes": mixed_model_bytes,
        "size_reduction_bytes": source_model_bytes - mixed_model_bytes,
        "size_reduction_ratio": (
            1.0 - (mixed_model_bytes / source_model_bytes)
            if source_model_bytes
            else 0.0
        ),
        "max_abs_diff": max_abs_diff,
        "argmax_match_rate": (
            argmax_match_count / argmax_total if argmax_total else 1.0
        ),
        "cases": validation_cases,
    }


def build_validation_cases(source_config: dict) -> list[dict[str, int | str]]:
    reference_validation = source_config.get("validation") or {}
    return [
        {
            "name": "reference",
            "source_length": int(reference_validation.get("source_length", 8)),
            "target_length": int(reference_validation.get("target_length", 8)),
        }
    ]


def build_validation_inputs(
    *,
    case: dict[str, int | str],
    source_config: dict,
) -> dict[str, np.ndarray]:
    export_config = source_config.get("export") or {}
    model_config = source_config.get("model_config") or {}

    input_names = export_config.get(
        "input_names",
        ["input_ids", "attention_mask", "decoder_input_ids"],
    )

    vocab_size = int(model_config.get("vocab_size", 8))
    bos_id = int(model_config.get("bos_id", 1))
    eos_id = int(model_config.get("eos_id", 2))
    source_length = int(case["source_length"])
    target_length = int(case["target_length"])

    return {
        input_names[0]: build_token_ids(
            length=source_length,
            vocab_size=vocab_size,
            bos_id=bos_id,
            eos_id=eos_id,
        ),
        input_names[1]: np.ones((1, source_length), dtype=np.int64),
        input_names[2]: build_token_ids(
            length=target_length,
            vocab_size=vocab_size,
            bos_id=bos_id,
            eos_id=eos_id,
        ),
    }


def build_token_ids(
    *,
    length: int,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
) -> np.ndarray:
    token_upper_bound = max(3, vocab_size - 1)
    token_ids = ((np.arange(length, dtype=np.int64) * 7) + 3) % token_upper_bound
    if length >= 1:
        token_ids[0] = bos_id
    if length >= 2:
        token_ids[-1] = eos_id
    return token_ids.reshape(1, length)
