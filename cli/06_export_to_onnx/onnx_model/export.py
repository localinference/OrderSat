from __future__ import annotations

import pathlib

import onnx
import onnxruntime as ort
import torch

from checkpoint.load import ExportModelConfig
from OnnxExportWrapper.constructor import OnnxExportWrapper


def export_onnx_model(
    *,
    model: torch.nn.Module,
    model_config: ExportModelConfig,
    onnx_model_path: pathlib.Path,
    opset_version: int,
) -> None:
    wrapper = OnnxExportWrapper(model).eval()
    dummy_inputs = build_dummy_inputs(model_config)
    external_data_path = onnx_model_path.with_name(f"{onnx_model_path.name}.data")

    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    if external_data_path.exists():
        external_data_path.unlink()
    previous_fastpath_state = torch.backends.mha.get_fastpath_enabled()

    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                args=dummy_inputs,
                f=str(onnx_model_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=True,
                external_data=False,
                fallback=False,
                input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_shapes=build_dynamic_shapes(),
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath_state)


def validate_exported_onnx_model(
    *,
    model: torch.nn.Module,
    model_config: ExportModelConfig,
    onnx_model_path: pathlib.Path,
) -> dict[str, object]:
    onnx_model = onnx.load(str(onnx_model_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
    )
    cases: list[dict[str, object]] = []

    for case_name, source_length, target_length in build_validation_cases(model_config):
        input_ids, attention_mask, decoder_input_ids = build_inputs(
            model_config=model_config,
            source_length=source_length,
            target_length=target_length,
        )
        with torch.inference_mode():
            torch_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).detach()

        ort_logits = session.run(
            ["logits"],
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
                "decoder_input_ids": decoder_input_ids.numpy(),
            },
        )[0]
        max_abs_diff = float(
            (torch_logits - torch.from_numpy(ort_logits)).abs().max().item()
        )
        cases.append(
            {
                "name": case_name,
                "source_length": source_length,
                "target_length": target_length,
                "logits_shape": list(ort_logits.shape),
                "max_abs_diff": max_abs_diff,
            }
        )

    reference = cases[0]
    return {
        "source_length": reference["source_length"],
        "target_length": reference["target_length"],
        "logits_shape": reference["logits_shape"],
        "max_abs_diff": max(case["max_abs_diff"] for case in cases),
        "validated_case_count": len(cases),
        "validated_dynamic_shapes": True,
        "reference": reference,
        "cases": cases,
    }


def build_dummy_inputs(
    model_config: ExportModelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_length = max(2, min(8, model_config.max_source_positions))
    target_length = max(2, min(8, model_config.max_target_positions))
    return build_inputs(
        model_config=model_config,
        source_length=source_length,
        target_length=target_length,
    )


def build_inputs(
    *,
    model_config: ExportModelConfig,
    source_length: int,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_length = max(1, source_length)
    target_length = max(1, target_length)

    input_ids = torch.full(
        (1, source_length),
        fill_value=model_config.bos_id,
        dtype=torch.int32,
    )
    input_ids[0, -1] = model_config.eos_id

    attention_mask = torch.ones((1, source_length), dtype=torch.int32)

    decoder_input_ids = torch.full(
        (1, target_length),
        fill_value=model_config.bos_id,
        dtype=torch.int32,
    )
    decoder_input_ids[0, -1] = model_config.eos_id

    return input_ids, attention_mask, decoder_input_ids


def build_dynamic_shapes() -> dict[str, dict[int, torch.export.Dim]]:
    return {
        "input_ids": {
            0: torch.export.Dim("batch_size"),
            1: torch.export.Dim("source_length"),
        },
        "attention_mask": {
            0: torch.export.Dim("batch_size"),
            1: torch.export.Dim("source_length"),
        },
        "decoder_input_ids": {
            0: torch.export.Dim("batch_size"),
            1: torch.export.Dim("target_length"),
        },
    }


def build_validation_cases(
    model_config: ExportModelConfig,
) -> list[tuple[str, int, int]]:
    reference_source_length = max(2, min(8, model_config.max_source_positions))
    reference_target_length = max(2, min(8, model_config.max_target_positions))
    candidates = [
        ("reference", reference_source_length, reference_target_length),
        (
            "longer_source",
            max(2, min(16, model_config.max_source_positions)),
            reference_target_length,
        ),
        (
            "longer_target",
            reference_source_length,
            max(2, min(16, model_config.max_target_positions)),
        ),
        (
            "max_source_decode_start",
            model_config.max_source_positions,
            1,
        ),
        (
            "wider_window",
            max(2, min(32, model_config.max_source_positions)),
            max(2, min(32, model_config.max_target_positions)),
        ),
    ]
    cases: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for case_name, source_length, target_length in candidates:
        case_key = (source_length, target_length)
        if case_key in seen:
            continue
        seen.add(case_key)
        cases.append((case_name, source_length, target_length))

    return cases
