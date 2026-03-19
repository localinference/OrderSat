from __future__ import annotations

import pathlib
import warnings

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
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using the legacy TorchScript-based ONNX export.",
                    category=DeprecationWarning,
                )
                torch.onnx.export(
                    wrapper,
                    args=dummy_inputs,
                    f=str(onnx_model_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    dynamo=False,
                    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                    output_names=["logits"],
                    dynamic_axes=build_dynamic_axes(),
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

    input_ids, attention_mask, decoder_input_ids = build_dummy_inputs(model_config)
    with torch.inference_mode():
        torch_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).detach()

    session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
    )
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

    return {
        "source_length": int(input_ids.size(1)),
        "target_length": int(decoder_input_ids.size(1)),
        "logits_shape": list(ort_logits.shape),
        "max_abs_diff": max_abs_diff,
    }


def build_dummy_inputs(
    model_config: ExportModelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_length = max(2, min(8, model_config.max_source_positions))
    target_length = max(2, min(8, model_config.max_target_positions))

    input_ids = torch.full(
        (1, source_length),
        fill_value=model_config.bos_id,
        dtype=torch.long,
    )
    input_ids[0, -1] = model_config.eos_id

    attention_mask = torch.ones((1, source_length), dtype=torch.long)

    decoder_input_ids = torch.full(
        (1, target_length),
        fill_value=model_config.bos_id,
        dtype=torch.long,
    )
    decoder_input_ids[0, -1] = model_config.eos_id

    return input_ids, attention_mask, decoder_input_ids


def build_dynamic_axes() -> dict[str, dict[int, str]]:
    return {
        "input_ids": {0: "batch_size", 1: "source_length"},
        "attention_mask": {0: "batch_size", 1: "source_length"},
        "decoder_input_ids": {0: "batch_size", 1: "target_length"},
        "logits": {0: "batch_size", 1: "target_length"},
    }
