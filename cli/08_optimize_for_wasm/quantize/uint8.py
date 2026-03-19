from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

from QuantizationPaths.consturctor import QuantizationPaths

OP_TYPES_TO_QUANTIZE = ("MatMul", "Gemm")


def quantize_with_uint8(*, paths: QuantizationPaths) -> dict[str, object]:
    if paths.quantized_model_external_data_path.exists():
        paths.quantized_model_external_data_path.unlink()

    with TemporaryDirectory() as temporary_directory:
        preprocessed_model_path = Path(temporary_directory) / "model.preprocessed.onnx"
        quant_pre_process(
            input_model=str(paths.source_model_path),
            output_model_path=str(preprocessed_model_path),
            skip_symbolic_shape=True,
            save_as_external_data=False,
        )
        quantize_dynamic(
            model_input=str(preprocessed_model_path),
            model_output=str(paths.quantized_model_path),
            op_types_to_quantize=list(OP_TYPES_TO_QUANTIZE),
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            use_external_data_format=False,
        )

    return {
        "backend": "wasm",
        "algorithm": "dynamic",
        "weight_type": "QUInt8",
        "op_types_to_quantize": list(OP_TYPES_TO_QUANTIZE),
        "preprocessed": True,
        "skip_symbolic_shape": True,
        "external_data": False,
    }
