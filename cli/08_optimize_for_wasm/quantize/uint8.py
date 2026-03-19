import pathlib
from dataclasses import dataclass

def quantize_with_uint8(args: argparse.Namespace, paths: QuantizationPaths) -> dict[str, Any]:
    quant_module = importlib.import_module("onnxruntime.quantization.matmul_nbits_quantizer")
    quant_utils = importlib.import_module("onnxruntime.quantization.quant_utils")

    model = onnx.load(str(paths.source_model_path))
    quant_config = quant_module.DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=args.int4_accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    quantizer = quant_module.MatMulNBitsQuantizer(
        model=model,
        bits=4,
        accuracy_level=args.int4_accuracy_level,
        nodes_to_exclude=args.nodes_to_exclude,
        nodes_to_include=args.nodes_to_include or None,
        algo_config=quant_config,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(paths.quantized_model_path), True)

    return {
        "backend": "int4",
        "algorithm": "MatMulNBits",
        "bits": 4,
        "blockSize": args.block_size,
        "accuracyLevel": args.int4_accuracy_level,
        "nodesExcluded": list(args.nodes_to_exclude),
        "nodesIncluded": list(args.nodes_to_include),
    }



@dataclass(frozen=True)
class QuantizationPaths:
    source_model_path: pathlib.Path
    source_config_path: pathlib.Path
    source_metrics_path: pathlib.Path
    source_tokenizer_model_path: pathlib.Path
    source_tokenizer_vocab_path: pathlib.Path
    quantized_dir: pathlib.Path
    quantized_model_path: pathlib.Path
    quantized_config_path: pathlib.Path
    quantized_metrics_path: pathlib.Path
    quantized_tokenizer_model_path: pathlib.Path
    quantized_tokenizer_vocab_path: pathlib.Path