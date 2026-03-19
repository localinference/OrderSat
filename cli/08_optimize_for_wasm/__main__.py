#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import shutil
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

import QuantizationPaths

from args.parse import parse_args
from file.require import require_file


ONNX_EXPORTS_ROOT = pathlib.Path("src/06_FP32_export_onnx_models")
ONNX_MODEL_NAME = "model.onnx"

WASM_MODELS_ROOT = pathlib.Path("src/08_INT8_cpu_onnx_models")
WASM_MODEL_NAME = "model.int8.onnx"

BACKEND = "int8"
BLOCK_SIZE = 256
INT8_ACCURACY_LEVEL = None
BNB4_QUANT_TYPE = "nf8"
CONFIG_FILE_NAME = "config.json"
TOKENIZER_MODEL_NAME = "tokenizer.model"




def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend == "bnb4":
        return "bnb4"
    if args.backend == "int4":
        ensure_int4_backend_available()
        return "int4"
    try:
        ensure_int4_backend_available()
        return "int4"
    except SystemExit:
        return "bnb4"


def ensure_int4_backend_available() -> None:
    try:
        importlib.import_module("onnx_ir")
        importlib.import_module("onnxruntime.quantization.matmul_nbits_quantizer")
    except Exception as error:
        raise SystemExit(
            "The int4 backend requires both onnxruntime.quantization.matmul_nbits_quantizer "
            f"and the 'onnx_ir' package. Install the missing dependency first. Detail: {error}"
        ) from error


def quantize_with_int4(args: argparse.Namespace, paths: QuantizationPaths) -> dict[str, Any]:
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


def quantize_with_bnb4(args: argparse.Namespace, paths: QuantizationPaths) -> dict[str, Any]:
    quant_module = importlib.import_module("onnxruntime.quantization.matmul_bnb4_quantizer")
    model = onnx.load(str(paths.source_model_path))
    quant_type = (
        quant_module.MatMulBnb4Quantizer.NF4
        if args.bnb4_quant_type == "nf4"
        else quant_module.MatMulBnb4Quantizer.FP4
    )
    quantizer = quant_module.MatMulBnb4Quantizer(
        model=model,
        quant_type=quant_type,
        block_size=args.block_size,
        nodes_to_exclude=args.nodes_to_exclude,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(paths.quantized_model_path), True)

    return {
        "backend": "bnb4",
        "algorithm": "MatMulBnb4",
        "bits": 4,
        "blockSize": args.block_size,
        "quantType": args.bnb4_quant_type,
        "nodesExcluded": list(args.nodes_to_exclude),
    }


def build_validation_cases(source_config: dict[str, Any]) -> list[dict[str, int | str]]:
    limits = source_config.get("limits", {})
    max_source_length = int(limits.get("maxInputLength", 16))
    max_target_length = int(limits.get("maxDecoderLength", 8))
    candidates = [
        {
            "name": "example",
            "source_length": min(max_source_length, 16),
            "target_length": min(max_target_length, 8),
        },
        {
            "name": "max_source_single_decoder",
            "source_length": max_source_length,
            "target_length": 1,
        },
        {
            "name": "max_lengths",
            "source_length": max_source_length,
            "target_length": max_target_length,
        },
    ]

    seen: set[tuple[int, int]] = set()
    cases: list[dict[str, int | str]] = []

    for candidate in candidates:
        key = (candidate["source_length"], candidate["target_length"])
        if key in seen:
            continue
        seen.add(key)
        cases.append(candidate)

    return cases


def validate_quantized_model(model_path: pathlib.Path, source_config: dict[str, Any]) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_names = source_config.get(
        "inputNames",
        ["input_ids", "attention_mask", "decoder_input_ids"],
    )
    output_names = source_config.get("outputNames", ["logits"])
    token_ids = source_config.get("tokenIds", {})
    bos_id = int(token_ids.get("bos", 1))
    vocab_size = int(source_config.get("model", {}).get("vocabSize", 0))
    validation_cases: list[dict[str, Any]] = []

    for case in build_validation_cases(source_config):
        source_length = int(case["source_length"])
        target_length = int(case["target_length"])
        ort_inputs = {
            input_names[0]: np.ones((1, source_length), dtype=np.int64),
            input_names[1]: np.ones((1, source_length), dtype=np.int64),
            input_names[2]: np.full((1, target_length), bos_id, dtype=np.int64),
        }
        outputs = session.run(output_names, ort_inputs)
        logits = outputs[0]

        expected_shape = (1, target_length)
        actual_prefix_shape = tuple(logits.shape[:2])
        if actual_prefix_shape != expected_shape:
            raise SystemExit(
                f"Quantized model output shape mismatch for case '{case['name']}': "
                f"expected prefix {expected_shape}, got {actual_prefix_shape}"
            )
        if vocab_size and logits.shape[-1] != vocab_size:
            raise SystemExit(
                f"Quantized model vocab axis mismatch for case '{case['name']}': "
                f"expected {vocab_size}, got {logits.shape[-1]}"
            )

        validation_cases.append(
            {
                "name": case["name"],
                "inputShape": {
                    input_names[0]: list(ort_inputs[input_names[0]].shape),
                    input_names[1]: list(ort_inputs[input_names[1]].shape),
                    input_names[2]: list(ort_inputs[input_names[2]].shape),
                },
                "outputShape": list(logits.shape),
            }
        )

    return {
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
        "providers": session.get_providers(),
        "cases": validation_cases,
    }


def copy_support_artifacts(paths: QuantizationPaths) -> None:
    shutil.copy2(paths.source_tokenizer_model_path, paths.quantized_tokenizer_model_path)
    shutil.copy2(paths.source_tokenizer_vocab_path, paths.quantized_tokenizer_vocab_path)
    if paths.source_metrics_path.exists():
        shutil.copy2(paths.source_metrics_path, paths.quantized_metrics_path)


def write_quantized_config(
    *,
    paths: QuantizationPaths,
    quantization: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    if paths.source_config_path.exists():
        source_config = json.loads(paths.source_config_path.read_text(encoding="utf8"))
    else:
        source_config = {}

    source_config["format"] = "onnx-4bit-quantized-build"
    source_config["modelFile"] = paths.quantized_model_path.name
    source_config["quantization"] = quantization
    source_config["validation"] = validation

    paths.quantized_config_path.write_text(
        f"{json.dumps(source_config, indent=2)}\n",
        encoding="utf8",
    )


def main() -> None:
    args = parse_args()
    language = args.langauge
    
    source_model_path=ONNX_EXPORTS_ROOT / language / ONNX_MODEL_NAME
    source_config_path=ONNX_EXPORTS_ROOT / language / CONFIG_FILE_NAME
    source_tokenizer_model_path=ONNX_EXPORTS_ROOT / language / TOKENIZER_MODEL_NAME

    quantized_model_path=WASM_MODELS_ROOT / language / WASM_MODEL_NAME
    quantized_config_path=WASM_MODELS_ROOT / CONFIG_FILE_NAME
    quantized_tokenizer_model_path=WASM_MODELS_ROOT / language / TOKENIZER_MODEL_NAME

    require_file(source_model_path, "Source ONNX model")
    require_file(source_config_path, "Source config")
    require_file(source_tokenizer_model_path, "Tokenizer model")

    targets = [
        quantized_model_path,
        quantized_config_path,
        quantized_tokenizer_model_path,
    ]
    for target in targets:
        if target.exists():
            if target.is_file():
                target.unlink()
            else:
                raise SystemExit(f"Expected file output path, got directory: {target}")


    source_config = json.loads(source_config_path.read_text(encoding="utf8"))

    backend = resolve_backend(args)
    if backend == "int4":
        quantization = quantize_with_int4(args, paths)
    else:
        quantization = quantize_with_bnb4(args, paths)

    validation = validate_quantized_model(paths.quantized_model_path, source_config)
    copy_support_artifacts(paths)
    write_quantized_config(
        paths=paths,
        quantization=quantization,
        validation=validation,
    )

    print(f"Language: {args.language}")
    print(f"Source model: {paths.source_model_path}")
    print(f"Quantized model: {paths.quantized_model_path}")
    print(f"Quantized config: {paths.quantized_config_path}")
    print(f"Backend: {quantization['backend']}")
    print(f"Inputs: {', '.join(validation['inputs'])}")
    print(f"Outputs: {', '.join(validation['outputs'])}")
    for case in validation["cases"]:
        print(f"Validation case {case['name']}: output_shape={case['outputShape']}")


if __name__ == "__main__":
    main()
