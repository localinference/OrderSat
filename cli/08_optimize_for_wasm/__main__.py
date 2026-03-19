#!/usr/bin/env python3
from __future__ import annotations
import json
import pathlib





from args.parse import parse_args
from file.require import require_file
from file.copy import copy_file
from quantize.uint8 import quantize_with_uint8
from quantize.validate import validate_quantized_model
from quantize.write import write_quantized_config



ONNX_EXPORTS_ROOT = pathlib.Path("src/06_FP32_export_onnx_models")
ONNX_MODEL_NAME = "model.onnx"

WASM_MODELS_ROOT = pathlib.Path("src/08_INT8_cpu_onnx_models")
WASM_MODEL_NAME = "model.uint8.onnx"

BACKEND = "uint8"
BLOCK_SIZE = 256
UINT8_ACCURACY_LEVEL = None
CONFIG_FILE_NAME = "config.json"
TOKENIZER_MODEL_NAME = "tokenizer.model"

def main() -> None:
    args = parse_args()
    language = args.langauge
    
    source_model_path=ONNX_EXPORTS_ROOT / language / ONNX_MODEL_NAME
    source_config_path=ONNX_EXPORTS_ROOT / language / CONFIG_FILE_NAME
    source_tokenizer_path=ONNX_EXPORTS_ROOT / language / TOKENIZER_MODEL_NAME

    quantized_model_path=WASM_MODELS_ROOT / language / WASM_MODEL_NAME
    quantized_config_path=WASM_MODELS_ROOT / CONFIG_FILE_NAME
    quantized_tokenizer_path=WASM_MODELS_ROOT / language / TOKENIZER_MODEL_NAME

    require_file(source_model_path, "Source ONNX model")
    require_file(source_config_path, "Source config")
    require_file(source_tokenizer_path, "Tokenizer model")

    targets = [
        quantized_model_path,
        quantized_config_path,
        quantized_tokenizer_path,
    ]
    for target in targets:
        if target.exists():
            if target.is_file():
                target.unlink()

    source_config = json.loads(source_config_path.read_text(encoding="utf8"))


    quantize_with_uint8(args, paths)


    validation = validate_quantized_model(paths.quantized_model_path, source_config)
    copy_file(source_tokenizer_model_path)
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
