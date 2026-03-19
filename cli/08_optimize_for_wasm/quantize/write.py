import QuantizationPaths

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
