import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize ONNX models from src/06_FP32_export_onnx_models/{language} into "
            "INT8 builds under src/08_INT8_cpu_onnx_models/{language}."
        )
    )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        default="eng",
        help="Language directory under src/06_FP32_export_onnx_models/{languages} and msrc/08_INT8_cpu_onnx_models/{language}.",
    )

    return parser.parse_args()
