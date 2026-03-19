import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize ONNX models from src/06_fp32_export_onnx_models/{language} into "
            " builds under src/07_mixed-fp16_gpu_onnx_models/{language}."
        )
    )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        default="eng",
        help="Language directory under src/06_fp32_export_onnx_models/{languages} and msrc/08_mixed-fp16_gpu_onnx_models/{language}.",
    )

    return parser.parse_args()
