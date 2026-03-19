from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the trained FP32 PyTorch seq2seq checkpoint to an ONNX "
            "bundle under src/06_FP32_export_onnx_models/{language}."
        )
    )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        default="eng",
        help=(
            "Language directory under src/03_tokenizers, "
            "src/05_pytorch_models, and src/06_FP32_export_onnx_models."
        ),
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version to export.",
    )
    return parser.parse_args()
