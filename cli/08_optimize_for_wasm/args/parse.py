from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize fp32 ONNX exports from "
            "src/06_fp32_export_onnx_models/{language} into uint8 WASM-targeted "
            "builds under src/08_uint8_cpu_onnx_models/{language}."
        )
    )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        default="eng",
        help=(
            "Language directory under src/06_fp32_export_onnx_models and "
            "src/08_uint8_cpu_onnx_models."
        ),
    )
    return parser.parse_args()
