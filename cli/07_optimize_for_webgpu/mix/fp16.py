from __future__ import annotations

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

from MixedPaths.consturctor import MixedPaths


def mix_with_fp16(*, paths: MixedPaths) -> dict[str, object]:
    if paths.mixed_model_external_data_path.exists():
        paths.mixed_model_external_data_path.unlink()

    source_model = onnx.load(str(paths.source_model_path), load_external_data=False)
    mixed_model = convert_float_to_float16(
        source_model,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    onnx.save(
        mixed_model,
        str(paths.mixed_model_path),
        save_as_external_data=False,
    )

    return {
        "backend": "webgpu",
        "algorithm": "mixed-fp16",
        "keep_io_types": True,
        "disable_shape_infer": True,
        "external_data": False,
    }
