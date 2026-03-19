from __future__ import annotations

import shutil

from QuantizationPaths.consturctor import QuantizationPaths


def copy_support_artifacts(*, paths: QuantizationPaths) -> None:
    shutil.copy2(
        paths.source_tokenizer_model_path,
        paths.quantized_tokenizer_model_path,
    )
