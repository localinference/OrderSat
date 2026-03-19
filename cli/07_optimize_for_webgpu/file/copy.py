from __future__ import annotations

import shutil

from MixedPaths.consturctor import MixedPaths


def copy_support_artifacts(*, paths: MixedPaths) -> None:
    shutil.copy2(
        paths.source_tokenizer_model_path,
        paths.mixed_tokenizer_model_path,
    )
