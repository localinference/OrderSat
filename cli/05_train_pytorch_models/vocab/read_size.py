from __future__ import annotations

import pathlib
import time

from reporting.log import log_stage_complete, log_stage_start


def read_vocab_size(vocab_path: pathlib.Path) -> int:
    started_at = time.perf_counter()
    log_stage_start("vocab.read", path=str(vocab_path))

    if not vocab_path.exists():
        raise SystemExit(f"Tokenizer vocab does not exist: {vocab_path}")
    if not vocab_path.is_file():
        raise SystemExit(f"Tokenizer vocab path is not a file: {vocab_path}")

    with vocab_path.open("r", encoding="utf8") as handle:
        size = sum(1 for line in handle if line.strip())

    if size < 3:
        raise SystemExit(
            f"Tokenizer vocab looks invalid or too small: {vocab_path}"
        )

    log_stage_complete(
        "vocab.read",
        duration_seconds=time.perf_counter() - started_at,
        path=str(vocab_path),
        vocab_size=size,
    )
    return size
