from __future__ import annotations

import random
import time

import numpy
import torch

from reporting.log import log_stage_complete, log_stage_start


def set_seed(seed: int) -> None:
    started_at = time.perf_counter()
    log_stage_start("seed.set", seed=seed)

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if getattr(torch.backends, "cudnn", None):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_stage_complete(
        "seed.set",
        duration_seconds=time.perf_counter() - started_at,
        seed=seed,
    )
