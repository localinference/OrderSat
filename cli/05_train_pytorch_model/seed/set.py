from __future__ import annotations

import random

import numpy
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if getattr(torch.backends, "cudnn", None):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
