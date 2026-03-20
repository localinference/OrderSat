from __future__ import annotations

import time

import torch

from reporting.log import log_stage_complete, log_stage_start


def build_device(device_name: str) -> torch.device:
    started_at = time.perf_counter()
    log_stage_start("device.build", requested_device=device_name)

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        log_stage_complete(
            "device.build",
            duration_seconds=time.perf_counter() - started_at,
            requested_device=device_name,
            resolved_device=str(device),
        )
        return device

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    if device_name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend or not mps_backend.is_available():
            raise SystemExit("MPS was requested but is not available.")

    device = torch.device(device_name)
    log_stage_complete(
        "device.build",
        duration_seconds=time.perf_counter() - started_at,
        requested_device=device_name,
        resolved_device=str(device),
    )
    return device
