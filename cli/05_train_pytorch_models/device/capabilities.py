from __future__ import annotations

import ctypes
import math
import os
import time
from dataclasses import asdict, dataclass

import torch

from reporting.log import log_stage_complete, log_stage_start


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _get_system_memory_bytes() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None

    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(page_count, int):
            return int(page_size * page_count)

    return None


def _resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024 ** 3)


def _is_cuda_device(device_name: str) -> bool:
    return device_name == "cuda" or device_name.startswith("cuda:")


def _is_mps_device(device_name: str) -> bool:
    return device_name == "mps"


@dataclass(frozen=True)
class DeviceCapabilities:
    requested_device: str
    resolved_device: str
    accelerator_name: str | None
    accelerator_memory_bytes: int | None
    accelerator_memory_gb: float | None
    system_memory_bytes: int | None
    system_memory_gb: float | None
    cpu_count: int
    supports_fp16: bool
    supports_bf16: bool
    pin_memory: bool
    memory_scale: float
    throughput_scale: float
    device_scale: float
    recommended_num_workers: int

    def to_dict(self) -> dict:
        return asdict(self)


def get_device_capabilities(device_name: str = "auto") -> DeviceCapabilities:
    started_at = time.perf_counter()
    log_stage_start("device.capabilities", requested_device=device_name)

    resolved_device = _resolve_device(device_name)
    system_memory_bytes = _get_system_memory_bytes()
    system_memory_gb = _bytes_to_gb(system_memory_bytes)
    cpu_count = max(1, os.cpu_count() or 1)

    accelerator_name: str | None = None
    accelerator_memory_bytes: int | None = None
    supports_fp16 = False
    supports_bf16 = False

    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )

    if _is_cuda_device(resolved_device) and cuda_available:
        accelerator_index = torch.device(resolved_device).index
        if accelerator_index is None:
            accelerator_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(accelerator_index)
        accelerator_name = properties.name
        accelerator_memory_bytes = int(properties.total_memory)
        supports_fp16 = True
        supports_bf16 = bool(
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
    elif _is_mps_device(resolved_device) and mps_available:
        accelerator_name = "Apple Silicon GPU"
        supports_fp16 = True
    elif _is_cuda_device(resolved_device):
        accelerator_name = "CUDA (unavailable)"
    elif _is_mps_device(resolved_device):
        accelerator_name = "MPS (unavailable)"
    else:
        accelerator_name = "CPU"

    accelerator_memory_gb = _bytes_to_gb(accelerator_memory_bytes)
    primary_memory_gb = accelerator_memory_gb or system_memory_gb or 8.0

    memory_scale = _clamp(math.sqrt(primary_memory_gb / 8.0), 0.5, 2.0)
    throughput_scale = _clamp(math.sqrt(cpu_count / 8.0), 0.5, 2.0)
    device_scale = min(memory_scale, throughput_scale)

    if cpu_count <= 2:
        recommended_num_workers = 0
    else:
        recommended_num_workers = min(8, cpu_count - 1)

    capabilities = DeviceCapabilities(
        requested_device=device_name,
        resolved_device=resolved_device,
        accelerator_name=accelerator_name,
        accelerator_memory_bytes=accelerator_memory_bytes,
        accelerator_memory_gb=accelerator_memory_gb,
        system_memory_bytes=system_memory_bytes,
        system_memory_gb=system_memory_gb,
        cpu_count=cpu_count,
        supports_fp16=supports_fp16,
        supports_bf16=supports_bf16,
        pin_memory=_is_cuda_device(resolved_device),
        memory_scale=memory_scale,
        throughput_scale=throughput_scale,
        device_scale=device_scale,
        recommended_num_workers=recommended_num_workers,
    )

    log_stage_complete(
        "device.capabilities",
        duration_seconds=time.perf_counter() - started_at,
        requested_device=device_name,
        resolved_device=capabilities.resolved_device,
        accelerator_name=capabilities.accelerator_name,
        accelerator_memory_gb=capabilities.accelerator_memory_gb,
        system_memory_gb=capabilities.system_memory_gb,
        cpu_count=capabilities.cpu_count,
        device_scale=f"{capabilities.device_scale:.4f}",
    )
    return capabilities
