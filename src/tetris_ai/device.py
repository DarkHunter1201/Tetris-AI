import logging
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class DeviceProfile:
    device: torch.device
    name: str
    total_vram_mib: int
    vram_limit_mib: int
    automatic: bool


def _limit_vram(total_mib: int, requested_mib: int, automatic_fraction: float, minimum_mib: int, reserve_mib: int) -> int:
    maximum = max(1, total_mib - min(reserve_mib, total_mib // 4))
    minimum = min(minimum_mib, maximum)
    if requested_mib <= 0:
        return min(maximum, max(minimum, int(total_mib * automatic_fraction)))
    return min(maximum, max(minimum, requested_mib))


def select_device_profile(requested_mib: int, automatic_fraction: float = 0.85, minimum_mib: int = 1024, reserve_mib: int = 1024) -> DeviceProfile:
    logger = logging.getLogger(__name__)
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        logger.info("CUDA unavailable; using CPU")
        return DeviceProfile(torch.device("cpu"), "CPU", 0, 0, True)
    try:
        candidates = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            candidates.append((properties.total_memory, properties.multi_processor_count, index, properties))
        _, _, index, properties = max(candidates)
        total_mib = int(properties.total_memory / (1024 * 1024))
        limit_mib = _limit_vram(total_mib, requested_mib, automatic_fraction, minimum_mib, reserve_mib)
        torch.cuda.set_device(index)
        torch.cuda.set_per_process_memory_fraction(limit_mib / total_mib, index)
        torch.set_float32_matmul_precision("high")
        device = torch.device(f"cuda:{index}")
        torch.empty(1, device=device)
        logger.info("CUDA enabled on %s with %d MiB limit", properties.name, limit_mib)
        return DeviceProfile(device, properties.name, total_mib, limit_mib, requested_mib <= 0)
    except (RuntimeError, AssertionError) as error:
        logger.warning("CUDA initialization failed; using CPU: %s", error)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return DeviceProfile(torch.device("cpu"), "CPU", 0, 0, True)


def apply_vram_limit(profile: DeviceProfile, requested_mib: int, automatic_fraction: float = 0.85, minimum_mib: int = 1024, reserve_mib: int = 1024) -> DeviceProfile:
    if profile.device.type != "cuda":
        return profile
    limit_mib = _limit_vram(profile.total_vram_mib, requested_mib, automatic_fraction, minimum_mib, reserve_mib)
    torch.cuda.set_per_process_memory_fraction(limit_mib / profile.total_vram_mib, profile.device.index or 0)
    return replace(profile, vram_limit_mib=limit_mib, automatic=requested_mib <= 0)


def select_device(vram_limit_mib: int) -> torch.device:
    return select_device_profile(vram_limit_mib).device


def neural_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.memory_allocated(device) / (1024 * 1024), torch.cuda.memory_reserved(device) / (1024 * 1024)
