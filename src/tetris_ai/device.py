import logging

import torch


def select_device(vram_limit_mib: int) -> torch.device:
    logger = logging.getLogger(__name__)
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable; using CPU")
        return torch.device("cpu")
    try:
        properties = torch.cuda.get_device_properties(0)
        total_mib = properties.total_memory / (1024 * 1024)
        fraction = min(1.0, vram_limit_mib / total_mib)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        device = torch.device("cuda:0")
        torch.empty(1, device=device)
        logger.info("CUDA enabled on %s with %.3f memory fraction", properties.name, fraction)
        return device
    except (RuntimeError, AssertionError) as error:
        logger.warning("CUDA initialization failed; using CPU: %s", error)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return torch.device("cpu")


def neural_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.memory_allocated(device) / (1024 * 1024), torch.cuda.memory_reserved(device) / (1024 * 1024)
