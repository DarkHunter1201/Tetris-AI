import threading
import time
from dataclasses import dataclass, replace

import psutil
import torch

from .device import neural_memory_mib


@dataclass(frozen=True)
class HardwareStats:
    cpu_load: float = 0.0
    ram_load: float = 0.0
    gpu_load: float | None = None
    gpu_temperature: int | None = None
    neural_allocated_mib: float = 0.0
    neural_reserved_mib: float = 0.0
    total_vram_used_mib: float | None = None
    total_vram_mib: float | None = None
    gpu_name: str | None = None


class HardwareMonitor:
    def __init__(self, device: torch.device, interval: float):
        self.device = device
        self.interval = interval
        self.lock = threading.Lock()
        self.stats = HardwareStats()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.nvml = None
        self.handle = None

    def start(self) -> None:
        self._initialize_nvml()
        self.thread = threading.Thread(target=self._run, name="hardware-monitor", daemon=True)
        self.thread.start()

    def _initialize_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            index = self.device.index or 0 if self.device.type == "cuda" else 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.nvml = pynvml
        except Exception:
            self.nvml = None
            self.handle = None

    def _run(self) -> None:
        psutil.cpu_percent(None)
        while not self.stop_event.wait(self.interval):
            self.poll()

    def poll(self) -> None:
        allocated, reserved = neural_memory_mib(self.device)
        updated = HardwareStats(
            cpu_load=psutil.cpu_percent(None),
            ram_load=psutil.virtual_memory().percent,
            neural_allocated_mib=allocated,
            neural_reserved_mib=reserved,
        )
        if self.nvml is not None and self.handle is not None:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                name = self.nvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                updated = replace(
                    updated,
                    gpu_load=float(utilization.gpu),
                    gpu_temperature=int(self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU)),
                    total_vram_used_mib=memory.used / (1024 * 1024),
                    total_vram_mib=memory.total / (1024 * 1024),
                    gpu_name=str(name),
                )
            except Exception:
                self.nvml = None
                self.handle = None
        with self.lock:
            self.stats = updated

    def snapshot(self) -> HardwareStats:
        with self.lock:
            return replace(self.stats)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval * 2))
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                self.nvml = None
