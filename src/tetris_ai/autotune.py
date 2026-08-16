import json
import os
import platform
import subprocess
from dataclasses import dataclass

import psutil
import torch

from .config import CONFIG
from .settings import RuntimeSettings, parse_runtime_settings, setting_values
from .translations import normalize_language


@dataclass(frozen=True)
class HardwareProfile:
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_mib: int
    ram_type: str
    ram_speed_mhz: int
    cuda_available: bool
    gpu_name: str
    total_vram_mib: int
    multiprocessor_count: int
    compute_capability: tuple[int, int]


RAM_TYPES = {
    20: "DDR",
    21: "DDR2",
    24: "DDR3",
    26: "DDR4",
    27: "LPDDR",
    28: "LPDDR2",
    29: "LPDDR3",
    30: "LPDDR4",
    34: "DDR5",
    35: "LPDDR5",
}


def detect_hardware_profile() -> HardwareProfile:
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    ram_mib = int(psutil.virtual_memory().total / (1024 * 1024))
    cpu_name = _windows_cpu_name() or platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
    ram_type, ram_speed = _windows_memory_details()
    gpu_name = ""
    total_vram_mib = 0
    multiprocessors = 0
    capability = (0, 0)
    cuda_available = False
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            candidates = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                candidates.append((properties.total_memory, properties.multi_processor_count, index, properties))
            _, _, index, properties = max(candidates)
            gpu_name = str(properties.name)
            total_vram_mib = int(properties.total_memory / (1024 * 1024))
            multiprocessors = int(properties.multi_processor_count)
            capability = tuple(int(item) for item in torch.cuda.get_device_capability(index))
            cuda_available = True
    except (AssertionError, RuntimeError):
        cuda_available = False
    return HardwareProfile(cpu_name, physical, logical, ram_mib, ram_type, ram_speed, cuda_available, gpu_name, total_vram_mib, multiprocessors, capability)


def recommend_runtime_settings(profile: HardwareProfile, language: str) -> RuntimeSettings:
    values = setting_values(CONFIG)
    values["language"] = normalize_language(language)
    ram_gib = profile.ram_mib / 1024
    if ram_gib < 8:
        ram_cap = 400
    elif ram_gib < 16:
        ram_cap = 900
    elif ram_gib < 32:
        ram_cap = 1800
    elif ram_gib < 64:
        ram_cap = 4000
    else:
        ram_cap = 8000
    if profile.cuda_available:
        vram_gib = profile.total_vram_mib / 1024
        if vram_gib >= 20:
            population, hidden, chunk, pieces = 6000, "160, 96", 512, 750
        elif vram_gib >= 12:
            population, hidden, chunk, pieces = 3800, "128, 96", 384, 650
        elif vram_gib >= 8:
            population, hidden, chunk, pieces = 2500, "112, 80", 256, 550
        elif vram_gib >= 6:
            population, hidden, chunk, pieces = 1700, "96, 64", 192, 500
        elif vram_gib >= 4:
            population, hidden, chunk, pieces = 1000, "80, 48", 128, 400
        else:
            population, hidden, chunk, pieces = 600, "64, 48", 96, 300
        gpu_factor = 1.15 if profile.multiprocessor_count >= 60 else 1.0 if profile.multiprocessor_count >= 30 else 0.8
        gpu_model = profile.gpu_name.upper()
        gpu_model_factor = 1.05 if "RTX 50" in gpu_model or "RTX 40" in gpu_model else 0.85 if "GTX" in gpu_model else 1.0
        cpu_factor = 1.1 if profile.physical_cores >= 12 else 1.0 if profile.physical_cores >= 6 else 0.85
        cpu_model = profile.cpu_name.upper()
        cpu_model_factor = 1.08 if "RYZEN 9" in cpu_model or "I9-" in cpu_model else 1.04 if "RYZEN 7" in cpu_model or "I7-" in cpu_model else 1.0
        memory_factor = 1.08 if profile.ram_type in ("DDR5", "LPDDR5") else 1.0
        memory_speed_factor = 1.05 if profile.ram_speed_mhz >= 5000 else 1.02 if profile.ram_speed_mhz >= 3200 else 1.0
        population = int(round(population * gpu_factor * gpu_model_factor * cpu_factor * cpu_model_factor * memory_factor * memory_speed_factor / 100) * 100)
        population = max(200, min(population, ram_cap))
        values["automatic_vram_fraction"] = "0.88" if vram_gib >= 12 else "0.82"
        values["gpu_reserve_mib"] = "1536" if vram_gib >= 12 else "1024"
        values["minimum_vram_limit_mib"] = "2048" if vram_gib >= 8 else "1024"
    else:
        memory_factor = 1.1 if profile.ram_type in ("DDR5", "LPDDR5") else 1.0
        memory_speed_factor = 1.05 if profile.ram_speed_mhz >= 5000 else 1.02 if profile.ram_speed_mhz >= 3200 else 1.0
        cpu_model = profile.cpu_name.upper()
        cpu_model_factor = 1.1 if "RYZEN 9" in cpu_model or "I9-" in cpu_model else 1.05 if "RYZEN 7" in cpu_model or "I7-" in cpu_model else 1.0
        population = int(round(max(200, profile.physical_cores * 90 * memory_factor * memory_speed_factor * cpu_model_factor) / 50) * 50)
        population = min(population, ram_cap, 1800)
        hidden = "64, 48" if profile.physical_cores >= 6 else "48, 32"
        chunk = max(16, min(128, profile.logical_cores * 8))
        pieces = 350 if profile.physical_cores >= 8 else 250
        values["automatic_vram_fraction"] = "0.85"
        values["gpu_reserve_mib"] = "1024"
        values["minimum_vram_limit_mib"] = "1024"
    elite = max(1, int(round(population * 0.02)))
    parents = max(elite, int(round(population * 0.1)))
    values["population_size"] = str(population)
    values["hidden_sizes"] = hidden
    values["elite_count"] = str(elite)
    values["parent_pool_size"] = str(parents)
    values["evaluation_chunk_size"] = str(min(population, chunk))
    values["max_pieces_per_game"] = str(pieces)
    values["neural_network_vram_limit_mib"] = "0"
    values["hardware_monitor_interval"] = "1.5"
    return parse_runtime_settings(values)


def profile_summary(profile: HardwareProfile, language: str) -> str:
    ram_gib = max(1, int(round(profile.ram_mib / 1024)))
    ram = f"{ram_gib} GiB {profile.ram_type}"
    if profile.ram_speed_mhz:
        ram += f" {profile.ram_speed_mhz} MHz"
    vram_gib = max(1, int(round(profile.total_vram_mib / 1024)))
    gpu = f"{profile.gpu_name}, {vram_gib} GiB VRAM" if profile.cuda_available else "CPU mode"
    if normalize_language(language) == "ru":
        gpu = f"{profile.gpu_name}, {vram_gib} GiB VRAM" if profile.cuda_available else "режим CPU"
        return f"{gpu}; {ram}; {profile.physical_cores} ядер CPU"
    return f"{gpu}; {ram}; {profile.physical_cores} CPU cores"


def automatic_runtime_settings(language: str) -> tuple[RuntimeSettings, str]:
    profile = detect_hardware_profile()
    return recommend_runtime_settings(profile, language), profile_summary(profile, language)


def _windows_memory_details() -> tuple[str, int]:
    if os.name != "nt":
        return "Unknown", 0
    command = "$m=Get-CimInstance Win32_PhysicalMemory | Select-Object SMBIOSMemoryType,ConfiguredClockSpeed; $m | ConvertTo-Json -Compress"
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = json.loads(result.stdout.strip())
        modules = payload if isinstance(payload, list) else [payload]
        type_codes = [int(item.get("SMBIOSMemoryType", 0)) for item in modules if isinstance(item, dict)]
        speeds = [int(item.get("ConfiguredClockSpeed", 0)) for item in modules if isinstance(item, dict)]
        types = sorted({RAM_TYPES.get(code, "Unknown") for code in type_codes})
        ram_type = "/".join(item for item in types if item != "Unknown") or "Unknown"
        speed = int(round(sum(value for value in speeds if value > 0) / max(1, len([value for value in speeds if value > 0]))))
        return ram_type, speed
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError, TypeError, ValueError):
        return "Unknown", 0


def _windows_cpu_name() -> str:
    if os.name != "nt":
        return ""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
