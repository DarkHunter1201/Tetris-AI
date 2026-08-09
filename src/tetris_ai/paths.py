import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / ".runtime"
RUNTIME_PATHS = {
    "cache": RUNTIME_ROOT / "cache",
    "temp": RUNTIME_ROOT / "temp",
    "logs": RUNTIME_ROOT / "logs",
    "checkpoints": RUNTIME_ROOT / "checkpoints",
    "pycache": RUNTIME_ROOT / "pycache",
    "pip_cache": RUNTIME_ROOT / "pip-cache",
    "torch_cache": RUNTIME_ROOT / "torch-cache",
    "cuda_cache": RUNTIME_ROOT / "cuda-cache",
    "inductor_cache": RUNTIME_ROOT / "inductor-cache",
    "triton_cache": RUNTIME_ROOT / "triton-cache",
    "numba_cache": RUNTIME_ROOT / "numba-cache",
    "matplotlib": RUNTIME_ROOT / "matplotlib",
    "data": RUNTIME_ROOT / "data",
}


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def ensure_runtime_directories() -> None:
    ensure_inside_project(RUNTIME_ROOT).mkdir(parents=True, exist_ok=True)
    for path in RUNTIME_PATHS.values():
        ensure_inside_project(path).mkdir(parents=True, exist_ok=True)


def configure_environment() -> None:
    ensure_runtime_directories()
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    values = {
        "TEMP": RUNTIME_PATHS["temp"],
        "TMP": RUNTIME_PATHS["temp"],
        "TMPDIR": RUNTIME_PATHS["temp"],
        "PIP_CACHE_DIR": RUNTIME_PATHS["pip_cache"],
        "PYTHONPYCACHEPREFIX": RUNTIME_PATHS["pycache"],
        "TORCH_HOME": RUNTIME_PATHS["torch_cache"],
        "XDG_CACHE_HOME": RUNTIME_PATHS["cache"],
        "CUDA_CACHE_PATH": RUNTIME_PATHS["cuda_cache"],
        "TORCHINDUCTOR_CACHE_DIR": RUNTIME_PATHS["inductor_cache"],
        "TRITON_CACHE_DIR": RUNTIME_PATHS["triton_cache"],
        "NUMBA_CACHE_DIR": RUNTIME_PATHS["numba_cache"],
        "MPLCONFIGDIR": RUNTIME_PATHS["matplotlib"],
        "PYTHONUSERBASE": RUNTIME_ROOT / "python-user-base",
    }
    for name, path in values.items():
        os.environ[name] = str(ensure_inside_project(path))


def safe_clear_runtime_directory(name: str) -> None:
    target = ensure_inside_project(RUNTIME_PATHS[name])
    target.relative_to(RUNTIME_ROOT.resolve())
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def checkpoint_path() -> Path:
    return ensure_inside_project(RUNTIME_PATHS["checkpoints"] / "training.pt")


def settings_path() -> Path:
    return ensure_inside_project(RUNTIME_PATHS["data"] / "settings.json")
