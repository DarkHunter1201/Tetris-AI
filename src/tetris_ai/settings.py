import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import ensure_inside_project


@dataclass(frozen=True)
class RuntimeSettings:
    vram_limit_mib: int = 0
    population_size: int = 0


class SettingsManager:
    def __init__(self, path: Path):
        self.path = ensure_inside_project(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RuntimeSettings:
        if not self.path.exists():
            return RuntimeSettings()
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            limit = max(0, int(values.get("vram_limit_mib", 0)))
            population_size = max(0, int(values.get("population_size", 0)))
            return RuntimeSettings(vram_limit_mib=limit, population_size=population_size)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return RuntimeSettings()

    def save(self, settings: RuntimeSettings) -> None:
        temporary = ensure_inside_project(self.path.with_suffix(self.path.suffix + ".tmp"))
        temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
