import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .config import AppConfig, CONFIG
from .paths import ensure_inside_project


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    label: str
    category: str
    value_type: str
    description: str
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] = ()


SETTING_DEFINITIONS = (
    SettingDefinition("language", "Language", "LANGUAGE", "choice", "Interface language. Click the field to switch between Russian and English; the choice is saved after applying.", options=("ru", "en")),
    SettingDefinition("board_width", "Board width", "GAME AND NETWORK", "int", "Number of columns in the Tetris board. The network input and output layers are rebuilt when this changes.", 4, 40),
    SettingDefinition("board_height", "Board height", "GAME AND NETWORK", "int", "Number of visible rows in the Tetris board. A taller board increases the network input size and GPU memory use.", 8, 80),
    SettingDefinition("population_size", "Agents per generation", "GAME AND NETWORK", "int", "Number of independently evaluated neural-network agents in one generation. More agents improve search diversity but require more memory and time.", 1),
    SettingDefinition("hidden_sizes", "Hidden layer sizes", "GAME AND NETWORK", "int_tuple", "Comma-separated neuron counts for every hidden layer, for example 96, 64. This controls model capacity and memory use."),
    SettingDefinition("elite_count", "Elite agents", "EVOLUTION", "int", "Number of highest-fitness agents copied unchanged into the next generation.", 1),
    SettingDefinition("parent_pool_size", "Parent pool", "EVOLUTION", "int", "Number of highest-fitness agents eligible to become parents. It must be at least the elite count.", 1),
    SettingDefinition("mutation_rate", "Mutation probability", "EVOLUTION", "float", "Probability that each inherited neural-network parameter is mutated. Range: 0 to 1.", 0, 1),
    SettingDefinition("mutation_scale", "Mutation strength", "EVOLUTION", "float", "Standard deviation of random changes applied to mutated parameters. This acts like evolutionary temperature: higher values explore more aggressively.", 0),
    SettingDefinition("crossover_rate", "Crossover probability", "EVOLUTION", "float", "Probability that a child combines parameters from two parents instead of inheriting one parent. Range: 0 to 1.", 0, 1),
    SettingDefinition("network_policy_weight", "Neural policy influence", "EVOLUTION", "float", "Multiplier for the neural network score when it is combined with the rule-aware placement score.", 0),
    SettingDefinition("seed", "Random seed", "EVOLUTION", "int", "Base seed for initial weights and deterministic piece sequences. The same seed and settings make experiments reproducible.", 0),
    SettingDefinition("rule.completed_lines", "Line reward", "RULE-AWARE PLACEMENT", "float", "Immediate placement bonus for completed lines before the neural policy score is added.", 0),
    SettingDefinition("rule.aggregate_height", "Height penalty", "RULE-AWARE PLACEMENT", "float", "Immediate penalty for the sum of all column heights. Higher values favor flatter, lower stacks.", 0),
    SettingDefinition("rule.holes", "Hole penalty", "RULE-AWARE PLACEMENT", "float", "Immediate penalty for empty cells trapped below blocks. Higher values strongly discourage inaccessible gaps.", 0),
    SettingDefinition("rule.bumpiness", "Bumpiness penalty", "RULE-AWARE PLACEMENT", "float", "Immediate penalty for height differences between adjacent columns.", 0),
    SettingDefinition("rule.maximum_height", "Peak-height penalty", "RULE-AWARE PLACEMENT", "float", "Immediate penalty for the tallest column, reducing the risk of an early game over.", 0),
    SettingDefinition("fitness.completed_lines", "Lines fitness reward", "GENERATION FITNESS", "float", "Final fitness reward per cleared line. This is the strongest signal for learning valid Tetris play.", 0),
    SettingDefinition("fitness.placed_pieces", "Survival reward", "GENERATION FITNESS", "float", "Final fitness reward per successfully placed piece. It rewards survival without replacing the line-clear objective.", 0),
    SettingDefinition("fitness.aggregate_height", "Height fitness penalty", "GENERATION FITNESS", "float", "Final fitness penalty for total column height at the end of an agent's game.", 0),
    SettingDefinition("fitness.holes", "Hole fitness penalty", "GENERATION FITNESS", "float", "Final fitness penalty for holes remaining in the board.", 0),
    SettingDefinition("fitness.bumpiness", "Bumpiness fitness penalty", "GENERATION FITNESS", "float", "Final fitness penalty for an uneven board surface.", 0),
    SettingDefinition("fitness.maximum_height", "Peak fitness penalty", "GENERATION FITNESS", "float", "Final fitness penalty for the highest column.", 0),
    SettingDefinition("fitness.game_over", "Game-over penalty", "GENERATION FITNESS", "float", "Final penalty applied when the agent reaches game over.", 0),
    SettingDefinition("max_pieces_per_game", "Piece limit per game", "EVALUATION", "int", "Maximum number of pieces evaluated for each agent in one generation. Larger values improve long-game measurement but take longer.", 1),
    SettingDefinition("evaluation_chunk_size", "Evaluation chunk", "EVALUATION", "int", "Preferred number of agents processed together. CUDA automatically reduces this value if memory is insufficient.", 1),
    SettingDefinition("checkpoint_interval_generations", "Checkpoint interval", "EVALUATION", "int", "Number of completed generations between automatic checkpoint saves.", 1),
    SettingDefinition("neural_network_vram_limit_mib", "Neural VRAM limit", "CUDA AND MEMORY", "int", "Maximum MiB available to neural-network tensors. Set 0 for automatic detection.", 0),
    SettingDefinition("automatic_vram_fraction", "Automatic VRAM fraction", "CUDA AND MEMORY", "float", "Fraction of total VRAM the automatic limiter may use after reserving memory for the desktop and other applications. Range: 0.05 to 1.", 0.05, 1),
    SettingDefinition("minimum_vram_limit_mib", "Minimum VRAM limit", "CUDA AND MEMORY", "int", "Preferred lower bound in MiB for the automatic CUDA memory budget when the GPU has enough free capacity.", 0),
    SettingDefinition("gpu_reserve_mib", "GPU reserve", "CUDA AND MEMORY", "int", "MiB kept outside the neural budget for the display, driver, and other applications.", 0),
    SettingDefinition("visualization_fps", "Interface FPS", "INTERFACE AND MONITORING", "int", "Maximum interface refresh rate. It does not change training speed directly.", 1, 360),
    SettingDefinition("visualization_drop_interval", "Demo drop interval", "INTERFACE AND MONITORING", "float", "Seconds between visible falling-piece steps in the demonstration board.", 0.001, 10),
    SettingDefinition("hardware_monitor_interval", "Hardware polling interval", "INTERFACE AND MONITORING", "float", "Seconds between CPU, RAM, GPU, temperature, and VRAM measurements. The minimum is 1.5 seconds.", 1.5, 60),
)


@dataclass(frozen=True)
class RuntimeSettings:
    vram_limit_mib: int = 0
    population_size: int = 0
    overrides: dict[str, Any] = field(default_factory=dict)


def setting_values(config: AppConfig) -> dict[str, str]:
    values: dict[str, str] = {}
    for definition in SETTING_DEFINITIONS:
        value = _config_value(config, definition.key)
        if isinstance(value, tuple):
            values[definition.key] = ", ".join(str(item) for item in value)
        else:
            values[definition.key] = str(value)
    return values


def parse_runtime_settings(values: dict[str, str]) -> RuntimeSettings:
    parsed: dict[str, Any] = {}
    for definition in SETTING_DEFINITIONS:
        if definition.key not in values:
            raise ValueError(f"Missing value: {definition.label}")
        parsed[definition.key] = _parse_value(definition, values[definition.key])
    population_size = int(parsed.pop("population_size"))
    vram_limit_mib = int(parsed.pop("neural_network_vram_limit_mib"))
    elite_count = int(parsed["elite_count"])
    parent_pool_size = int(parsed["parent_pool_size"])
    if elite_count > population_size:
        raise ValueError("Elite agents cannot exceed agents per generation")
    if parent_pool_size < elite_count:
        raise ValueError("Parent pool cannot be smaller than elite agents")
    if parent_pool_size > population_size:
        raise ValueError("Parent pool cannot exceed agents per generation")
    return RuntimeSettings(vram_limit_mib, population_size, parsed)


def apply_runtime_settings(config: AppConfig, runtime: RuntimeSettings) -> AppConfig:
    top_level = {item.name for item in fields(AppConfig)}
    changes: dict[str, Any] = {}
    rule_changes: dict[str, Any] = {}
    fitness_changes: dict[str, Any] = {}
    definitions = {definition.key: definition for definition in SETTING_DEFINITIONS}
    for key, raw_value in runtime.overrides.items():
        definition = definitions.get(key)
        if definition is None:
            continue
        value = _coerce_stored_value(definition, raw_value)
        if key.startswith("rule."):
            rule_changes[key.split(".", 1)[1]] = value
        elif key.startswith("fitness."):
            fitness_changes[key.split(".", 1)[1]] = value
        elif key in top_level:
            changes[key] = value
    if runtime.population_size > 0:
        changes["population_size"] = runtime.population_size
        if "elite_count" not in runtime.overrides:
            elite_ratio = config.elite_count / config.population_size
            changes["elite_count"] = max(1, min(runtime.population_size, int(round(runtime.population_size * elite_ratio))))
        if "parent_pool_size" not in runtime.overrides:
            parent_ratio = config.parent_pool_size / config.population_size
            minimum_parent = int(changes.get("elite_count", config.elite_count))
            changes["parent_pool_size"] = max(minimum_parent, min(runtime.population_size, int(round(runtime.population_size * parent_ratio))))
    changes["neural_network_vram_limit_mib"] = max(0, runtime.vram_limit_mib)
    if rule_changes:
        changes["rule_weights"] = replace(config.rule_weights, **rule_changes)
    if fitness_changes:
        changes["fitness_weights"] = replace(config.fitness_weights, **fitness_changes)
    board_width = int(changes.get("board_width", config.board_width))
    changes["output_size"] = board_width * 4
    candidate = replace(config, **changes)
    parse_runtime_settings(setting_values(candidate))
    return candidate


def _config_value(config: AppConfig, key: str) -> Any:
    if key.startswith("rule."):
        return getattr(config.rule_weights, key.split(".", 1)[1])
    if key.startswith("fitness."):
        return getattr(config.fitness_weights, key.split(".", 1)[1])
    return getattr(config, key)


def _parse_value(definition: SettingDefinition, text: str) -> Any:
    stripped = str(text).strip()
    try:
        if definition.value_type == "int":
            value: Any = int(stripped)
        elif definition.value_type == "float":
            value = float(stripped.replace(",", "."))
        elif definition.value_type == "int_tuple":
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
            value = tuple(int(part) for part in parts)
            if not value or len(value) > 8 or any(item < 1 or item > 4096 for item in value):
                raise ValueError
        else:
            value = stripped.lower()
            if value not in definition.options:
                raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Invalid value for {definition.label}") from None
    scalar = float(value) if definition.value_type in ("int", "float") else None
    if scalar is not None and not math.isfinite(scalar):
        raise ValueError(f"Invalid value for {definition.label}")
    if scalar is not None and definition.minimum is not None and scalar < definition.minimum:
        raise ValueError(f"{definition.label} must be at least {definition.minimum:g}")
    if scalar is not None and definition.maximum is not None and scalar > definition.maximum:
        raise ValueError(f"{definition.label} must not exceed {definition.maximum:g}")
    return value


def _coerce_stored_value(definition: SettingDefinition, value: Any) -> Any:
    if definition.value_type == "int_tuple" and isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    return _parse_value(definition, str(value))


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
            overrides = values.get("overrides", {})
            if not isinstance(overrides, dict):
                overrides = {}
            if "hardware_monitor_interval" in overrides:
                try:
                    if float(overrides["hardware_monitor_interval"]) < 1.5:
                        overrides["hardware_monitor_interval"] = 1.5
                except (TypeError, ValueError):
                    overrides.pop("hardware_monitor_interval", None)
            runtime = RuntimeSettings(limit, population_size, overrides)
            try:
                apply_runtime_settings(CONFIG, runtime)
            except (TypeError, ValueError):
                return RuntimeSettings(limit, population_size)
            return runtime
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return RuntimeSettings()

    def save(self, settings: RuntimeSettings) -> None:
        temporary = ensure_inside_project(self.path.with_suffix(self.path.suffix + ".tmp"))
        temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
