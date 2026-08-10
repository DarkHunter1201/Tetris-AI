from dataclasses import dataclass

from .engine import FitnessWeights, RuleWeights


@dataclass(frozen=True)
class AppConfig:
    version: str = "1.2.0 beta"
    board_width: int = 10
    board_height: int = 20
    population_size: int = 1500
    neural_network_vram_limit_mib: int = 0
    automatic_vram_fraction: float = 0.85
    minimum_vram_limit_mib: int = 1024
    gpu_reserve_mib: int = 1024
    hidden_sizes: tuple[int, ...] = (96, 64)
    output_size: int = 40
    elite_count: int = 30
    parent_pool_size: int = 150
    mutation_rate: float = 0.08
    mutation_scale: float = 0.12
    crossover_rate: float = 0.55
    visualization_fps: int = 60
    visualization_drop_interval: float = 0.055
    hardware_monitor_interval: float = 1.0
    checkpoint_interval_generations: int = 1
    evaluation_chunk_size: int = 192
    max_pieces_per_game: int = 500
    rule_weights: RuleWeights = RuleWeights()
    fitness_weights: FitnessWeights = FitnessWeights()
    network_policy_weight: float = 1.5
    seed: int = 94721


CONFIG = AppConfig()
