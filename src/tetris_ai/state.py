import threading
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class TrainingStats:
    generation: int = 0
    current_best_fitness: float = 0.0
    all_time_best_fitness: float = 0.0
    best_score: int = 0
    best_lines: int = 0
    evaluated_agents: int = 0
    status: str = "Starting"
    error: str = ""
    device: str = "CPU"
    genome_revision: int = 0
    paused: bool = False
    rotation_rate: float = 0.0
    vram_limit_mib: int = 0
    total_vram_mib: int = 0
    vram_automatic: bool = True
    population_size: int = 1500


class SharedTrainingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = TrainingStats()
        self.best_genome: torch.Tensor | None = None

    def snapshot(self) -> TrainingStats:
        with self.lock:
            return replace(self.stats)

    def update(self, **values: object) -> None:
        with self.lock:
            self.stats = replace(self.stats, **values)

    def publish_best(self, genome: torch.Tensor, **values: object) -> None:
        immutable = genome.detach().cpu().clone()
        with self.lock:
            revision = self.stats.genome_revision + 1
            self.best_genome = immutable
            self.stats = replace(self.stats, genome_revision=revision, **values)

    def reset_training(self, genome: torch.Tensor, device: str, paused: bool, vram_limit_mib: int, total_vram_mib: int, vram_automatic: bool, population_size: int) -> None:
        immutable = genome.detach().cpu().clone()
        with self.lock:
            revision = self.stats.genome_revision + 1
            self.best_genome = immutable
            status = "Paused · fresh start" if paused else "Training · fresh start"
            self.stats = TrainingStats(status=status, device=device, genome_revision=revision, paused=paused, vram_limit_mib=vram_limit_mib, total_vram_mib=total_vram_mib, vram_automatic=vram_automatic, population_size=population_size)

    def genome_snapshot(self) -> tuple[torch.Tensor | None, int]:
        with self.lock:
            genome = None if self.best_genome is None else self.best_genome.clone()
            return genome, self.stats.genome_revision
