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

    def reset_training(self, genome: torch.Tensor, device: str) -> None:
        immutable = genome.detach().cpu().clone()
        with self.lock:
            revision = self.stats.genome_revision + 1
            self.best_genome = immutable
            self.stats = TrainingStats(status="Training · fresh start", device=device, genome_revision=revision)

    def genome_snapshot(self) -> tuple[torch.Tensor | None, int]:
        with self.lock:
            genome = None if self.best_genome is None else self.best_genome.clone()
            return genome, self.stats.genome_revision
