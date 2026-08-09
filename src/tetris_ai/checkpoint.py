import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .network import NetworkSpec


@dataclass
class TrainingCheckpoint:
    generation: int
    population: torch.Tensor
    best_genome: torch.Tensor
    best_fitness: float
    best_score: int
    best_lines: int
    network_spec: NetworkSpec
    evolution_settings: dict[str, Any]
    random_state: torch.Tensor


class CheckpointManager:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: TrainingCheckpoint) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "format_version": 1,
            "generation": checkpoint.generation,
            "population": checkpoint.population.detach().cpu(),
            "best_genome": checkpoint.best_genome.detach().cpu(),
            "best_fitness": checkpoint.best_fitness,
            "best_score": checkpoint.best_score,
            "best_lines": checkpoint.best_lines,
            "network_spec": checkpoint.network_spec.layer_sizes,
            "evolution_settings": checkpoint.evolution_settings,
            "random_state": checkpoint.random_state.cpu(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, self.path)

    def load(self) -> TrainingCheckpoint | None:
        if not self.path.exists():
            return None
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        return TrainingCheckpoint(
            generation=int(payload["generation"]),
            population=payload["population"].float(),
            best_genome=payload["best_genome"].float(),
            best_fitness=float(payload["best_fitness"]),
            best_score=int(payload["best_score"]),
            best_lines=int(payload["best_lines"]),
            network_spec=NetworkSpec(tuple(payload["network_spec"])),
            evolution_settings=dict(payload["evolution_settings"]),
            random_state=payload["random_state"],
        )
