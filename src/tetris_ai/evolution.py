from dataclasses import dataclass

import torch

from .network import NetworkSpec, create_population


@dataclass(frozen=True)
class EvolutionSettings:
    population_size: int
    elite_count: int
    parent_pool_size: int
    mutation_rate: float
    mutation_scale: float
    crossover_rate: float


def initial_population(spec: NetworkSpec, settings: EvolutionSettings, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return create_population(spec, settings.population_size, generator)


def evolve_population(population: torch.Tensor, fitness: torch.Tensor, settings: EvolutionSettings, generator: torch.Generator) -> torch.Tensor:
    order = torch.argsort(fitness, descending=True)
    ranked = population[order]
    elites = ranked[:settings.elite_count].clone()
    parents = ranked[:settings.parent_pool_size]
    child_count = settings.population_size - settings.elite_count
    first_indices = torch.randint(settings.parent_pool_size, (child_count,), generator=generator)
    second_indices = torch.randint(settings.parent_pool_size, (child_count,), generator=generator)
    first = parents[first_indices]
    second = parents[second_indices]
    crossover = torch.rand((child_count, 1), generator=generator) < settings.crossover_rate
    gene_mask = torch.rand((child_count, population.shape[1]), generator=generator) < 0.5
    mixed = torch.where(gene_mask, first, second)
    children = torch.where(crossover, mixed, first).clone()
    mutation_mask = torch.rand(children.shape, generator=generator) < settings.mutation_rate
    noise = torch.randn(children.shape, generator=generator) * settings.mutation_scale
    children.add_(noise * mutation_mask)
    return torch.cat((elites, children), dim=0)
