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


def evolve_population(population: torch.Tensor, fitness: torch.Tensor, settings: EvolutionSettings, generator: torch.Generator, chunk_size: int | None = None) -> torch.Tensor:
    order = torch.argsort(fitness, descending=True)
    elites = population[order[:settings.elite_count]].clone()
    parents = population[order[:settings.parent_pool_size]]
    child_count = settings.population_size - settings.elite_count
    output = torch.empty_like(population)
    output[:settings.elite_count] = elites
    if child_count == 0:
        return output
    batch = child_count if chunk_size is None else max(1, min(child_count, chunk_size))
    for start in range(0, child_count, batch):
        count = min(batch, child_count - start)
        first_indices = torch.randint(settings.parent_pool_size, (count,), generator=generator, device=population.device)
        second_indices = torch.randint(settings.parent_pool_size, (count,), generator=generator, device=population.device)
        first = parents[first_indices]
        second = parents[second_indices]
        crossover = torch.rand((count, 1), generator=generator, device=population.device) < settings.crossover_rate
        gene_mask = torch.rand((count, population.shape[1]), generator=generator, device=population.device) < 0.5
        children = torch.where(crossover, torch.where(gene_mask, first, second), first)
        mutation_mask = torch.rand(children.shape, generator=generator, device=population.device) < settings.mutation_rate
        noise = torch.randn(children.shape, generator=generator, device=population.device) * settings.mutation_scale
        output[settings.elite_count + start:settings.elite_count + start + count] = children + noise * mutation_mask
    return output
