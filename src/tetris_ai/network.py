from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NetworkSpec:
    layer_sizes: tuple[int, ...]

    @property
    def parameter_count(self) -> int:
        total = 0
        for input_size, output_size in zip(self.layer_sizes, self.layer_sizes[1:]):
            total += input_size * output_size + output_size
        return total


def create_population(spec: NetworkSpec, population_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
    genomes = torch.empty((population_size, spec.parameter_count), dtype=torch.float32)
    offset = 0
    for input_size, output_size in zip(spec.layer_sizes, spec.layer_sizes[1:]):
        weight_count = input_size * output_size
        scale = (2.0 / input_size) ** 0.5
        genomes[:, offset:offset + weight_count].normal_(0.0, scale, generator=generator)
        offset += weight_count
        genomes[:, offset:offset + output_size].zero_()
        offset += output_size
    return genomes


def population_forward(genomes: torch.Tensor, states: torch.Tensor, spec: NetworkSpec) -> torch.Tensor:
    values = states
    offset = 0
    layer_pairs = tuple(zip(spec.layer_sizes, spec.layer_sizes[1:]))
    for index, (input_size, output_size) in enumerate(layer_pairs):
        weight_count = input_size * output_size
        weights = genomes[:, offset:offset + weight_count].view(-1, output_size, input_size)
        offset += weight_count
        biases = genomes[:, offset:offset + output_size]
        offset += output_size
        values = torch.bmm(weights, values.unsqueeze(2)).squeeze(2) + biases
        if index < len(layer_pairs) - 1:
            values = torch.relu(values)
    return values


def single_forward(genome: torch.Tensor, state: torch.Tensor, spec: NetworkSpec) -> torch.Tensor:
    return population_forward(genome.unsqueeze(0), state.unsqueeze(0), spec).squeeze(0)
