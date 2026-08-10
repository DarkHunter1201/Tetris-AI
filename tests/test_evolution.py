import unittest

import torch

from tetris_ai.evolution import EvolutionSettings, evolve_population, initial_population
from tetris_ai.network import NetworkSpec


class EvolutionTests(unittest.TestCase):
    def settings(self):
        return EvolutionSettings(1500, 30, 150, 0.08, 0.12, 0.55)

    def test_population_contains_exactly_1500_agents(self):
        population = initial_population(NetworkSpec((4, 3, 2)), self.settings(), 1)
        self.assertEqual(population.shape[0], 1500)

    def test_elites_are_preserved_exactly(self):
        settings = self.settings()
        population = torch.arange(1500 * 6, dtype=torch.float32).view(1500, 6)
        fitness = torch.arange(1500, dtype=torch.float32)
        generator = torch.Generator().manual_seed(7)
        evolved = evolve_population(population, fitness, settings, generator)
        expected = population[torch.argsort(fitness, descending=True)[:settings.elite_count]]
        self.assertTrue(torch.equal(evolved[:settings.elite_count], expected))

    def test_single_agent_population_is_supported(self):
        settings = EvolutionSettings(1, 1, 1, 0.08, 0.12, 0.55)
        population = initial_population(NetworkSpec((4, 3, 2)), settings, 5)
        evolved = evolve_population(population, torch.tensor([1.0]), settings, torch.Generator().manual_seed(6))
        self.assertEqual(evolved.shape[0], 1)
        self.assertTrue(torch.equal(evolved, population))
        self.assertEqual(evolved.shape, population.shape)


if __name__ == "__main__":
    unittest.main()
