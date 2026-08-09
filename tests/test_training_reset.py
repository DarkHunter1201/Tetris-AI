import shutil
import unittest
import uuid
from dataclasses import replace

import torch

from tetris_ai.checkpoint import CheckpointManager, TrainingCheckpoint
from tetris_ai.config import CONFIG
from tetris_ai.paths import RUNTIME_PATHS, ensure_inside_project
from tetris_ai.state import SharedTrainingState
from tetris_ai.trainer import Trainer


class TrainingResetTests(unittest.TestCase):
    def setUp(self):
        self.directory = ensure_inside_project(RUNTIME_PATHS["temp"] / f"reset-test-{uuid.uuid4().hex}")
        self.directory.mkdir(parents=True)
        self.manager = CheckpointManager(self.directory / "training.pt")
        self.shared = SharedTrainingState()
        config = replace(CONFIG, population_size=8, elite_count=2, parent_pool_size=4, hidden_sizes=(8,))
        self.trainer = Trainer(config, self.manager, self.shared)

    def tearDown(self):
        ensure_inside_project(self.directory)
        shutil.rmtree(self.directory)

    def test_reset_deletes_experience_and_restarts_population(self):
        parameter_count = self.trainer.spec.parameter_count
        old_population = torch.ones((8, parameter_count), dtype=torch.float32)
        self.trainer.population = old_population
        self.trainer.best_genome = old_population[0].clone()
        self.trainer.generation = 12
        self.trainer.best_fitness = 900.0
        self.trainer.best_score = 700
        self.trainer.best_lines = 5
        checkpoint = TrainingCheckpoint(
            generation=12,
            population=old_population,
            best_genome=old_population[0],
            best_fitness=900.0,
            best_score=700,
            best_lines=5,
            network_spec=self.trainer.spec,
            evolution_settings=self.trainer.settings.__dict__.copy(),
            random_state=self.trainer.generator.get_state(),
        )
        self.manager.save(checkpoint)
        self.trainer._reset_training()
        stats = self.shared.snapshot()
        self.assertFalse(self.manager.path.exists())
        self.assertEqual(self.trainer.generation, 0)
        self.assertEqual(self.trainer.best_fitness, float("-inf"))
        self.assertEqual(self.trainer.best_score, 0)
        self.assertEqual(self.trainer.best_lines, 0)
        self.assertEqual(self.trainer.population.shape[0], 8)
        self.assertFalse(torch.equal(self.trainer.population, old_population))
        self.assertEqual(stats.generation, 0)
        self.assertEqual(stats.all_time_best_fitness, 0.0)
        self.assertEqual(stats.best_score, 0)
        self.assertEqual(stats.best_lines, 0)
        self.assertEqual(stats.status, "Training · fresh start")

    def test_reset_request_sets_worker_event(self):
        self.trainer.request_reset()
        self.assertTrue(self.trainer.reset_event.is_set())
        self.assertEqual(self.shared.snapshot().status, "Reset requested")


if __name__ == "__main__":
    unittest.main()
