import shutil
import unittest
import uuid

import torch

from tetris_ai.checkpoint import CheckpointManager, TrainingCheckpoint
from tetris_ai.network import NetworkSpec
from tetris_ai.paths import RUNTIME_PATHS, ensure_inside_project


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = ensure_inside_project(RUNTIME_PATHS["temp"] / f"checkpoint-test-{uuid.uuid4().hex}")
        self.directory.mkdir(parents=True)
        self.manager = CheckpointManager(self.directory / "training.pt")

    def tearDown(self):
        ensure_inside_project(self.directory)
        shutil.rmtree(self.directory)

    def test_checkpoint_round_trip(self):
        population = torch.randn(4, 12)
        checkpoint = TrainingCheckpoint(
            generation=8,
            population=population,
            best_genome=population[2],
            best_fitness=123.5,
            best_score=400,
            best_lines=3,
            network_spec=NetworkSpec((3, 2)),
            evolution_settings={"elite_count": 1},
            random_state=torch.Generator().get_state(),
        )
        self.manager.save(checkpoint)
        loaded = self.manager.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.generation, 8)
        self.assertTrue(torch.equal(loaded.population, population))
        self.assertEqual(loaded.best_lines, 3)
        self.assertFalse(self.manager.path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
