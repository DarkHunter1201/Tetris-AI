import shutil
import time
import unittest
import uuid
from dataclasses import replace
from unittest.mock import patch

import torch

from tetris_ai.checkpoint import CheckpointManager, TrainingCheckpoint
from tetris_ai.config import CONFIG
from tetris_ai.device import DeviceProfile
from tetris_ai.paths import RUNTIME_PATHS, ensure_inside_project
from tetris_ai.state import SharedTrainingState
from tetris_ai.trainer import Trainer


class TrainingResetTests(unittest.TestCase):
    def setUp(self):
        self.directory = ensure_inside_project(RUNTIME_PATHS["temp"] / f"reset-test-{uuid.uuid4().hex}")
        self.directory.mkdir(parents=True)
        self.manager = CheckpointManager(self.directory / "training.pt")
        self.shared = SharedTrainingState()
        config = replace(CONFIG, population_size=8, elite_count=2, parent_pool_size=4, hidden_sizes=(8,), max_pieces_per_game=2, evaluation_chunk_size=4)
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
        temporary = self.manager.path.with_suffix(".pt.tmp")
        temporary.touch()
        old_genome = self.trainer.best_genome.clone()
        with patch("tetris_ai.trainer.secrets.randbits", return_value=123456):
            self.trainer._reset_training()
        stats = self.shared.snapshot()
        shared_genome, _ = self.shared.genome_snapshot()
        self.assertFalse(self.manager.path.exists())
        self.assertFalse(temporary.exists())
        self.assertEqual(self.trainer.generation, 0)
        self.assertEqual(self.trainer.best_fitness, float("-inf"))
        self.assertEqual(self.trainer.best_score, 0)
        self.assertEqual(self.trainer.best_lines, 0)
        self.assertEqual(self.trainer.population.shape[0], 8)
        self.assertFalse(torch.equal(self.trainer.population, old_population))
        self.assertFalse(torch.equal(self.trainer.best_genome, old_genome))
        self.assertTrue(torch.equal(shared_genome, self.trainer.best_genome.cpu()))
        self.assertEqual(stats.generation, 0)
        self.assertEqual(stats.all_time_best_fitness, 0.0)
        self.assertEqual(stats.best_score, 0)
        self.assertEqual(stats.best_lines, 0)
        self.assertEqual(stats.evaluated_agents, 0)
        self.assertEqual(stats.rotation_rate, 0.0)
        self.assertEqual(stats.status, "Training · fresh start")

    def test_reset_request_sets_worker_event(self):
        self.trainer.request_reset()
        self.assertTrue(self.trainer.reset_event.is_set())
        self.assertTrue(self.trainer.pause_event.is_set())
        self.assertTrue(self.shared.snapshot().paused)
        self.assertEqual(self.shared.snapshot().status, "Reset requested")

    def test_pause_and_resume_requests(self):
        self.trainer.request_pause()
        self.assertTrue(self.trainer.pause_event.is_set())
        self.assertTrue(self.shared.snapshot().paused)
        self.trainer.request_resume()
        self.assertFalse(self.trainer.pause_event.is_set())
        self.assertFalse(self.shared.snapshot().paused)

    def test_agent_count_request_resets_with_new_population(self):
        self.trainer.request_population_size(3)
        self.assertTrue(self.trainer.pause_event.is_set())
        self.assertTrue(self.trainer.reset_event.is_set())
        self.trainer._apply_pending_population_size()
        with patch("tetris_ai.trainer.secrets.randbits", return_value=654321):
            self.trainer._reset_training()
        self.assertEqual(self.trainer.population_size, 3)
        self.assertEqual(self.trainer.population.shape[0], 3)
        self.assertEqual(self.trainer.settings.population_size, 3)
        self.assertEqual(self.shared.snapshot().population_size, 3)

    def test_reset_stays_paused_until_start(self):
        cpu = DeviceProfile(torch.device("cpu"), "CPU", 0, 0, True)
        self.trainer.pause_event.set()
        with patch("tetris_ai.trainer.select_device_profile", return_value=cpu):
            self.trainer.start(generation_limit=1)
            deadline = time.monotonic() + 3.0
            while self.shared.snapshot().status != "Paused" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.trainer.request_reset()
            while self.shared.snapshot().status != "Paused · fresh start" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.trainer.generation, 0)
            time.sleep(0.05)
            self.assertEqual(self.trainer.generation, 0)
            self.assertTrue(self.trainer.thread.is_alive())
            self.trainer.request_resume()
            self.trainer.join(3.0)
        self.assertFalse(self.trainer.thread.is_alive())
        self.assertEqual(self.trainer.generation, 1)


if __name__ == "__main__":
    unittest.main()
