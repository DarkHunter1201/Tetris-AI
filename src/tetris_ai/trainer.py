import logging
import threading

import torch

from .checkpoint import CheckpointManager, TrainingCheckpoint
from .config import AppConfig
from .device import select_device
from .engine import TetrisGame
from .evolution import EvolutionSettings, evolve_population, initial_population
from .network import NetworkSpec, population_forward
from .state import SharedTrainingState


class Trainer:
    def __init__(self, config: AppConfig, checkpoint_manager: CheckpointManager, shared: SharedTrainingState):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.shared = shared
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        state_size = config.board_width * config.board_height + 7 + config.board_width + 3
        self.spec = NetworkSpec((state_size, *config.hidden_sizes, config.output_size))
        self.settings = EvolutionSettings(
            population_size=config.population_size,
            elite_count=config.elite_count,
            parent_pool_size=config.parent_pool_size,
            mutation_rate=config.mutation_rate,
            mutation_scale=config.mutation_scale,
            crossover_rate=config.crossover_rate,
        )
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)
        self.device = torch.device("cpu")
        self.population: torch.Tensor | None = None
        self.generation = 0
        self.best_genome: torch.Tensor | None = None
        self.best_fitness = float("-inf")
        self.best_score = 0
        self.best_lines = 0

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name="training-worker", daemon=True)
        self.thread.start()

    def request_stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)

    def _load_or_create(self) -> None:
        checkpoint = self.checkpoint_manager.load()
        if checkpoint is None:
            self.population = initial_population(self.spec, self.settings, self.config.seed)
            self.best_genome = self.population[0].clone()
            self.shared.publish_best(self.best_genome, status="Training")
            return
        if checkpoint.network_spec != self.spec:
            raise ValueError("Checkpoint network architecture does not match version 1.0.0 alpha")
        if checkpoint.population.shape[0] != self.config.population_size:
            raise ValueError("Checkpoint population size does not equal 1500")
        self.population = checkpoint.population
        self.generation = checkpoint.generation
        self.best_genome = checkpoint.best_genome
        self.best_fitness = checkpoint.best_fitness
        self.best_score = checkpoint.best_score
        self.best_lines = checkpoint.best_lines
        self.generator.set_state(checkpoint.random_state)
        self.shared.publish_best(
            self.best_genome,
            generation=self.generation,
            all_time_best_fitness=self.best_fitness,
            best_score=self.best_score,
            best_lines=self.best_lines,
            status="Resumed",
        )

    def run(self, generation_limit: int | None = None) -> None:
        logger = logging.getLogger(__name__)
        try:
            self.device = select_device(self.config.neural_network_vram_limit_mib)
            self.shared.update(device=self.device_label(), status="Loading checkpoint")
            self._load_or_create()
            completed = 0
            while not self.stop_event.is_set() and (generation_limit is None or completed < generation_limit):
                self.generation += 1
                fitness, scores, lines = self.evaluate_generation()
                if self.stop_event.is_set():
                    break
                best_index = int(torch.argmax(fitness).item())
                current_fitness = float(fitness[best_index].item())
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_genome = self.population[best_index].clone()
                    self.best_score = scores[best_index]
                    self.best_lines = lines[best_index]
                    self.shared.publish_best(
                        self.best_genome,
                        generation=self.generation,
                        current_best_fitness=current_fitness,
                        all_time_best_fitness=self.best_fitness,
                        best_score=self.best_score,
                        best_lines=self.best_lines,
                        status="Evolving",
                    )
                    self.save_checkpoint()
                else:
                    self.shared.update(
                        generation=self.generation,
                        current_best_fitness=current_fitness,
                        all_time_best_fitness=self.best_fitness,
                        status="Evolving",
                    )
                self.population = evolve_population(self.population, fitness, self.settings, self.generator)
                if self.generation % self.config.checkpoint_interval_generations == 0:
                    self.save_checkpoint()
                completed += 1
            self.shared.update(status="Stopped" if self.stop_event.is_set() else "Complete")
        except Exception as error:
            logger.exception("Training worker failed")
            self.shared.update(status="Error", error=f"{type(error).__name__}: {error}")
        finally:
            if self.population is not None and self.best_genome is not None:
                try:
                    self.save_checkpoint()
                except Exception:
                    logger.exception("Final checkpoint save failed")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def evaluate_generation(self) -> tuple[torch.Tensor, list[int], list[int]]:
        if self.population is None:
            raise RuntimeError("Population is unavailable")
        games = [
            TetrisGame(self.config.board_width, self.config.board_height, self.config.seed + self.generation * self.config.population_size + index)
            for index in range(self.config.population_size)
        ]
        chunk_size = self.config.evaluation_chunk_size
        for piece_round in range(self.config.max_pieces_per_game):
            if self.stop_event.is_set():
                break
            active = [index for index, game in enumerate(games) if not game.game_over]
            if not active:
                break
            self.shared.update(
                generation=self.generation,
                evaluated_agents=self.config.population_size - len(active),
                status=f"Training · piece {piece_round + 1}",
            )
            cursor = 0
            while cursor < len(active):
                if self.stop_event.is_set():
                    break
                indices = active[cursor:cursor + chunk_size]
                try:
                    self._evaluate_chunk(indices, games)
                    cursor += len(indices)
                except torch.OutOfMemoryError:
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    if chunk_size <= 8:
                        raise
                    chunk_size = max(8, chunk_size // 2)
        fitness = torch.tensor([game.fitness() for game in games], dtype=torch.float32)
        scores = [game.score for game in games]
        lines = [game.lines for game in games]
        return fitness, scores, lines

    def _evaluate_chunk(self, indices: list[int], games: list[TetrisGame]) -> None:
        if self.population is None:
            raise RuntimeError("Population is unavailable")
        states = torch.tensor([games[index].state_vector() for index in indices], dtype=torch.float32, device=self.device)
        genomes = self.population[indices].to(self.device)
        with torch.inference_mode():
            logits = population_forward(genomes, states, self.spec).cpu()
        for row, index in enumerate(indices):
            game = games[index]
            legal = game.legal_actions()
            if not legal:
                game.game_over = True
                continue
            action = max(legal, key=lambda candidate: float(logits[row, candidate]))
            game.apply_action(action)

    def save_checkpoint(self) -> None:
        if self.population is None or self.best_genome is None:
            return
        checkpoint = TrainingCheckpoint(
            generation=self.generation,
            population=self.population,
            best_genome=self.best_genome,
            best_fitness=self.best_fitness,
            best_score=self.best_score,
            best_lines=self.best_lines,
            network_spec=self.spec,
            evolution_settings=self.settings.__dict__.copy(),
            random_state=self.generator.get_state(),
        )
        self.checkpoint_manager.save(checkpoint)

    def device_label(self) -> str:
        if self.device.type == "cuda":
            return f"CUDA · {torch.cuda.get_device_name(self.device)}"
        return "CPU"
