import logging
import secrets
import threading

import torch

from .batch_engine import BatchedTetris
from .checkpoint import CheckpointManager, TrainingCheckpoint
from .config import AppConfig
from .device import DeviceProfile, apply_vram_limit, select_device_profile
from .engine import piece_sequence
from .evolution import EvolutionSettings, evolve_population, initial_population
from .network import NetworkSpec, population_forward
from .settings import RuntimeSettings, SettingsManager
from .state import SharedTrainingState


class Trainer:
    def __init__(self, config: AppConfig, checkpoint_manager: CheckpointManager, shared: SharedTrainingState, settings_manager: SettingsManager | None = None):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.shared = shared
        self.settings_manager = settings_manager
        self.runtime_settings = settings_manager.load() if settings_manager is not None else RuntimeSettings(config.neural_network_vram_limit_mib, config.population_size)
        self.population_size = self.runtime_settings.population_size or config.population_size
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.pause_event = threading.Event()
        self.control_event = threading.Event()
        self.control_lock = threading.Lock()
        self.pending_vram_limit_mib: int | None = None
        self.pending_population_size: int | None = None
        self.thread: threading.Thread | None = None
        state_size = config.board_width * config.board_height + 7 + config.board_width + 3
        self.spec = NetworkSpec((state_size, *config.hidden_sizes, config.output_size))
        if config.output_size != config.board_width * 4:
            raise ValueError("Network output size must equal four rotations times board width")
        self.settings = self._evolution_settings(self.population_size)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)
        self.profile = DeviceProfile(torch.device("cpu"), "CPU", 0, 0, True)
        self.device = self.profile.device
        self.population: torch.Tensor | None = None
        self.generation = 0
        self.best_genome: torch.Tensor | None = None
        self.best_fitness = float("-inf")
        self.best_score = 0
        self.best_lines = 0
        self.chunk_size = config.evaluation_chunk_size

    def start(self, generation_limit: int | None = None) -> None:
        self.thread = threading.Thread(target=self.run, args=(generation_limit,), name="training-worker", daemon=True)
        self.thread.start()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.control_event.set()

    def request_pause(self) -> None:
        if not self.stop_event.is_set():
            self.pause_event.set()
            self.shared.update(status="Pausing", paused=True)
            self.control_event.set()

    def request_resume(self) -> None:
        if not self.stop_event.is_set():
            self.pause_event.clear()
            self.shared.update(status="Resuming", paused=False)
            self.control_event.set()

    def request_reset(self) -> None:
        if not self.stop_event.is_set():
            self.pause_event.set()
            self.reset_event.set()
            self.shared.update(status="Reset requested", paused=True)
            self.control_event.set()

    def request_vram_limit(self, limit_mib: int) -> None:
        if self.device.type != "cuda" or self.stop_event.is_set():
            return
        with self.control_lock:
            self.pending_vram_limit_mib = max(0, int(limit_mib))
        self.shared.update(status="Applying VRAM limit")
        self.control_event.set()

    def request_population_size(self, population_size: int) -> None:
        if self.stop_event.is_set():
            return
        with self.control_lock:
            self.pending_population_size = max(1, int(population_size))
        self.pause_event.set()
        self.reset_event.set()
        self.shared.update(status="Applying agent count", paused=True)
        self.control_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)

    def _initialize_device(self) -> None:
        requested = self.runtime_settings.vram_limit_mib
        self.profile = select_device_profile(
            requested,
            self.config.automatic_vram_fraction,
            self.config.minimum_vram_limit_mib,
            self.config.gpu_reserve_mib,
        )
        self.device = self.profile.device
        self.chunk_size = self._recommended_chunk_size()
        self.shared.update(
            device=self.device_label(),
            status="Loading checkpoint",
            vram_limit_mib=self.profile.vram_limit_mib,
            total_vram_mib=self.profile.total_vram_mib,
            vram_automatic=self.profile.automatic,
            population_size=self.population_size,
        )

    def _evolution_settings(self, population_size: int) -> EvolutionSettings:
        elite_ratio = self.config.elite_count / self.config.population_size
        parent_ratio = self.config.parent_pool_size / self.config.population_size
        elite_count = max(1, min(population_size, int(round(population_size * elite_ratio))))
        parent_pool_size = max(elite_count, min(population_size, int(round(population_size * parent_ratio))))
        return EvolutionSettings(
            population_size=population_size,
            elite_count=elite_count,
            parent_pool_size=parent_pool_size,
            mutation_rate=self.config.mutation_rate,
            mutation_scale=self.config.mutation_scale,
            crossover_rate=self.config.crossover_rate,
        )

    def _recommended_chunk_size(self) -> int:
        if self.device.type != "cuda":
            return self.config.evaluation_chunk_size
        parameter_bytes = self.spec.parameter_count * 4
        population_bytes = parameter_bytes * self.population_size
        available = self.profile.vram_limit_mib * 1024 * 1024 - population_bytes * 2 - 128 * 1024 * 1024
        per_agent = parameter_bytes * 8 + self.spec.layer_sizes[0] * 4 + self.config.board_width * self.config.board_height * 24
        estimated = max(8, int(max(0, available) / max(1, per_agent)))
        return min(self.population_size, estimated)

    def _load_or_create(self) -> None:
        checkpoint = self.checkpoint_manager.load()
        if checkpoint is None:
            self.population = initial_population(self.spec, self.settings, self.config.seed).to(self.device)
            self.best_genome = self.population[0].clone()
            self.shared.publish_best(self.best_genome, status="Training")
            return
        if checkpoint.network_spec != self.spec:
            raise ValueError("Checkpoint network architecture does not match version 1.2.0 beta")
        if checkpoint.population.shape[0] != self.population_size:
            logging.getLogger(__name__).info("Checkpoint agent count differs from requested count; starting fresh")
            self.checkpoint_manager.delete()
            self.population = initial_population(self.spec, self.settings, self.config.seed).to(self.device)
            self.best_genome = self.population[0].clone()
            self.shared.publish_best(self.best_genome, status="Training · new agent count", population_size=self.population_size)
            return
        self.population = checkpoint.population.to(self.device)
        self.generation = checkpoint.generation
        self.best_genome = checkpoint.best_genome.to(self.device)
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
            self._initialize_device()
            self._load_or_create()
            completed = 0
            while not self.stop_event.is_set() and (generation_limit is None or completed < generation_limit):
                self._apply_pending_vram_limit()
                self._apply_pending_population_size()
                if self.reset_event.is_set():
                    self._reset_training()
                self._pause_barrier()
                if self.stop_event.is_set():
                    break
                if self.reset_event.is_set():
                    continue
                self.generation += 1
                fitness, scores, lines, rotation_rate = self.evaluate_generation()
                if self.stop_event.is_set():
                    break
                if self.reset_event.is_set():
                    self._apply_pending_population_size()
                    self._reset_training()
                    continue
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
                        rotation_rate=rotation_rate,
                        status="Evolving",
                    )
                    self.save_checkpoint()
                else:
                    self.shared.update(
                        generation=self.generation,
                        current_best_fitness=current_fitness,
                        all_time_best_fitness=self.best_fitness,
                        rotation_rate=rotation_rate,
                        status="Evolving",
                    )
                self.population = self._evolve(fitness)
                if self.generation % self.config.checkpoint_interval_generations == 0:
                    self.save_checkpoint()
                completed += 1
            self.shared.update(status="Stopped" if self.stop_event.is_set() else "Complete", paused=self.pause_event.is_set())
        except Exception as error:
            logger.exception("Training worker failed")
            self.shared.update(status="Error", error=f"{type(error).__name__}: {error}")
        finally:
            if self.reset_event.is_set():
                try:
                    self._apply_pending_population_size()
                    self._reset_training()
                except Exception:
                    logger.exception("Training reset during shutdown failed")
            if self.population is not None and self.best_genome is not None:
                try:
                    self.save_checkpoint()
                except Exception:
                    logger.exception("Final checkpoint save failed")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def evaluate_generation(self) -> tuple[torch.Tensor, list[int], list[int], float]:
        if self.population is None:
            raise RuntimeError("Population is unavailable")
        games = BatchedTetris(self.population_size, self.config.board_width, self.config.board_height, self.device, self.config.rule_weights, self.config.fitness_weights, self.config.network_policy_weight)
        sequence = piece_sequence(self.config.seed + self.generation * 104729, self.config.max_pieces_per_game)
        chunk_size = self.chunk_size
        for piece_round, piece in enumerate(sequence):
            self._apply_pending_vram_limit()
            chunk_size = self.chunk_size
            self._pause_barrier()
            if self.stop_event.is_set() or self.reset_event.is_set():
                break
            active = games.active_indices()
            if len(active) == 0:
                break
            self.shared.update(
                generation=self.generation,
                evaluated_agents=self.population_size - len(active),
                status=f"Training · piece {piece_round + 1}",
            )
            cursor = 0
            while cursor < len(active):
                self._apply_pending_vram_limit()
                chunk_size = self.chunk_size
                self._pause_barrier()
                if self.stop_event.is_set() or self.reset_event.is_set():
                    break
                indices = active[cursor:cursor + chunk_size]
                try:
                    states = games.state_vectors(indices, piece)
                    genomes = self.population[indices]
                    with torch.inference_mode():
                        logits = population_forward(genomes, states, self.spec)
                    games.apply_logits(indices, logits, piece)
                    cursor += len(indices)
                except torch.OutOfMemoryError:
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    if chunk_size <= 8:
                        raise
                    chunk_size = max(8, chunk_size // 2)
                    self.chunk_size = chunk_size
        fitness = games.fitness()
        scores = [int(value) for value in games.scores.cpu().tolist()]
        lines = [int(value) for value in games.lines.cpu().tolist()]
        return fitness, scores, lines, games.rotation_rate()

    def _evolve(self, fitness: torch.Tensor) -> torch.Tensor:
        if self.population is None:
            raise RuntimeError("Population is unavailable")
        evolution_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
        chunk_size = self.chunk_size
        while True:
            evolution_generator = torch.Generator(device=self.device)
            evolution_generator.manual_seed(evolution_seed)
            try:
                return evolve_population(self.population, fitness, self.settings, evolution_generator, chunk_size)
            except torch.OutOfMemoryError:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if chunk_size <= 8:
                    raise
                chunk_size = max(8, chunk_size // 2)
                self.chunk_size = chunk_size

    def _pause_barrier(self) -> None:
        announced = False
        while self.pause_event.is_set() and not self.stop_event.is_set() and not self.reset_event.is_set():
            self._apply_pending_vram_limit()
            if not announced:
                current_status = self.shared.snapshot().status
                status = current_status if current_status == "Paused · fresh start" else "Paused"
                self.shared.update(status=status, paused=True)
                announced = True
            self.control_event.wait(0.1)
            self.control_event.clear()
        if announced and not self.stop_event.is_set() and not self.reset_event.is_set():
            self.shared.update(status="Training", paused=False)

    def _apply_pending_vram_limit(self) -> None:
        with self.control_lock:
            requested = self.pending_vram_limit_mib
            self.pending_vram_limit_mib = None
        if requested is None or self.device.type != "cuda":
            return
        torch.cuda.empty_cache()
        self.profile = apply_vram_limit(
            self.profile,
            requested,
            self.config.automatic_vram_fraction,
            self.config.minimum_vram_limit_mib,
            self.config.gpu_reserve_mib,
        )
        self.chunk_size = self._recommended_chunk_size()
        self.runtime_settings = RuntimeSettings(requested, self.population_size)
        self._save_runtime_settings()
        self.shared.update(
            status="Paused" if self.pause_event.is_set() else "Training",
            vram_limit_mib=self.profile.vram_limit_mib,
            total_vram_mib=self.profile.total_vram_mib,
            vram_automatic=self.profile.automatic,
        )

    def _apply_pending_population_size(self) -> None:
        with self.control_lock:
            requested = self.pending_population_size
            self.pending_population_size = None
        if requested is None:
            return
        self.population_size = requested
        self.settings = self._evolution_settings(requested)
        self.chunk_size = self._recommended_chunk_size()
        self.runtime_settings = RuntimeSettings(self.runtime_settings.vram_limit_mib, requested)
        self._save_runtime_settings()
        self.shared.update(population_size=requested, evaluated_agents=0, status="Resetting for new agent count", paused=True)

    def _save_runtime_settings(self) -> None:
        if self.settings_manager is not None:
            self.settings_manager.save(self.runtime_settings)

    def _reset_training(self) -> None:
        logging.getLogger(__name__).info("Resetting training progress")
        self.shared.update(status="Resetting progress", paused=True)
        self.checkpoint_manager.delete()
        reset_seed = secrets.randbits(63)
        self.generator.manual_seed(reset_seed)
        self.population = None
        self.best_genome = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.population = initial_population(self.spec, self.settings, reset_seed).to(self.device)
        self.generation = 0
        self.best_genome = self.population[0].clone()
        self.best_fitness = float("-inf")
        self.best_score = 0
        self.best_lines = 0
        self.reset_event.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.shared.reset_training(
            self.best_genome,
            self.device_label(),
            self.pause_event.is_set(),
            self.profile.vram_limit_mib,
            self.profile.total_vram_mib,
            self.profile.automatic,
            self.population_size,
        )

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
            return f"CUDA · {self.profile.name}"
        return "CPU"
