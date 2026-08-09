import argparse
import logging
from dataclasses import replace

from .checkpoint import CheckpointManager
from .config import CONFIG
from .gui import TetrisWindow
from .monitor import HardwareMonitor
from .paths import RUNTIME_PATHS, checkpoint_path, settings_path
from .settings import SettingsManager
from .state import SharedTrainingState
from .trainer import Trainer


def configure_logging() -> None:
    log_path = RUNTIME_PATHS["logs"] / "tetris-ai.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Tetris AI")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run() -> int:
    configure_logging()
    arguments = parse_arguments()
    config = CONFIG
    generation_limit = arguments.generations
    if arguments.smoke_test:
        config = replace(CONFIG, max_pieces_per_game=2, evaluation_chunk_size=64)
        arguments.headless = True
        generation_limit = 1
    shared = SharedTrainingState()
    manager = CheckpointManager(checkpoint_path())
    settings = SettingsManager(settings_path())
    trainer = Trainer(config, manager, shared, settings)
    if arguments.headless:
        trainer.run(generation_limit=generation_limit)
        return 1 if shared.snapshot().error else 0
    trainer.start()
    while trainer.thread is not None and trainer.thread.is_alive() and shared.snapshot().status == "Starting":
        trainer.thread.join(0.01)
    monitor = HardwareMonitor(trainer.device, config.hardware_monitor_interval)
    monitor.start()
    window = TetrisWindow(
        config,
        shared,
        monitor,
        trainer.spec,
        trainer.request_reset,
        trainer.request_pause,
        trainer.request_resume,
        trainer.request_vram_limit,
    )
    try:
        window.run()
    except KeyboardInterrupt:
        shared.update(status="Stopping")
    finally:
        trainer.request_stop()
        trainer.join(timeout=30.0)
        if trainer.thread is not None and trainer.thread.is_alive():
            logging.getLogger(__name__).error("Training worker did not stop within timeout")
        monitor.stop()
        window.close()
    return 1 if shared.snapshot().error else 0
