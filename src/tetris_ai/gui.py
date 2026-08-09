import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

import pygame
import torch

from .config import AppConfig
from .engine import Placement, ROTATIONS, TetrisGame
from .monitor import HardwareMonitor, HardwareStats
from .network import NetworkSpec, single_forward
from .state import SharedTrainingState, TrainingStats


BACKGROUND = (12, 15, 23)
PANEL = (23, 28, 40)
BOARD_BACKGROUND = (8, 10, 16)
GRID = (29, 34, 48)
TEXT = (225, 231, 244)
MUTED = (143, 154, 177)
ACCENT = (80, 200, 255)
EXIT_COLOR = (224, 72, 91)
RESET_COLOR = (226, 151, 52)
PIECE_COLORS = {
    1: (66, 214, 230),
    2: (244, 205, 70),
    3: (171, 103, 227),
    4: (92, 204, 120),
    5: (232, 77, 92),
    6: (73, 124, 225),
    7: (239, 145, 58),
}


@dataclass
class FallingPiece:
    placement: Placement
    y: int


class DemoController:
    def __init__(self, config: AppConfig, spec: NetworkSpec, shared: SharedTrainingState):
        self.config = config
        self.spec = spec
        self.shared = shared
        self.game = TetrisGame(config.board_width, config.board_height, config.seed + 777)
        self.genome: torch.Tensor | None = None
        self.revision = -1
        self.falling: FallingPiece | None = None
        self.last_drop = time.monotonic()

    def update(self) -> None:
        genome, revision = self.shared.genome_snapshot()
        if genome is not None and revision != self.revision:
            self.genome = genome
            self.revision = revision
            self.game.reset(self.config.seed + revision * 103)
            self.falling = None
        if self.genome is None:
            return
        now = time.monotonic()
        if self.falling is None:
            legal = self.game.legal_actions()
            if not legal:
                self.game.reset(self.config.seed + self.revision * 103 + self.game.pieces)
                return
            state = torch.tensor(self.game.state_vector(), dtype=torch.float32)
            with torch.inference_mode():
                logits = single_forward(self.genome, state, self.spec)
            action = max(legal, key=lambda candidate: float(logits[candidate]))
            placement = self.game.placement_for_action(action)
            if placement is not None:
                self.falling = FallingPiece(placement, 0)
                self.last_drop = now
            return
        if now - self.last_drop < self.config.visualization_drop_interval:
            return
        self.last_drop = now
        if self.falling.y < self.falling.placement.y:
            self.falling.y += 1
            return
        self.game.lock_placement(self.falling.placement)
        self.falling = None
        if self.game.game_over:
            self.game.reset(self.config.seed + self.revision * 103 + self.game.pieces)


class TetrisWindow:
    def __init__(self, config: AppConfig, shared: SharedTrainingState, monitor: HardwareMonitor, spec: NetworkSpec, reset_training: Callable[[], None]):
        self.config = config
        self.shared = shared
        self.monitor = monitor
        self.reset_training = reset_training
        self.reset_armed_until = 0.0
        self.demo = DemoController(config, spec, shared)
        pygame.init()
        pygame.display.set_caption(f"Tetris AI {config.version}")
        self.cell = 30
        self.margin = 28
        self.board_width = config.board_width * self.cell
        self.board_height = config.board_height * self.cell
        self.panel_width = 390
        self.size = (self.margin * 3 + self.board_width + self.panel_width, self.margin * 2 + self.board_height)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("segoeui", 30, bold=True)
        self.section_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.text_font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        button_x = self.margin * 2 + self.board_width + 28
        button_y = self.size[1] - 76
        self.reset_rect = pygame.Rect(button_x, button_y, 210, 48)
        self.exit_rect = pygame.Rect(button_x + 222, button_y, 112, 48)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.reset_rect.collidepoint(event.pos):
                        self._handle_reset_click()
                    if self.exit_rect.collidepoint(event.pos):
                        running = False
            self.demo.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(self.config.visualization_fps)

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self._draw_board()
        self._draw_panel(self.shared.snapshot(), self.monitor.snapshot())

    def _draw_board(self) -> None:
        origin_x = self.margin
        origin_y = self.margin
        pygame.draw.rect(self.screen, BOARD_BACKGROUND, (origin_x, origin_y, self.board_width, self.board_height), border_radius=6)
        for y, row in enumerate(self.demo.game.board):
            for x, value in enumerate(row):
                self._draw_cell(origin_x, origin_y, x, y, int(value))
        if self.demo.falling is not None:
            placement = self.demo.falling.placement
            shape = ROTATIONS[placement.piece][placement.rotation % len(ROTATIONS[placement.piece])]
            for cell_x, cell_y in shape:
                self._draw_cell(origin_x, origin_y, placement.x + cell_x, self.demo.falling.y + cell_y, int(placement.piece))

    def _draw_cell(self, origin_x: int, origin_y: int, x: int, y: int, value: int) -> None:
        rect = pygame.Rect(origin_x + x * self.cell, origin_y + y * self.cell, self.cell, self.cell)
        pygame.draw.rect(self.screen, GRID, rect, 1)
        if value:
            inner = rect.inflate(-4, -4)
            color = PIECE_COLORS[value]
            pygame.draw.rect(self.screen, color, inner, border_radius=4)
            pygame.draw.line(self.screen, tuple(min(255, channel + 30) for channel in color), inner.topleft, inner.topright, 2)

    def _draw_panel(self, stats: TrainingStats, hardware: HardwareStats) -> None:
        x = self.margin * 2 + self.board_width
        panel_rect = pygame.Rect(x, self.margin, self.panel_width, self.board_height)
        pygame.draw.rect(self.screen, PANEL, panel_rect, border_radius=10)
        cursor_x = x + 28
        cursor_y = self.margin + 24
        self.screen.blit(self.title_font.render("Tetris AI", True, TEXT), (cursor_x, cursor_y))
        cursor_y += 42
        self.screen.blit(self.small_font.render(f"Version {self.config.version}  ·  Neuroevolution", True, ACCENT), (cursor_x, cursor_y))
        cursor_y += 42
        training_rows = [
            ("Generation", str(stats.generation)),
            ("Current best", self._number(stats.current_best_fitness)),
            ("All-time best", self._number(stats.all_time_best_fitness)),
            ("Best score", str(stats.best_score)),
            ("Best lines", str(stats.best_lines)),
            ("Evaluated", f"{stats.evaluated_agents} / {self.config.population_size}"),
        ]
        cursor_y = self._draw_rows(cursor_x, cursor_y, "TRAINING", training_rows)
        hardware_rows = [
            ("CPU Load", f"{hardware.cpu_load:.0f} %"),
            ("RAM Load", f"{hardware.ram_load:.0f} %"),
            ("GPU Load", self._optional(hardware.gpu_load, " %")),
            ("GPU Temp", self._optional(hardware.gpu_temperature, " °C")),
            ("Neural VRAM", f"{hardware.neural_allocated_mib:.0f} / {self.config.neural_network_vram_limit_mib} MiB" if stats.device.startswith("CUDA") else "N/A"),
            ("Reserved VRAM", f"{hardware.neural_reserved_mib:.0f} MiB" if stats.device.startswith("CUDA") else "N/A"),
            ("Total VRAM", self._vram(hardware)),
        ]
        cursor_y = self._draw_rows(cursor_x, cursor_y + 8, "HARDWARE", hardware_rows)
        device = hardware.gpu_name if stats.device.startswith("CUDA") and hardware.gpu_name else stats.device
        status_color = EXIT_COLOR if stats.error else ACCENT
        self.screen.blit(self.small_font.render(f"Device: {device}", True, MUTED), (cursor_x, cursor_y + 8))
        self.screen.blit(self.small_font.render(f"Status: {stats.status}", True, status_color), (cursor_x, cursor_y + 30))
        if stats.error:
            error_text = stats.error[:48]
            self.screen.blit(self.small_font.render(error_text, True, EXIT_COLOR), (cursor_x, cursor_y + 52))
        reset_armed = time.monotonic() <= self.reset_armed_until
        reset_label = "CONFIRM RESET" if reset_armed else "RESET PROGRESS"
        reset_color = EXIT_COLOR if reset_armed else RESET_COLOR
        self._draw_button(self.reset_rect, reset_label, reset_color)
        self._draw_button(self.exit_rect, "EXIT", EXIT_COLOR)

    def _handle_reset_click(self) -> None:
        now = time.monotonic()
        if now <= self.reset_armed_until:
            self.reset_armed_until = 0.0
            self.reset_training()
            return
        self.reset_armed_until = now + 3.0

    def _draw_button(self, rect: pygame.Rect, text: str, color: tuple[int, int, int]) -> None:
        if rect.collidepoint(pygame.mouse.get_pos()):
            color = tuple(min(255, channel + 18) for channel in color)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        label = self.section_font.render(text, True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_rows(self, x: int, y: int, title: str, rows: list[tuple[str, str]]) -> int:
        self.screen.blit(self.section_font.render(title, True, MUTED), (x, y))
        y += 31
        for label, value in rows:
            self.screen.blit(self.text_font.render(label, True, MUTED), (x, y))
            rendered = self.text_font.render(value, True, TEXT)
            self.screen.blit(rendered, (x + 326 - rendered.get_width(), y))
            y += 25
        return y

    def _optional(self, value: float | int | None, suffix: str) -> str:
        return "N/A" if value is None else f"{value:.0f}{suffix}"

    def _vram(self, hardware: HardwareStats) -> str:
        if hardware.total_vram_used_mib is None or hardware.total_vram_mib is None:
            return "N/A"
        return f"{hardware.total_vram_used_mib:.0f} / {hardware.total_vram_mib:.0f} MiB"

    def _number(self, value: float) -> str:
        return "0" if value == float("-inf") else f"{value:.1f}"

    def close(self) -> None:
        pygame.quit()
