import math
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
from .settings import RuntimeSettings
from .settings_panel import SettingsPanel
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
PAUSE_COLOR = (72, 137, 224)
START_COLOR = (57, 180, 112)
INPUT_COLOR = (36, 43, 59)
PIECE_COLORS = {
    1: (66, 214, 230),
    2: (244, 205, 70),
    3: (171, 103, 227),
    4: (92, 204, 120),
    5: (232, 77, 92),
    6: (73, 124, 225),
    7: (239, 145, 58),
}

TRAINING_INFO = {
    "Generation": "Number of the generation currently being evaluated. One generation tests every agent and creates the next population.",
    "Current best": "Highest fitness reached by an agent in the current generation.",
    "All-time best": "Highest fitness reached since the current training run began.",
    "Best score": "Largest standard Tetris score reached by the all-time best agent.",
    "Best lines": "Largest line-clear count reached by the all-time best agent.",
    "Evaluated": "Agents already finished or eliminated in the current generation versus the total population.",
    "Rotated moves": "Percentage of placements that use a non-default rotation. It helps confirm that rotations are being explored.",
}

HARDWARE_INFO = {
    "CPU Load": "Current total processor utilization. Interface, monitoring, checkpoint work, and CPU-only training contribute to it.",
    "RAM Load": "Percentage of system memory currently used by all applications.",
    "GPU Load": "Current NVIDIA GPU compute utilization reported by the driver.",
    "GPU Temp": "Current physical GPU temperature reported by NVIDIA NVML. N/A means the sensor is unavailable.",
    "Neural VRAM": "VRAM actively allocated by PyTorch tensors compared with the configured neural memory limit.",
    "Reserved VRAM": "VRAM reserved by PyTorch's caching allocator for fast reuse.",
    "Total VRAM": "VRAM used by the whole system compared with the GPU's physical capacity.",
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

    def update(self, paused: bool = False) -> None:
        genome, revision = self.shared.genome_snapshot()
        if genome is not None and revision != self.revision:
            self.genome = genome
            self.revision = revision
            self.game.reset(self.config.seed + revision * 103)
            self.falling = None
        if self.genome is None or paused:
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
            action = max(legal, key=lambda candidate: self.game.rule_score(candidate, self.config.rule_weights) + math.tanh(float(logits[candidate])) * self.config.network_policy_weight)
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
    def __init__(
        self,
        config: AppConfig,
        shared: SharedTrainingState,
        monitor: HardwareMonitor,
        spec: NetworkSpec,
        reset_training: Callable[[], None],
        pause_training: Callable[[], None],
        resume_training: Callable[[], None],
        set_vram_limit: Callable[[int], None],
        set_population_size: Callable[[int], None],
        apply_settings: Callable[[RuntimeSettings], None],
    ):
        self.config = config
        self.shared = shared
        self.monitor = monitor
        self.reset_training = reset_training
        self.pause_training = pause_training
        self.resume_training = resume_training
        self.set_vram_limit = set_vram_limit
        self.set_population_size = set_population_size
        self.apply_settings = apply_settings
        self.reset_armed_until = 0.0
        self.demo = DemoController(config, spec, shared)
        pygame.init()
        pygame.display.set_caption(f"Tetris AI {config.version}")
        self.minimum_size = (860, 780)
        self.size = (940, 820)
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("segoeui", 30, bold=True)
        self.section_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.text_font = pygame.font.SysFont("consolas", 15)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.button_font = pygame.font.SysFont("segoeui", 16, bold=True)
        self.margin = 28
        self.cell = 30
        self.board_origin = (self.margin, self.margin)
        self.board_width = config.board_width * self.cell
        self.board_height = config.board_height * self.cell
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_rect = pygame.Rect(0, 0, 0, 0)
        self.settings_rect = pygame.Rect(0, 0, 0, 0)
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.vram_slider_rect = pygame.Rect(0, 0, 0, 0)
        self.vram_input_rect = pygame.Rect(0, 0, 0, 0)
        self.vram_auto_rect = pygame.Rect(0, 0, 0, 0)
        self.agents_input_rect = pygame.Rect(0, 0, 0, 0)
        self.agents_apply_rect = pygame.Rect(0, 0, 0, 0)
        self.vram_dragging = False
        self.vram_input_active = False
        self.vram_input_value = ""
        self.vram_preview_mib: int | None = None
        self.agents_input_active = False
        self.agents_input_value = ""
        self.info_targets: list[tuple[pygame.Rect, str]] = []
        self.settings_was_paused = False
        self.settings_panel = SettingsPanel(
            self.screen,
            self.title_font,
            self.section_font,
            self.text_font,
            self.small_font,
            self.button_font,
            self.apply_settings,
        )
        self._layout()

    def run(self) -> bool:
        running = True
        restart_requested = False
        while running:
            stats = self.shared.snapshot()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    width = max(self.minimum_size[0], event.w)
                    height = max(self.minimum_size[1], event.h)
                    self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                    self.settings_panel.update_screen(self.screen)
                    self._layout()
                elif self.settings_panel.visible:
                    result = self.settings_panel.handle_event(event)
                    if result == "cancel" and not self.settings_was_paused:
                        self.resume_training()
                    elif result == "applied":
                        restart_requested = True
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_down(event.pos, stats)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._handle_mouse_up(stats)
                elif event.type == pygame.MOUSEMOTION and self.vram_dragging:
                    self.vram_preview_mib = self._slider_value(event.pos[0], stats)
                elif event.type == pygame.KEYDOWN:
                    if self.vram_input_active:
                        self._handle_vram_key(event, stats)
                    elif self.agents_input_active:
                        self._handle_agents_key(event)
            stats = self.shared.snapshot()
            self.demo.update(stats.paused)
            self.draw(stats)
            pygame.display.flip()
            self.clock.tick(self.config.visualization_fps)
        return restart_requested

    def _handle_mouse_down(self, position: tuple[int, int], stats: TrainingStats) -> None:
        if self.pause_rect.collidepoint(position):
            if stats.paused:
                self.resume_training()
            else:
                self.pause_training()
            return
        if self.reset_rect.collidepoint(position):
            self._handle_reset_click()
            return
        if self.settings_rect.collidepoint(position):
            self.settings_was_paused = stats.paused
            if not stats.paused:
                self.pause_training()
            self.vram_input_active = False
            self.agents_input_active = False
            self.settings_panel.open(self.config)
            return
        if self.exit_rect.collidepoint(position):
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return
        if self.agents_apply_rect.collidepoint(position):
            self._apply_agent_count()
            return
        if self.agents_input_rect.collidepoint(position):
            self.agents_input_active = True
            self.vram_input_active = False
            self.agents_input_value = str(stats.population_size)
            return
        self.agents_input_active = False
        if stats.total_vram_mib <= 0:
            self.vram_input_active = False
            return
        if self.vram_auto_rect.collidepoint(position):
            self.vram_preview_mib = None
            self.vram_input_active = False
            self.set_vram_limit(0)
            return
        if self.vram_input_rect.collidepoint(position):
            self.vram_input_active = True
            self.agents_input_active = False
            self.vram_input_value = str(stats.vram_limit_mib)
            return
        self.vram_input_active = False
        if self.vram_slider_rect.inflate(0, 16).collidepoint(position):
            self.vram_dragging = True
            self.vram_preview_mib = self._slider_value(position[0], stats)

    def _handle_mouse_up(self, stats: TrainingStats) -> None:
        if not self.vram_dragging:
            return
        self.vram_dragging = False
        if self.vram_preview_mib is not None and stats.total_vram_mib > 0:
            self.set_vram_limit(self.vram_preview_mib)

    def _handle_vram_key(self, event: pygame.event.Event, stats: TrainingStats) -> None:
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.vram_input_value:
                value = self._clamp_vram(int(self.vram_input_value), stats)
                self.vram_preview_mib = value
                self.set_vram_limit(value)
            self.vram_input_active = False
        elif event.key == pygame.K_ESCAPE:
            self.vram_input_active = False
        elif event.key == pygame.K_BACKSPACE:
            self.vram_input_value = self.vram_input_value[:-1]
        elif event.unicode.isdigit() and len(self.vram_input_value) < 6:
            self.vram_input_value += event.unicode

    def _handle_agents_key(self, event: pygame.event.Event) -> None:
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._apply_agent_count()
        elif event.key == pygame.K_ESCAPE:
            self.agents_input_active = False
        elif event.key == pygame.K_BACKSPACE:
            self.agents_input_value = self.agents_input_value[:-1]
        elif event.unicode.isdigit():
            self.agents_input_value += event.unicode

    def _apply_agent_count(self) -> None:
        if self.agents_input_value:
            self.set_population_size(max(1, int(self.agents_input_value)))
        self.agents_input_active = False

    def draw(self, stats: TrainingStats | None = None) -> None:
        self._layout()
        current = self.shared.snapshot() if stats is None else stats
        self.info_targets = []
        if self.settings_panel.visible:
            self.settings_panel.draw(self._draw_info_icon)
            self._draw_hover_tooltip()
            return
        self.screen.fill(BACKGROUND)
        self._draw_board()
        self._draw_panel(current, self.monitor.snapshot())
        self._draw_hover_tooltip()

    def _layout(self) -> None:
        width, height = self.screen.get_size()
        self.margin = max(20, min(32, min(width, height) // 26))
        panel_width = max(410, min(510, int(width * 0.48)))
        left_width = width - panel_width - self.margin * 3
        self.cell = max(4, min(left_width // self.config.board_width, (height - self.margin * 2) // self.config.board_height))
        self.board_width = self.config.board_width * self.cell
        self.board_height = self.config.board_height * self.cell
        board_x = self.margin + max(0, (left_width - self.board_width) // 2)
        board_y = max(self.margin, (height - self.board_height) // 2)
        self.board_origin = (board_x, board_y)
        panel_x = width - self.margin - panel_width
        self.panel_rect = pygame.Rect(panel_x, self.margin, panel_width, height - self.margin * 2)
        inner_x = panel_x + 24
        inner_width = panel_width - 48
        button_y = self.panel_rect.bottom - 60
        gap = 8
        button_width = (inner_width - gap * 3) // 4
        self.pause_rect = pygame.Rect(inner_x, button_y, button_width, 42)
        self.reset_rect = pygame.Rect(self.pause_rect.right + gap, button_y, button_width, 42)
        self.settings_rect = pygame.Rect(self.reset_rect.right + gap, button_y, button_width, 42)
        self.exit_rect = pygame.Rect(self.settings_rect.right + gap, button_y, inner_x + inner_width - self.settings_rect.right - gap, 42)

    def _draw_board(self) -> None:
        origin_x, origin_y = self.board_origin
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
            inset = max(2, self.cell // 8)
            inner = rect.inflate(-inset * 2, -inset * 2)
            color = PIECE_COLORS[value]
            pygame.draw.rect(self.screen, color, inner, border_radius=max(2, self.cell // 8))
            pygame.draw.line(self.screen, tuple(min(255, channel + 30) for channel in color), inner.topleft, inner.topright, max(1, self.cell // 15))

    def _draw_panel(self, stats: TrainingStats, hardware: HardwareStats) -> None:
        pygame.draw.rect(self.screen, PANEL, self.panel_rect, border_radius=10)
        cursor_x = self.panel_rect.x + 24
        value_right = self.panel_rect.right - 24
        cursor_y = self.panel_rect.y + 20
        self.screen.blit(self.title_font.render("Tetris AI", True, TEXT), (cursor_x, cursor_y))
        cursor_y += 38
        self.screen.blit(self.small_font.render(f"Version {self.config.version}  ·  Neuroevolution", True, ACCENT), (cursor_x, cursor_y))
        cursor_y += 34
        training_rows = [
            ("Generation", str(stats.generation), TRAINING_INFO["Generation"]),
            ("Current best", self._number(stats.current_best_fitness), TRAINING_INFO["Current best"]),
            ("All-time best", self._number(stats.all_time_best_fitness), TRAINING_INFO["All-time best"]),
            ("Best score", str(stats.best_score), TRAINING_INFO["Best score"]),
            ("Best lines", str(stats.best_lines), TRAINING_INFO["Best lines"]),
            ("Evaluated", f"{stats.evaluated_agents} / {stats.population_size}", TRAINING_INFO["Evaluated"]),
            ("Rotated moves", f"{stats.rotation_rate:.1f} %", TRAINING_INFO["Rotated moves"]),
        ]
        cursor_y = self._draw_rows(cursor_x, cursor_y, value_right, "TRAINING", training_rows)
        hardware_rows = [
            ("CPU Load", f"{hardware.cpu_load:.0f} %", HARDWARE_INFO["CPU Load"]),
            ("RAM Load", f"{hardware.ram_load:.0f} %", HARDWARE_INFO["RAM Load"]),
            ("GPU Load", self._optional(hardware.gpu_load, " %"), HARDWARE_INFO["GPU Load"]),
            ("GPU Temp", self._optional(hardware.gpu_temperature, " °C"), HARDWARE_INFO["GPU Temp"]),
            ("Neural VRAM", f"{hardware.neural_allocated_mib:.0f} / {stats.vram_limit_mib} MiB" if stats.total_vram_mib else "N/A", HARDWARE_INFO["Neural VRAM"]),
            ("Reserved VRAM", f"{hardware.neural_reserved_mib:.0f} MiB" if stats.total_vram_mib else "N/A", HARDWARE_INFO["Reserved VRAM"]),
            ("Total VRAM", self._vram(hardware), HARDWARE_INFO["Total VRAM"]),
        ]
        cursor_y = self._draw_rows(cursor_x, cursor_y + 2, value_right, "HARDWARE", hardware_rows)
        cursor_y = self._draw_vram_control(cursor_x, cursor_y + 2, value_right, stats)
        cursor_y = self._draw_agents_control(cursor_x, cursor_y, value_right, stats)
        device = hardware.gpu_name if stats.total_vram_mib and hardware.gpu_name else stats.device
        status_color = EXIT_COLOR if stats.error else ACCENT
        device_text = self._fit_text(f"Device: {device}", self.small_font, self.panel_rect.width - 48)
        status_text = self._fit_text(f"Status: {stats.status}", self.small_font, self.panel_rect.width - 48)
        self.screen.blit(self.small_font.render(device_text, True, MUTED), (cursor_x, cursor_y + 2))
        self._draw_info_icon(value_right - 6, cursor_y + 10, "Compute device selected automatically for neural-network evaluation and evolution.")
        self.screen.blit(self.small_font.render(status_text, True, status_color), (cursor_x, cursor_y + 22))
        self._draw_info_icon(value_right - 6, cursor_y + 30, "Current lifecycle state of the training worker, including pause, evaluation, evolution, reset, and errors.")
        if stats.error:
            error_text = self._fit_text(stats.error, self.small_font, self.panel_rect.width - 48)
            self.screen.blit(self.small_font.render(error_text, True, EXIT_COLOR), (cursor_x, cursor_y + 42))
        reset_armed = time.monotonic() <= self.reset_armed_until
        reset_label = "CONFIRM RESET" if reset_armed else "RESET"
        reset_color = EXIT_COLOR if reset_armed else RESET_COLOR
        pause_label = "START" if stats.paused else "PAUSE"
        pause_color = START_COLOR if stats.paused else PAUSE_COLOR
        self._draw_button(self.pause_rect, pause_label, pause_color)
        self._draw_button(self.reset_rect, reset_label, reset_color)
        self._draw_button(self.settings_rect, "SETTINGS", INPUT_COLOR)
        self._draw_button(self.exit_rect, "EXIT", EXIT_COLOR)

    def _draw_vram_control(self, x: int, y: int, value_right: int, stats: TrainingStats) -> int:
        self.screen.blit(self.section_font.render("VRAM LIMIT", True, MUTED), (x, y))
        self._draw_info_icon(x + self.section_font.size("VRAM LIMIT")[0] + 12, y + 10, "Controls the maximum VRAM budget used by neural tensors. AUTO derives a safe budget from detected GPU capacity and reserve settings.")
        y += 26
        input_width = 78
        auto_width = 54
        gap = 8
        self.vram_auto_rect = pygame.Rect(value_right - auto_width, y - 6, auto_width, 28)
        self.vram_input_rect = pygame.Rect(self.vram_auto_rect.x - gap - input_width, y - 6, input_width, 28)
        self.vram_slider_rect = pygame.Rect(x, y + 4, max(60, self.vram_input_rect.x - gap - x), 8)
        enabled = stats.total_vram_mib > 0
        limit = self.vram_preview_mib if self.vram_preview_mib is not None else stats.vram_limit_mib
        track_color = GRID if enabled else INPUT_COLOR
        pygame.draw.rect(self.screen, track_color, self.vram_slider_rect, border_radius=4)
        if enabled:
            minimum, maximum = self._vram_range(stats)
            ratio = (self._clamp_vram(limit, stats) - minimum) / max(1, maximum - minimum)
            knob_x = int(self.vram_slider_rect.x + ratio * self.vram_slider_rect.width)
            pygame.draw.circle(self.screen, ACCENT, (knob_x, self.vram_slider_rect.centery), 7)
        input_color = ACCENT if self.vram_input_active else GRID
        pygame.draw.rect(self.screen, INPUT_COLOR, self.vram_input_rect, border_radius=5)
        pygame.draw.rect(self.screen, input_color, self.vram_input_rect, 1, border_radius=5)
        if self.vram_input_active:
            input_text = self.vram_input_value
        elif not enabled:
            input_text = "N/A"
        else:
            input_text = str(limit)
        rendered = self.small_font.render(input_text, True, TEXT if enabled else MUTED)
        self.screen.blit(rendered, rendered.get_rect(center=self.vram_input_rect.center))
        auto_color = START_COLOR if stats.vram_automatic and enabled else INPUT_COLOR
        pygame.draw.rect(self.screen, auto_color, self.vram_auto_rect, border_radius=5)
        auto_label = self.small_font.render("AUTO", True, TEXT if enabled else MUTED)
        self.screen.blit(auto_label, auto_label.get_rect(center=self.vram_auto_rect.center))
        unit = self.small_font.render("MiB", True, MUTED)
        self.screen.blit(unit, (self.vram_input_rect.x + 22, self.vram_input_rect.bottom + 1))
        return y + 40

    def _draw_agents_control(self, x: int, y: int, value_right: int, stats: TrainingStats) -> int:
        self.screen.blit(self.section_font.render("AGENTS PER GENERATION", True, MUTED), (x, y))
        self._draw_info_icon(x + self.section_font.size("AGENTS PER GENERATION")[0] + 12, y + 10, "Controls population size. Applying a new value starts fresh because populations of different sizes cannot share one checkpoint safely.")
        y += 26
        apply_width = 62
        input_width = 112
        gap = 8
        self.agents_apply_rect = pygame.Rect(value_right - apply_width, y - 6, apply_width, 28)
        self.agents_input_rect = pygame.Rect(self.agents_apply_rect.x - gap - input_width, y - 6, input_width, 28)
        pygame.draw.rect(self.screen, INPUT_COLOR, self.agents_input_rect, border_radius=5)
        border = ACCENT if self.agents_input_active else GRID
        pygame.draw.rect(self.screen, border, self.agents_input_rect, 1, border_radius=5)
        value = self.agents_input_value if self.agents_input_active else str(stats.population_size)
        value = self._fit_number(value, self.small_font, self.agents_input_rect.width - 8)
        rendered = self.small_font.render(value, True, TEXT)
        self.screen.blit(rendered, rendered.get_rect(center=self.agents_input_rect.center))
        pygame.draw.rect(self.screen, PAUSE_COLOR, self.agents_apply_rect, border_radius=5)
        apply_label = self.small_font.render("APPLY", True, TEXT)
        self.screen.blit(apply_label, apply_label.get_rect(center=self.agents_apply_rect.center))
        warning = self.small_font.render("Applying fully resets training", True, MUTED)
        self.screen.blit(warning, (x, y + 25))
        return y + 47

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
        label = self.button_font.render(text, True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_rows(self, x: int, y: int, value_right: int, title: str, rows: list[tuple[str, str, str]]) -> int:
        self.screen.blit(self.section_font.render(title, True, MUTED), (x, y))
        y += 27
        for label, value, description in rows:
            rendered_label = self.text_font.render(label, True, MUTED)
            self.screen.blit(rendered_label, (x, y))
            self._draw_info_icon(x + rendered_label.get_width() + 12, y + 8, description)
            rendered = self.text_font.render(value, True, TEXT)
            self.screen.blit(rendered, (value_right - rendered.get_width(), y))
            y += 22
        return y

    def _draw_info_icon(self, center_x: int, center_y: int, description: str) -> None:
        rect = pygame.Rect(center_x - 7, center_y - 7, 14, 14)
        if not rect.colliderect(self.screen.get_clip()):
            return
        pygame.draw.circle(self.screen, MUTED, (center_x, center_y), 6, 1)
        glyph = self.small_font.render("i", True, MUTED)
        self.screen.blit(glyph, glyph.get_rect(center=(center_x, center_y - 1)))
        self.info_targets.append((rect, description))

    def _draw_hover_tooltip(self) -> None:
        mouse = pygame.mouse.get_pos()
        description = next((text for rect, text in reversed(self.info_targets) if rect.collidepoint(mouse)), None)
        if description is None:
            return
        maximum_width = min(360, self.screen.get_width() - 32)
        words = description.split()
        lines: list[str] = []
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if self.small_font.size(candidate)[0] <= maximum_width - 24:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        width = min(maximum_width, max(self.small_font.size(item)[0] for item in lines) + 24)
        height = len(lines) * 20 + 18
        x = min(mouse[0] + 16, self.screen.get_width() - width - 10)
        y = min(mouse[1] + 18, self.screen.get_height() - height - 10)
        rect = pygame.Rect(max(10, x), max(10, y), width, height)
        pygame.draw.rect(self.screen, (7, 9, 15), rect, border_radius=7)
        pygame.draw.rect(self.screen, ACCENT, rect, 1, border_radius=7)
        for index, item in enumerate(lines):
            self.screen.blit(self.small_font.render(item, True, TEXT), (rect.x + 12, rect.y + 9 + index * 20))

    def _vram_range(self, stats: TrainingStats) -> tuple[int, int]:
        reserve = min(self.config.gpu_reserve_mib, stats.total_vram_mib // 4)
        maximum = max(1, stats.total_vram_mib - reserve)
        minimum = min(self.config.minimum_vram_limit_mib, maximum)
        return minimum, maximum

    def _clamp_vram(self, value: int, stats: TrainingStats) -> int:
        minimum, maximum = self._vram_range(stats)
        return min(maximum, max(minimum, value))

    def _slider_value(self, mouse_x: int, stats: TrainingStats) -> int:
        minimum, maximum = self._vram_range(stats)
        ratio = (mouse_x - self.vram_slider_rect.x) / max(1, self.vram_slider_rect.width)
        raw = minimum + min(1.0, max(0.0, ratio)) * (maximum - minimum)
        return self._clamp_vram(int(round(raw / 128.0) * 128), stats)

    def _fit_text(self, text: str, font: pygame.font.Font, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size(shortened + "…")[0] > width:
            shortened = shortened[:-1]
        return shortened + "…"

    def _fit_number(self, text: str, font: pygame.font.Font, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size("…" + shortened)[0] > width:
            shortened = shortened[1:]
        return "…" + shortened

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
