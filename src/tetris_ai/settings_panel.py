import time
from collections.abc import Callable

import pygame

from .autotune import automatic_runtime_settings
from .config import AppConfig, CONFIG
from .settings import SETTING_DEFINITIONS, RuntimeSettings, apply_runtime_settings, parse_runtime_settings, setting_values
from .translations import error_text, language_name, normalize_language, text


BACKGROUND = (12, 15, 23)
PANEL = (23, 28, 40)
GRID = (44, 51, 69)
TEXT = (225, 231, 244)
MUTED = (143, 154, 177)
ACCENT = (80, 200, 255)
RESET_COLOR = (226, 151, 52)
EXIT_COLOR = (224, 72, 91)
INPUT_COLOR = (36, 43, 59)


class SettingsPanel:
    def __init__(
        self,
        screen: pygame.Surface,
        title_font: pygame.font.Font,
        section_font: pygame.font.Font,
        text_font: pygame.font.Font,
        small_font: pygame.font.Font,
        button_font: pygame.font.Font,
        apply_settings: Callable[[RuntimeSettings], None],
        auto_settings: Callable[[str], tuple[RuntimeSettings, str]] = automatic_runtime_settings,
    ):
        self.screen = screen
        self.title_font = title_font
        self.section_font = section_font
        self.text_font = text_font
        self.small_font = small_font
        self.button_font = button_font
        self.apply_settings = apply_settings
        self.auto_settings = auto_settings
        self.visible = False
        self.values: dict[str, str] = {}
        self.original_values: dict[str, str] = {}
        self.active_key: str | None = None
        self.select_all = False
        self.scroll = 0
        self.maximum_scroll = 0
        self.error = ""
        self.notice = ""
        self.apply_armed_until = 0.0
        self.field_rects: dict[str, pygame.Rect] = {}
        self.content_rect = pygame.Rect(0, 0, 0, 0)
        self.cancel_rect = pygame.Rect(0, 0, 0, 0)
        self.apply_rect = pygame.Rect(0, 0, 0, 0)
        self.auto_rect = pygame.Rect(0, 0, 0, 0)
        self.defaults_rect = pygame.Rect(0, 0, 0, 0)

    def update_screen(self, screen: pygame.Surface) -> None:
        self.screen = screen

    def open(self, config: AppConfig) -> None:
        self.values = setting_values(config)
        self.original_values = dict(self.values)
        self.active_key = None
        self.select_all = False
        self.scroll = 0
        self.error = ""
        self.notice = ""
        self.apply_armed_until = 0.0
        self.visible = True

    def close(self) -> None:
        self.visible = False
        self.active_key = None

    def handle_event(self, event: pygame.event.Event) -> str:
        if event.type == pygame.MOUSEWHEEL:
            self.scroll = min(self.maximum_scroll, max(0, self.scroll - event.y * 42))
            return "none"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            direction = -1 if event.button == 4 else 1
            self.scroll = min(self.maximum_scroll, max(0, self.scroll + direction * 42))
            return "none"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.auto_rect.collidepoint(event.pos):
                self._load_automatic()
                return "none"
            if self.defaults_rect.collidepoint(event.pos):
                self._load_defaults()
                return "none"
            if self.cancel_rect.collidepoint(event.pos):
                self.close()
                return "cancel"
            if self.apply_rect.collidepoint(event.pos):
                return self._apply()
            for key, rect in self.field_rects.items():
                if rect.collidepoint(event.pos):
                    definition = next(item for item in SETTING_DEFINITIONS if item.key == key)
                    if definition.value_type == "choice":
                        current = self.values.get(key, definition.options[0])
                        index = definition.options.index(current) if current in definition.options else 0
                        self.values[key] = definition.options[(index + 1) % len(definition.options)]
                        self.error = ""
                        self.notice = ""
                        return "none"
                    self.active_key = key
                    self.select_all = True
                    self.error = ""
                    return "none"
            self.active_key = None
            return "none"
        if event.type != pygame.KEYDOWN:
            return "none"
        if event.key == pygame.K_ESCAPE:
            if self.active_key is not None:
                self.active_key = None
                return "none"
            self.close()
            return "cancel"
        if self.active_key is None:
            return "none"
        keys = [definition.key for definition in SETTING_DEFINITIONS if definition.value_type != "choice"]
        if event.key == pygame.K_TAB:
            current = keys.index(self.active_key)
            direction = -1 if event.mod & pygame.KMOD_SHIFT else 1
            self.active_key = keys[(current + direction) % len(keys)]
            self.select_all = True
            return "none"
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.active_key = None
            return "none"
        if event.mod & pygame.KMOD_CTRL and event.key == pygame.K_a:
            self.select_all = True
            return "none"
        if event.key == pygame.K_BACKSPACE:
            if self.select_all:
                self.values[self.active_key] = ""
                self.select_all = False
            else:
                self.values[self.active_key] = self.values[self.active_key][:-1]
            return "none"
        if event.unicode and event.unicode in "0123456789.,- ":
            if self.select_all:
                self.values[self.active_key] = event.unicode
                self.select_all = False
            elif len(self.values[self.active_key]) < 48:
                self.values[self.active_key] += event.unicode
        return "none"

    def _apply(self) -> str:
        now = time.monotonic()
        requires_reset = self._requires_reset()
        if requires_reset and now > self.apply_armed_until:
            self.apply_armed_until = now + 4.0
            self.error = self._t("Press CONFIRM RESET to erase the current training and apply all values")
            self.active_key = None
            return "none"
        try:
            runtime = parse_runtime_settings(self.values)
            self.apply_settings(runtime)
        except (OSError, ValueError) as error:
            self.error = error_text(self._language(), str(error))
            self.apply_armed_until = 0.0
            return "none"
        self.close()
        return "applied_reset" if requires_reset else "applied"

    def _load_defaults(self) -> None:
        language = self._language()
        self.values = setting_values(CONFIG)
        self.values["language"] = language
        self.active_key = None
        self.error = ""
        self.notice = self._t("Default model values loaded. Press APPLY & RESET to use them.")
        self.apply_armed_until = 0.0

    def _load_automatic(self) -> None:
        language = self._language()
        try:
            runtime, profile = self.auto_settings(language)
            configured = apply_runtime_settings(CONFIG, runtime)
            self.values = setting_values(configured)
            self.values["language"] = language
            self.error = ""
            self.notice = self._t("Automatic values loaded for: {profile}. Press APPLY & RESET to use them.", profile=profile)
        except (OSError, RuntimeError, ValueError) as error:
            self.error = str(error)
            self.notice = ""
        self.active_key = None
        self.apply_armed_until = 0.0

    def _language(self) -> str:
        return normalize_language(self.values.get("language", CONFIG.language))

    def _requires_reset(self) -> bool:
        return any(self.values.get(key) != value for key, value in self.original_values.items() if key != "language")

    def _t(self, source: str, **values: object) -> str:
        return text(self._language(), source, **values)

    def draw(self, draw_info: Callable[[int, int, str], None]) -> None:
        width, height = self.screen.get_size()
        self.screen.fill(BACKGROUND)
        margin = max(16, min(30, min(width, height) // 28))
        modal = pygame.Rect(margin, margin, width - margin * 2, height - margin * 2)
        pygame.draw.rect(self.screen, PANEL, modal, border_radius=12)
        header_x = modal.x + 28
        header_y = modal.y + 20
        self.screen.blit(self.title_font.render(self._t("MODEL SETTINGS"), True, TEXT), (header_x, header_y))
        subtitle = self._t("Every value is editable. Applying settings starts training from generation 0.")
        self.screen.blit(self.small_font.render(subtitle, True, ACCENT), (header_x, header_y + 40))
        self.defaults_rect = pygame.Rect(modal.right - 178, header_y + 2, 150, 32)
        self.auto_rect = pygame.Rect(self.defaults_rect.x - 100, header_y + 2, 90, 32)
        self._draw_button(self.auto_rect, self._t("AUTO"), ACCENT)
        self._draw_button(self.defaults_rect, self._t("DEFAULTS"), INPUT_COLOR)
        footer_height = 78
        self.content_rect = pygame.Rect(modal.x + 20, modal.y + 82, modal.width - 40, modal.height - 82 - footer_height)
        footer_y = self.content_rect.bottom + 12
        button_height = 42
        self.cancel_rect = pygame.Rect(modal.right - 374, footer_y + 10, 126, button_height)
        self.apply_rect = pygame.Rect(modal.right - 236, footer_y + 10, 208, button_height)
        content_height = self._content_height()
        self.maximum_scroll = max(0, content_height - self.content_rect.height)
        self.scroll = min(self.maximum_scroll, max(0, self.scroll))
        self.field_rects = {}
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.content_rect)
        self._draw_fields(draw_info)
        self.screen.set_clip(previous_clip)
        if self.maximum_scroll:
            track = pygame.Rect(self.content_rect.right - 5, self.content_rect.y, 4, self.content_rect.height)
            pygame.draw.rect(self.screen, INPUT_COLOR, track, border_radius=2)
            thumb_height = max(34, int(track.height * self.content_rect.height / content_height))
            thumb_y = track.y + int((track.height - thumb_height) * self.scroll / self.maximum_scroll)
            pygame.draw.rect(self.screen, ACCENT, (track.x, thumb_y, track.width, thumb_height), border_radius=2)
        if self.error:
            error = self._fit_text(self.error, self.small_font, max(200, self.cancel_rect.x - header_x - 16))
            self.screen.blit(self.small_font.render(error, True, EXIT_COLOR), (header_x, footer_y + 22))
        elif self.notice:
            notice_width = max(200, self.cancel_rect.x - header_x - 16)
            for index, notice in enumerate(self._wrap_text(self.notice, self.small_font, notice_width, 2)):
                self.screen.blit(self.small_font.render(notice, True, ACCENT), (header_x, footer_y + 12 + index * 18))
        else:
            message = self._t("Changing only the language keeps the current checkpoint.") if not self._requires_reset() and self.values.get("language") != self.original_values.get("language") else self._t("Changes are stored locally in .runtime/data/settings.json")
            self.screen.blit(self.small_font.render(message, True, MUTED), (header_x, footer_y + 22))
        self._draw_button(self.cancel_rect, self._t("CANCEL"), INPUT_COLOR)
        armed = time.monotonic() <= self.apply_armed_until
        requires_reset = self._requires_reset()
        label = self._t("CONFIRM RESET") if armed and requires_reset else self._t("APPLY & RESET") if requires_reset else self._t("APPLY CHANGES")
        self._draw_button(self.apply_rect, label, EXIT_COLOR if armed and requires_reset else RESET_COLOR)

    def _draw_fields(self, draw_info: Callable[[int, int, str], None]) -> None:
        x = self.content_rect.x + 10
        right = self.content_rect.right - 16
        input_width = max(190, min(330, int(self.content_rect.width * 0.38)))
        input_x = right - input_width
        y = self.content_rect.y - self.scroll
        category = ""
        for definition in SETTING_DEFINITIONS:
            if definition.category != category:
                category = definition.category
                self.screen.blit(self.section_font.render(self._t(category), True, MUTED), (x, y + 6))
                y += 36
            label = self.text_font.render(self._t(definition.label), True, TEXT)
            rect = pygame.Rect(input_x, y + 3, input_width, 32)
            row_rect = pygame.Rect(x, y, right - x, 40)
            if row_rect.colliderect(self.content_rect):
                self.screen.blit(label, (x, y + 9))
                info_x = min(input_x - 18, x + label.get_width() + 14)
                draw_info(info_x, y + 17, self._t(definition.description))
                self.field_rects[definition.key] = rect
                pygame.draw.rect(self.screen, INPUT_COLOR, rect, border_radius=5)
                border = ACCENT if self.active_key == definition.key else GRID
                pygame.draw.rect(self.screen, border, rect, 1, border_radius=5)
                value = self.values.get(definition.key, "")
                if definition.value_type == "choice":
                    value = f"<  {language_name(self._language(), value)}  >"
                rendered_value = self._fit_number(value, self.text_font, rect.width - 18)
                self.screen.blit(rendered_value, (rect.x + 9, rect.y + 7))
            y += 44

    def _content_height(self) -> int:
        categories = len({definition.category for definition in SETTING_DEFINITIONS})
        return len(SETTING_DEFINITIONS) * 44 + categories * 36 + 8

    def _draw_button(self, rect: pygame.Rect, label: str, color: tuple[int, int, int]) -> None:
        if rect.collidepoint(pygame.mouse.get_pos()):
            color = tuple(min(255, channel + 18) for channel in color)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        font = self.button_font if self.button_font.size(label)[0] <= rect.width - 10 else self.small_font
        rendered = font.render(label, True, (255, 255, 255))
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def _fit_text(self, text: str, font: pygame.font.Font, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size(shortened + "...")[0] > width:
            shortened = shortened[:-1]
        return shortened + "..."

    def _fit_number(self, text: str, font: pygame.font.Font, width: int) -> pygame.Surface:
        if font.size(text)[0] <= width:
            return font.render(text, True, TEXT)
        shortened = text
        while shortened and font.size("..." + shortened)[0] > width:
            shortened = shortened[1:]
        return font.render("..." + shortened, True, TEXT)

    def _wrap_text(self, value: str, font: pygame.font.Font, width: int, maximum_lines: int) -> list[str]:
        words = value.split()
        lines: list[str] = []
        current = ""
        for index, word in enumerate(words):
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
            elif current and len(lines) < maximum_lines - 1:
                lines.append(current)
                current = word
            else:
                remaining = " ".join(([current] if current else []) + words[index:])
                lines.append(self._fit_text(remaining, font, width))
                return lines
        if current:
            lines.append(current)
        return lines
