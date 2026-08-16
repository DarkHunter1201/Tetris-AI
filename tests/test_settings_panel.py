import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from tetris_ai.config import CONFIG
from tetris_ai.settings import RuntimeSettings
from tetris_ai.settings_panel import SettingsPanel


class SettingsPanelTests(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.screen = pygame.display.set_mode((940, 820))
        self.applied: list[RuntimeSettings] = []
        self.panel = SettingsPanel(
            self.screen,
            pygame.font.SysFont("segoeui", 30, bold=True),
            pygame.font.SysFont("segoeui", 18, bold=True),
            pygame.font.SysFont("consolas", 15),
            pygame.font.SysFont("segoeui", 14),
            pygame.font.SysFont("segoeui", 16, bold=True),
            self.applied.append,
        )
        self.panel.open(CONFIG)

    def tearDown(self):
        pygame.quit()

    def test_panel_scrolls_and_exposes_info_for_visible_fields(self):
        descriptions = []
        self.panel.draw(lambda x, y, text: descriptions.append(text))
        top_fields = set(self.panel.field_rects)
        self.assertGreater(self.panel.maximum_scroll, 0)
        self.assertTrue(descriptions)
        self.panel.scroll = self.panel.maximum_scroll
        descriptions.clear()
        self.panel.draw(lambda x, y, text: descriptions.append(text))
        self.assertTrue(descriptions)
        self.assertNotEqual(top_fields, set(self.panel.field_rects))

    def test_input_replacement_and_confirmed_apply(self):
        self.panel.draw(lambda x, y, text: None)
        target = self.panel.field_rects["population_size"]
        click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=target.center)
        self.panel.handle_event(click)
        self.panel.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2, unicode="2", mod=0))
        self.assertEqual(self.panel.values["population_size"], "2")
        self.panel.values["elite_count"] = "1"
        self.panel.values["parent_pool_size"] = "2"
        self.assertEqual(self.panel._apply(), "none")
        self.assertEqual(self.panel._apply(), "applied_reset")
        self.assertEqual(self.applied[0].population_size, 2)

    def test_invalid_value_stays_open_and_displays_error(self):
        self.panel.values["population_size"] = "2"
        self.panel.values["elite_count"] = "3"
        self.panel._apply()
        self.assertEqual(self.panel._apply(), "none")
        self.assertTrue(self.panel.visible)
        self.assertIn("не может превышать", self.panel.error)

    def test_language_choice_and_defaults_are_available_without_applying(self):
        self.panel.draw(lambda x, y, text: None)
        language = self.panel.field_rects["language"]
        self.panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=language.center))
        self.assertEqual(self.panel.values["language"], "en")
        self.panel.values["board_width"] = "17"
        self.panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=self.panel.defaults_rect.center))
        self.assertEqual(self.panel.values["board_width"], "10")
        self.assertEqual(self.panel.values["language"], "en")
        self.assertEqual(self.applied, [])

    def test_language_only_apply_does_not_request_training_reset(self):
        self.panel.values["language"] = "en"
        self.assertEqual(self.panel._apply(), "applied")
        self.assertEqual(self.applied[0].overrides["language"], "en")

    def test_auto_button_loads_recommendation_as_draft(self):
        self.panel.auto_settings = lambda language: (RuntimeSettings(population_size=7), "Test PC")
        self.panel.draw(lambda x, y, text: None)
        self.panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=self.panel.auto_rect.center))
        self.assertEqual(self.panel.values["population_size"], "7")
        self.assertIn("Test PC", self.panel.notice)
        self.assertEqual(self.applied, [])


if __name__ == "__main__":
    unittest.main()
