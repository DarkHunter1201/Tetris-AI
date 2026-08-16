import unittest

from tetris_ai.settings import SETTING_DEFINITIONS
from tetris_ai.translations import error_text, language_name, status_text, text


class TranslationTests(unittest.TestCase):
    def test_every_setting_has_complete_russian_translation(self):
        for definition in SETTING_DEFINITIONS:
            self.assertNotEqual(text("ru", definition.label), definition.label)
            self.assertNotEqual(text("ru", definition.description), definition.description)
            self.assertNotEqual(text("ru", definition.category), definition.category)

    def test_language_names_and_statuses(self):
        self.assertEqual(language_name("ru", "en"), "Английский")
        self.assertEqual(language_name("en", "ru"), "Russian")
        self.assertEqual(status_text("ru", "Training · piece 17"), "Обучение · фигура 17")
        self.assertEqual(status_text("en", "Paused"), "Paused")

    def test_validation_errors_are_localized(self):
        self.assertEqual(error_text("ru", "Invalid value for Language"), "Некорректное значение: Язык")
        self.assertIn("не может превышать", error_text("ru", "Elite agents cannot exceed agents per generation"))


if __name__ == "__main__":
    unittest.main()
