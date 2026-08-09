import shutil
import unittest
import uuid

from tetris_ai.paths import RUNTIME_PATHS, ensure_inside_project
from tetris_ai.settings import RuntimeSettings, SettingsManager


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = ensure_inside_project(RUNTIME_PATHS["temp"] / f"settings-test-{uuid.uuid4().hex}")
        self.directory.mkdir(parents=True)
        self.manager = SettingsManager(self.directory / "settings.json")

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_vram_limit_round_trip(self):
        self.manager.save(RuntimeSettings(vram_limit_mib=8192))
        self.assertEqual(self.manager.load().vram_limit_mib, 8192)
        self.assertFalse((self.directory / "settings.json.tmp").exists())

    def test_missing_and_invalid_settings_use_auto(self):
        self.assertEqual(self.manager.load().vram_limit_mib, 0)
        self.manager.path.write_text("invalid", encoding="utf-8")
        self.assertEqual(self.manager.load().vram_limit_mib, 0)


if __name__ == "__main__":
    unittest.main()
