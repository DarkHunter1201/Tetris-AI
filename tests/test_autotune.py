import unittest

from tetris_ai.autotune import HardwareProfile, profile_summary, recommend_runtime_settings
from tetris_ai.config import CONFIG
from tetris_ai.settings import apply_runtime_settings


class AutotuneTests(unittest.TestCase):
    def test_powerful_cuda_system_receives_large_gpu_profile(self):
        profile = HardwareProfile("Fast CPU", 16, 32, 65536, "DDR5", 6000, True, "Fast GPU", 24576, 100, (12, 0))
        config = apply_runtime_settings(CONFIG, recommend_runtime_settings(profile, "en"))
        self.assertEqual(config.population_size, 8000)
        self.assertEqual(config.hidden_sizes, (160, 96))
        self.assertEqual(config.evaluation_chunk_size, 512)
        self.assertEqual(config.neural_network_vram_limit_mib, 0)
        self.assertEqual(config.automatic_vram_fraction, 0.88)
        self.assertEqual(config.hardware_monitor_interval, 1.5)
        self.assertEqual(config.language, "en")

    def test_small_cpu_system_receives_bounded_profile(self):
        profile = HardwareProfile("Small CPU", 2, 4, 4096, "DDR3", 1600, False, "", 0, 0, (0, 0))
        config = apply_runtime_settings(CONFIG, recommend_runtime_settings(profile, "ru"))
        self.assertEqual(config.population_size, 200)
        self.assertEqual(config.hidden_sizes, (48, 32))
        self.assertEqual(config.evaluation_chunk_size, 32)
        self.assertLessEqual(config.max_pieces_per_game, 250)
        self.assertEqual(config.language, "ru")

    def test_profile_summary_reports_detected_resources(self):
        profile = HardwareProfile("CPU", 8, 16, 32768, "DDR5", 5600, True, "GPU", 16384, 50, (9, 0))
        english = profile_summary(profile, "en")
        russian = profile_summary(profile, "ru")
        self.assertIn("GPU, 16 GiB VRAM", english)
        self.assertIn("32 GiB DDR5 5600 MHz", english)
        self.assertIn("8 CPU cores", english)
        self.assertIn("8 ядер CPU", russian)


if __name__ == "__main__":
    unittest.main()
