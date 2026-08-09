import builtins
import unittest
from unittest.mock import patch

import torch

from tetris_ai.device import _limit_vram, select_device
from tetris_ai.monitor import HardwareMonitor


class DeviceMonitorTests(unittest.TestCase):
    def test_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device(5000), torch.device("cpu"))

    def test_automatic_and_manual_vram_limits(self):
        self.assertEqual(_limit_vram(16000, 0, 0.85, 1024, 1024), 13600)
        self.assertEqual(_limit_vram(16000, 6000, 0.85, 1024, 1024), 6000)
        self.assertEqual(_limit_vram(16000, 20000, 0.85, 1024, 1024), 14976)
        self.assertEqual(_limit_vram(768, 0, 0.85, 1024, 1024), 576)

    def test_missing_nvml_produces_unavailable_gpu_metrics(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError(name)
            return original_import(name, *args, **kwargs)

        monitor = HardwareMonitor(torch.device("cpu"), 1.0)
        with patch("builtins.__import__", side_effect=guarded_import):
            monitor._initialize_nvml()
        monitor.poll()
        stats = monitor.snapshot()
        self.assertIsNone(stats.gpu_load)
        self.assertIsNone(stats.gpu_temperature)
        self.assertIsNone(stats.total_vram_mib)


if __name__ == "__main__":
    unittest.main()
