import unittest
from pathlib import Path

from tetris_ai.paths import PROJECT_ROOT, RUNTIME_PATHS, ensure_inside_project


class PathTests(unittest.TestCase):
    def test_all_runtime_paths_are_inside_project(self):
        for path in RUNTIME_PATHS.values():
            path.resolve().relative_to(PROJECT_ROOT.resolve())

    def test_outside_path_is_rejected(self):
        outside = PROJECT_ROOT.resolve().parent / "outside-tetris-ai"
        with self.assertRaises(ValueError):
            ensure_inside_project(outside)


if __name__ == "__main__":
    unittest.main()
