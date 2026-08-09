import ast
import io
import tokenize
import unittest
from pathlib import Path


class CodePolicyTests(unittest.TestCase):
    def python_files(self):
        root = Path(__file__).resolve().parents[1]
        return sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))

    def test_python_contains_no_comments(self):
        found = []
        for path in self.python_files():
            source = path.read_text(encoding="utf-8")
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            found.extend(f"{path}:{token.start[0]}" for token in tokens if token.type == tokenize.COMMENT)
        self.assertEqual(found, [])

    def test_python_contains_no_docstrings(self):
        found = []
        for path in self.python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node, clean=False):
                    found.append(f"{path}:{getattr(node, 'lineno', 1)}")
        self.assertEqual(found, [])

    def test_python_contains_no_stub_markers(self):
        forbidden = ("TO" + "DO", "FIX" + "ME", "place" + "holder")
        found = []
        for path in self.python_files():
            source = path.read_text(encoding="utf-8")
            if any(marker.lower() in source.lower() for marker in forbidden):
                found.append(str(path))
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
