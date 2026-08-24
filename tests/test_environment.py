import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from logic.environment import load_project_env


class ProjectEnvironmentTests(unittest.TestCase):
    def test_dotenv_loads_but_explicit_process_values_win(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry_point = root / "api.py"
            (root / ".env").write_text(
                "RAG_ENABLED=true\nAPI_PORT=9999\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {"RAG_ENABLED": "false"}, clear=True):
                load_project_env(entry_point)
                self.assertEqual(os.environ["RAG_ENABLED"], "false")
                self.assertEqual(os.environ["API_PORT"], "9999")


if __name__ == "__main__":
    unittest.main()
