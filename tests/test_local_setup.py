import unittest
import importlib.metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from logic.token_budget import TokenCounter
from tools import check_local_setup as setup_module

from tools.check_local_setup import (
    SetupCheckError,
    check_setup,
    verify_install_report,
    verify_wheel_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class LauncherConfigurationTests(unittest.TestCase):
    def test_launcher_uses_canonical_files_cpu_wheel_and_loopback(self):
        launcher = (ROOT / "run_local.ps1").read_text(encoding="utf-8")
        self.assertIn("--requirement requirements.txt", launcher)
        self.assertIn("--constraint constraints.txt", launcher)
        self.assertIn("--only-binary=llama-cpp-python", launcher)
        self.assertIn("abetlen.github.io/llama-cpp-python/whl/cpu/", launcher)
        self.assertIn('"--host", "127.0.0.1"', launcher)


class WheelVerificationTests(unittest.TestCase):
    def _report(self, *, version="0.3.35", url=None):
        return {
            "install": [
                {
                    "metadata": {
                        "name": "llama-cpp-python",
                        "version": version,
                    },
                    "download_info": {
                        "url": url
                        or "https://github.com/abetlen/llama-cpp-python/releases/"
                        "download/v0.3.35/"
                        "llama_cpp_python-0.3.35-py3-none-win_amd64.whl"
                    },
                }
            ]
        }

    def test_official_windows_wheel_report_is_accepted(self):
        verify_install_report(self._report(), "0.3.35")

    def test_source_or_wrong_platform_report_is_rejected(self):
        for url in (
            "https://files.pythonhosted.org/llama_cpp_python-0.3.35.tar.gz",
            "https://abetlen.github.io/llama_cpp_python-0.3.35-py3-none-linux.whl",
        ):
            with self.subTest(url=url), self.assertRaises(SetupCheckError):
                verify_install_report(self._report(url=url), "0.3.35")

    def test_wrong_report_version_is_rejected(self):
        with self.assertRaises(SetupCheckError):
            verify_install_report(self._report(version="0.3.34"), "0.3.35")

    def test_installed_wheel_metadata_requires_windows_binary_tag(self):
        distribution = Mock(version="0.3.35")
        distribution.read_text.return_value = (
            "Wheel-Version: 1.0\nTag: py3-none-win_amd64\n"
        )
        verify_wheel_metadata(distribution, "0.3.35")
        distribution.read_text.return_value = "Wheel-Version: 1.0\nTag: py3-none-any\n"
        with self.assertRaises(SetupCheckError):
            verify_wheel_metadata(distribution, "0.3.35")


class SetupCheckerTests(unittest.TestCase):
    def _paths(self, root):
        constraints = root / "constraints.txt"
        constraints.write_text(
            "chromadb==1.3.5\n"
            "sentence-transformers==5.1.2\n"
            "llama-cpp-python==0.3.35\n",
            encoding="utf-8",
        )
        e5 = root / "e5"
        e5.mkdir()
        gguf = root / "model.gguf"
        gguf.write_bytes(b"gguf")
        return constraints, e5, gguf

    def test_missing_package_wrong_version_assets_and_tokenizer_are_rejected(self):
        versions = {
            "chromadb": "1.3.5",
            "sentence-transformers": "5.1.2",
            "llama-cpp-python": "0.3.35",
        }
        with TemporaryDirectory() as temporary:
            constraints, e5, gguf = self._paths(Path(temporary))
            with (
                patch("tools.check_local_setup.platform.system", return_value="Linux"),
                patch.object(setup_module.importlib, "import_module"),
                patch.object(
                    setup_module.importlib.metadata,
                    "version",
                    side_effect=lambda name: versions[name],
                ),
                patch.object(TokenCounter, "load", return_value=Mock(mode="gguf_native")),
            ):
                check_setup(
                    constraints_path=constraints,
                    e5_path=e5,
                    gguf_path=gguf,
                    install_report_path=None,
                )

    def test_windows_setup_requires_install_report(self):
        versions = {
            "chromadb": "1.3.5",
            "sentence-transformers": "5.1.2",
            "llama-cpp-python": "0.3.35",
        }
        distribution = Mock(version="0.3.35")
        distribution.read_text.return_value = "Tag: py3-none-win_amd64\n"
        with TemporaryDirectory() as temporary:
            constraints, e5, gguf = self._paths(Path(temporary))
            with (
                self.assertRaisesRegex(SetupCheckError, "install_report_missing"),
                patch("tools.check_local_setup.platform.system", return_value="Windows"),
                patch("tools.check_local_setup.platform.machine", return_value="AMD64"),
                patch.object(setup_module.importlib, "import_module"),
                patch.object(
                    setup_module.importlib.metadata,
                    "version",
                    side_effect=lambda name: versions[name],
                ),
                patch.object(
                    setup_module.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
            ):
                check_setup(
                    constraints_path=constraints,
                    e5_path=e5,
                    gguf_path=gguf,
                    install_report_path=None,
                )

            cases = (
                ("package", {**versions, "chromadb": "0.0"}, e5, "gguf_native"),
                ("e5", versions, e5 / "missing", "gguf_native"),
                ("tokenizer", versions, e5, "utf8_upper_bound"),
            )
            for name, installed, model_path, tokenizer_mode in cases:
                with self.subTest(name=name), self.assertRaises(SetupCheckError), (
                    patch("tools.check_local_setup.platform.system", return_value="Linux")
                ), patch.object(setup_module.importlib, "import_module"), patch.object(
                    setup_module.importlib.metadata,
                    "version",
                    side_effect=lambda package, values=installed: values[package],
                ), patch.object(
                    TokenCounter, "load", return_value=Mock(mode=tokenizer_mode)
                ):
                    check_setup(
                        constraints_path=constraints,
                        e5_path=model_path,
                        gguf_path=gguf,
                        install_report_path=None,
                    )

            def missing_chroma(package):
                if package == "chromadb":
                    raise importlib.metadata.PackageNotFoundError(package)
                return versions[package]

            with self.assertRaises(SetupCheckError), patch.object(
                setup_module.importlib.metadata,
                "version",
                side_effect=missing_chroma,
            ):
                check_setup(
                    constraints_path=constraints,
                    e5_path=e5,
                    gguf_path=gguf,
                    install_report_path=None,
                )


if __name__ == "__main__":
    unittest.main()
