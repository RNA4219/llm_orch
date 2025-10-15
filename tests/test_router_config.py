import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.orch.router as router_module
from src.orch.providers import ProviderRegistry


def write_config(tmp_path, provider_type: str = "mock"):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "providers.toml").write_text(
        """
[alpha]
type = "{provider_type}"
base_url = "https://example.com"
model = "gpt"
auth_env = "TOKEN"
rpm = 60
concurrency = 4
""".format(provider_type=provider_type)
    )
    (config_dir / "router.yaml").write_text(
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-orch-task-kind
routes:
  task-a:
    primary: beta
    fallback:
      - alpha
""",
        encoding="utf-8",
    )
    return str(config_dir)


def test_load_config_fails_for_unknown_provider(tmp_path):
    config_dir = write_config(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        router_module.load_config(config_dir)

    assert "beta" in str(excinfo.value)


def test_provider_registry_rejects_unknown_provider_type(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "providers.toml").write_text(
        """
[alpha]
type = "unknown"
base_url = "https://example.com"
model = "gpt"
auth_env = "TOKEN"
rpm = 60
concurrency = 4
"""
    )
    (config_dir / "router.yaml").write_text(
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-orch-task-kind
routes:
  task-a:
    primary: alpha
""",
        encoding="utf-8",
    )
    loaded = router_module.load_config(str(config_dir))

    with pytest.raises(ValueError) as excinfo:
        ProviderRegistry(loaded.providers)

    message = str(excinfo.value)
    assert "alpha" in message
    assert "unknown" in message


def test_load_config_without_tomllib(monkeypatch, tmp_path):
    config_dir_path = tmp_path / "config"
    config_dir_path.mkdir()
    (config_dir_path / "providers.toml").write_text(
        """
[alpha]
type = "mock"
base_url = "https://example.com"
model = "gpt"
auth_env = "TOKEN"
rpm = 60
concurrency = 4
"""
    )
    (config_dir_path / "router.yaml").write_text(
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-orch-task-kind
routes:
  task-a:
    primary: alpha
""",
        encoding="utf-8",
    )
    config_dir = str(config_dir_path)

    original_tomllib = sys.modules.get("tomllib")
    monkeypatch.delitem(sys.modules, "tomllib", raising=False)
    monkeypatch.delitem(sys.modules, "src.orch.router", raising=False)

    original_import = builtins.__import__

    fake_tomli = types.ModuleType("tomli")
    if original_tomllib is None:
        pytest.skip("tomllib module not available to build tomli shim")
    fake_tomli.load = original_tomllib.load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tomli", fake_tomli)

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tomllib":
            raise ModuleNotFoundError("mocked missing tomllib")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    reloaded = importlib.import_module("src.orch.router")
    loaded = reloaded.load_config(config_dir)

    assert loaded.providers["alpha"].name == "alpha"
