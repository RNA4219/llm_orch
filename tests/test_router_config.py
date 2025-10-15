from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.router import load_config
from src.orch.providers import ProviderRegistry


def write_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "providers.toml").write_text(
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
        load_config(config_dir)

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
    loaded = load_config(str(config_dir))

    with pytest.raises(ValueError) as excinfo:
        ProviderRegistry(loaded.providers)

    message = str(excinfo.value)
    assert "alpha" in message
    assert "unknown" in message
