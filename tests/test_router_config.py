import builtins
import sys
import types
import importlib
import random
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.router import RoutePlanner, load_config
from src.orch.providers import ProviderRegistry


def write_config(tmp_path, provider_type: str = "mock", concurrency: int = 4):
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
concurrency = {concurrency}
""".format(provider_type=provider_type, concurrency=concurrency)
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


def write_router(tmp_path, text: str) -> str:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "router.yaml").write_text(text, encoding="utf-8")
    return str(config_dir)


@pytest.mark.parametrize("invalid_concurrency", [0, -1])
def test_load_config_rejects_non_positive_concurrency(tmp_path, invalid_concurrency):
    config_dir = write_config(tmp_path, concurrency=invalid_concurrency)

    with pytest.raises(ValueError) as excinfo:
        load_config(config_dir)

    message = str(excinfo.value)
    assert "concurrency" in message
    assert "alpha" in message


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


def _write_providers(tmp_path) -> str:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "providers.toml").write_text(
        """
[alpha]
type = "mock"
base_url = "https://example.com"
model = "gpt"
auth_env = "TOKEN"
rpm = 60
concurrency = 4

[beta]
type = "mock"
base_url = "https://example.com"
model = "gpt"
auth_env = "TOKEN"
rpm = 60
concurrency = 4
""",
        encoding="utf-8",
    )
    return str(config_dir)
def test_load_config_succeeds_without_tomllib(monkeypatch, tmp_path):
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

[beta]
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

    original_import = builtins.__import__
    real_tomllib = importlib.import_module("tomllib")
    fake_tomli = types.ModuleType("tomli")
    fake_tomli.load = real_tomllib.load

    def _fake_import(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError("No module named 'tomllib'")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tomllib", raising=False)
    monkeypatch.delitem(sys.modules, "src.orch.router", raising=False)
    monkeypatch.setitem(sys.modules, "tomli", fake_tomli)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    router_module = importlib.import_module("src.orch.router")
    importlib.reload(router_module)

    loaded = router_module.load_config(str(config_dir))

    assert set(loaded.providers) == {"alpha", "beta"}


def test_route_config_weighted_strategy(tmp_path):
    config_dir = Path(_write_providers(tmp_path))
    write_router(
        tmp_path,
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-task
routes:
  weighted-task:
    strategy: weighted
    targets:
      - provider: alpha
        weight: 3
        circuit_breaker:
          failure_threshold: 2
          recovery_time_s: 5
      - provider: beta
        weight: 1
  DEFAULT:
    primary: alpha
""",
    )
    loaded = load_config(str(config_dir))
    route = loaded.router.routes["weighted-task"]
    assert route.strategy == "weighted"
    assert route.targets[0].circuit_breaker is not None
    planner = RoutePlanner(loaded.router, loaded.providers)
    rand = random.Random(0)
    primaries = {planner.plan("weighted-task", rand=rand).primary for _ in range(8)}
    assert primaries == {"alpha", "beta"}
    selection = planner.plan("weighted-task", rand=rand)
    assert selection.primary in {"alpha", "beta"}
    assert set(selection.fallback) == {"alpha", "beta"} - {selection.primary}


def test_route_config_priority_strategy_and_default(tmp_path):
    config_dir = Path(_write_providers(tmp_path))
    write_router(
        tmp_path,
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-task
routes:
  priority-task:
    strategy: priority
    targets:
      - provider: beta
      - provider: alpha
  DEFAULT:
    primary: alpha
    fallback:
      - beta
""",
    )
    loaded = load_config(str(config_dir))
    planner = RoutePlanner(loaded.router, loaded.providers)
    selection = planner.plan("priority-task")
    assert selection.primary == "beta"
    assert selection.fallback == ["alpha"]
    default_selection = planner.plan("unknown-task")
    assert default_selection.primary == "alpha"
    assert default_selection.fallback == ["beta"]


def test_route_config_sticky_strategy(tmp_path):
    config_dir = Path(_write_providers(tmp_path))
    write_router(
        tmp_path,
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-task
routes:
  sticky-task:
    strategy: sticky
    sticky_ttl: 5
    targets:
      - provider: alpha
        circuit_breaker:
          failure_threshold: 1
          recovery_time_s: 10
      - provider: beta
  DEFAULT:
    primary: alpha
""",
    )
    loaded = load_config(str(config_dir))
    planner = RoutePlanner(loaded.router, loaded.providers)
    first = planner.plan("sticky-task", sticky_key="user", now=0.0)
    assert first.primary == "alpha"
    repeat = planner.plan("sticky-task", sticky_key="user", now=3.0)
    assert repeat.primary == "alpha"
    planner.record_failure("alpha", now=3.5)
    rerouted = planner.plan("sticky-task", sticky_key="user", now=4.0)
    assert rerouted.primary == "beta"
    post_recovery = planner.plan("sticky-task", sticky_key="user", now=20.0)
    assert post_recovery.primary == "alpha"


def test_route_config_circuit_breaker_respected(tmp_path):
    config_dir = Path(_write_providers(tmp_path))
    write_router(
        tmp_path,
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-task
routes:
  priority-task:
    targets:
      - provider: alpha
        circuit_breaker:
          failure_threshold: 2
          recovery_time_s: 5
      - provider: beta
  DEFAULT:
    primary: alpha
""",
    )
    loaded = load_config(str(config_dir))
    planner = RoutePlanner(loaded.router, loaded.providers)
    now = 0.0
    planner.record_failure("alpha", now=now)
    planner.record_failure("alpha", now=now + 1)
    routed = planner.plan("priority-task", now=now + 2)
    assert routed.primary == "beta"
    restored = planner.plan("priority-task", now=now + 10)
    assert restored.primary == "alpha"


def test_route_config_schema_validation(tmp_path):
    config_dir = Path(_write_providers(tmp_path))
    write_router(
        tmp_path,
        """
defaults:
  temperature: 0.1
  max_tokens: 128
  task_header: x-task
routes:
  weighted-task:
    strategy: weighted
    targets:
      - provider: alpha
      - provider: beta
""",
    )
    with pytest.raises(ValueError) as excinfo:
        load_config(str(config_dir))
    assert "weighted-task" in str(excinfo.value)
