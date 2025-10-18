from __future__ import annotations

import importlib
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.types import ProviderChatResponse


def _load_app(config_dir: Path) -> tuple[Any, Any]:
    assert config_dir.exists(), "configuration directory must exist"
    module_name = "src.orch.server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("src.orch", None)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module.app, module


@pytest.fixture(name="sticky_config")
def fixture_sticky_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ORCH_USE_DUMMY", "1")
    providers_file = tmp_path / "providers.dummy.toml"
    providers_file.write_text(
        """
[dummy]
type = "dummy"
model = "dummy"
base_url = ""
rpm = 60
concurrency = 1
""".strip()
    )
    router_file = tmp_path / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.2
  max_tokens: 64
  task_header: "x-orch-task-kind"
  task_header_value: "PLAN"
  sticky_header: "x-orch-sticky-key"
routes:
  PLAN:
    primary: dummy
""".strip()
    )
    return tmp_path


class DummyPlanner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.record_success_calls: list[str] = []
        self.record_failure_calls: list[str] = []

    def plan(self, task: str, *, sticky_key: str | None = None) -> Any:
        self.calls.append((task, sticky_key))

        class _Route:
            primary = "dummy"
            fallback: list[str] = []

        return _Route()

    def record_success(self, provider: str) -> None:
        self.record_success_calls.append(provider)

    def record_failure(self, provider: str, *, now: float | None = None) -> None:
        self.record_failure_calls.append(provider)


class DummyGuard:
    def acquire(self, *, estimated_prompt_tokens: int):
        @asynccontextmanager
        async def _ctx() -> Any:
            yield object()

        return _ctx()

    def record_usage(
        self,
        _lease: object,
        *,
        usage_prompt_tokens: int,
        usage_completion_tokens: int,
    ) -> None:
        return None


class DummyProvider:
    model = "dummy"

    async def chat(self, *args: Any, **kwargs: Any) -> ProviderChatResponse:
        return ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
            usage_prompt_tokens=1,
            usage_completion_tokens=1,
        )


async def _noop_metrics_writer(record: dict[str, object]) -> None:
    return None


def _prepare_app(sticky_config: Path) -> tuple[TestClient, DummyPlanner]:
    app, server_module = _load_app(sticky_config)
    planner = DummyPlanner()
    guard = DummyGuard()
    providers = DummyProvider()
    server_module.planner = planner  # type: ignore[assignment]
    server_module.guards = type("_Guards", (), {"get": lambda self, name: guard})()
    server_module.providers = type("_Providers", (), {"get": lambda self, name: providers})()
    server_module.metrics.write = _noop_metrics_writer  # type: ignore[attr-defined]
    client = TestClient(app)
    return client, planner


def test_plan_receives_sticky_key_from_header(sticky_config: Path) -> None:
    client, planner = _prepare_app(sticky_config)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "PLAN", "x-orch-sticky-key": "tenant-42"},
        json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert planner.calls == [("PLAN", "tenant-42")]


def test_plan_receives_none_when_header_missing(sticky_config: Path) -> None:
    client, planner = _prepare_app(sticky_config)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "PLAN"},
        json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert planner.calls == [("PLAN", None)]
