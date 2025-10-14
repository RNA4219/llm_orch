from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from src.orch.router import RouteDef, RoutePlanner


def load_app(dummy_env: str | None = None) -> FastAPI:
    module_name = "src.orch.server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("src.orch", None)
    if dummy_env is None:
        os.environ.pop("ORCH_USE_DUMMY", None)
    else:
        os.environ["ORCH_USE_DUMMY"] = dummy_env
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module.app


@pytest.fixture(name="route_test_config")
def fixture_route_test_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))
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
routes:
  PLAN:
    primary: dummy
""".strip()
    )
    return tmp_path


def test_chat_missing_route_and_default_returns_400(route_test_config: Path) -> None:
    client = TestClient(load_app("1"))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "IDEATE"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "no route configured for task 'IDEATE' and no DEFAULT route defined in router configuration."
    )


def test_chat_missing_header_uses_default_task(route_test_config: Path) -> None:
    client = TestClient(load_app("1"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "dummy:hi"


def test_chat_missing_header_routes_to_task_header_default(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(load_app("1"))
    from src.orch.router import RouteDef as RouterRouteDef, RoutePlanner as RouterRoutePlanner

    recorded_tasks: list[str] = []
    original_plan: Callable[[RouterRoutePlanner, str], RouterRouteDef] = RouterRoutePlanner.plan

    def recording_plan(self: RouterRoutePlanner, task: str) -> RouterRouteDef:
        recorded_tasks.append(task)
        return original_plan(self, task)

    monkeypatch.setattr(RouterRoutePlanner, "plan", recording_plan)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert recorded_tasks == ["PLAN"]


def test_chat_custom_task_header_routes_only_with_configured_header(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router_file = route_test_config / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.2
  max_tokens: 64
  task_header: "x-orch-custom-task"
  task_header_value: "PLAN"
routes:
  PLAN:
    primary: dummy
  CUSTOM:
    primary: dummy
""".strip()
    )

    client = TestClient(load_app("1"))
    from src.orch.router import RouteDef as RouterRouteDef, RoutePlanner as RouterRoutePlanner

    recorded_tasks: list[str] = []
    original_plan: Callable[[RouterRoutePlanner, str], RouterRouteDef] = RouterRoutePlanner.plan

    def recording_plan(self: RouterRoutePlanner, task: str) -> RouterRouteDef:
        recorded_tasks.append(task)
        return original_plan(self, task)

    monkeypatch.setattr(RouterRoutePlanner, "plan", recording_plan)

    response_default = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response_default.status_code == 200
    assert recorded_tasks == ["PLAN"]

    recorded_tasks.clear()

    response_custom = client.post(
        "/v1/chat/completions",
        headers={"x-orch-custom-task": "CUSTOM"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response_custom.status_code == 200
    assert recorded_tasks == ["CUSTOM"]
