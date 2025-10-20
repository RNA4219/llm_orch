from __future__ import annotations

import asyncio
import importlib
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_server_routes import load_app

from src.orch.types import ProviderChatResponse


def test_config_refresh_loop_runs_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    configs = {
        "providers.dummy.toml": "[dummy]\ntype = \"dummy\"\nmodel = \"dummy\"\n",
        "router.yaml": (
            "defaults:\n  temperature: 0.2\nroutes:\n  PLAN:\n    primary: dummy\n"
        ),
    }
    for name, content in configs.items():
        (tmp_path / name).write_text(content)
    for key, value in {
        "ORCH_CONFIG_DIR": str(tmp_path),
        "ORCH_USE_DUMMY": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ORCH_CONFIG_REFRESH_INTERVAL", "0.01")
    sys.modules.pop("src.orch.server", None)
    sys.modules.pop("src.orch", None)
    importlib.invalidate_caches()
    server_module = importlib.import_module("src.orch.server")

    refresh_calls: list[float] = []
    event = Event()

    def fake_refresh() -> bool:
        refresh_calls.append(time.perf_counter())
        if len(refresh_calls) >= 3:
            event.set()
        return False

    monkeypatch.setattr(server_module.planner, "refresh", fake_refresh)

    with TestClient(server_module.app):
        assert event.wait(timeout=1.0)
        task = server_module._config_refresh_task
        assert task is not None
        assert not task.done()
        expected_calls = len(refresh_calls)

    time.sleep(0.05)
    assert len(refresh_calls) == expected_calls
    assert task.cancelled()


def test_reload_configuration_replaces_runtime_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {
        "providers.dummy.toml": (
            "[dummy]\n"
            "type = \"dummy\"\n"
            "model = \"initial\"\n"
            "base_url = \"http://initial\"\n"
            "rpm = 60\n"
            "concurrency = 1\n"
        ),
        "router.yaml": (
            "defaults:\n"
            "  temperature: 0.2\n"
            "  max_tokens: 64\n"
            "  task_header: \"x-orch-task-kind\"\n"
            "  task_header_value: \"PLAN\"\n"
            "routes:\n"
            "  PLAN:\n"
            "    primary: dummy\n"
        ),
    }
    for name, content in configs.items():
        (tmp_path / name).write_text(content)
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))

    load_app("1")
    server_module = sys.modules["src.orch.server"]

    assert server_module.planner._config_dir == str(tmp_path)

    provider_before = server_module.providers.get("dummy")
    assert provider_before.defn.base_url == "http://initial"
    assert server_module.cfg.router.defaults.temperature == 0.2
    assert server_module.planner.cfg.defaults.temperature == 0.2
    guard_before = server_module.guards.get("dummy")
    assert guard_before.sem._value == 1

    time.sleep(0.05)
    (tmp_path / "providers.dummy.toml").write_text(
        "[dummy]\n"
        "type = \"dummy\"\n"
        "model = \"updated\"\n"
        "base_url = \"http://updated\"\n"
        "rpm = 120\n"
        "concurrency = 2\n"
    )
    (tmp_path / "router.yaml").write_text(
        "defaults:\n"
        "  temperature: 0.5\n"
        "  max_tokens: 128\n"
        "  task_header: \"x-orch-task-kind\"\n"
        "  task_header_value: \"PLAN\"\n"
        "routes:\n"
        "  PLAN:\n"
        "    primary: dummy\n"
    )

    server_module.reload_configuration()

    provider_after = server_module.providers.get("dummy")
    assert provider_after is not provider_before
    assert provider_after.defn.base_url == "http://updated"
    assert server_module.cfg.router.defaults.temperature == 0.5
    assert server_module.planner.cfg.defaults.temperature == 0.5
    assert server_module.planner.providers["dummy"].base_url == "http://updated"
    guard_after = server_module.guards.get("dummy")
    assert guard_after.sem._value == 2


def test_immediate_request_uses_reloaded_provider_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "providers.dummy.toml").write_text(
        "[dummy_initial]\ntype = \"dummy\"\nmodel = \"dummy-model\"\nrpm = 60\nconcurrency = 1\n"
    )
    (tmp_path / "router.yaml").write_text(
        "defaults:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 64\n"
        "  task_header: \"x-orch-task-kind\"\n"
        "  task_header_value: \"PLAN\"\n"
        "routes:\n"
        "  PLAN:\n"
        "    primary: dummy_initial\n"
    )
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ORCH_USE_DUMMY", "1")

    load_app("1")
    server_module = sys.modules["src.orch.server"]
    from src.orch import rate_limiter as rate_limiter_module

    now = {"value": 1_700_000_000.0}

    def tick() -> float:
        now["value"] += 0.001
        return now["value"]

    for obj, name in [
        (server_module.time, "time"),
        (server_module.time, "perf_counter"),
        (rate_limiter_module.time, "time"),
    ]:
        monkeypatch.setattr(obj, name, tick)

    with TestClient(server_module.app) as client:
        (tmp_path / "providers.dummy.toml").write_text(
            "[dummy_reloaded]\ntype = \"dummy\"\nmodel = \"dummy-model\"\nrpm = 120\nconcurrency = 3\n"
        )
        (tmp_path / "router.yaml").write_text(
            "defaults:\n"
            "  temperature: 0.7\n"
            "  max_tokens: 128\n"
            "  task_header: \"x-orch-task-kind\"\n"
            "  task_header_value: \"PLAN\"\n"
            "routes:\n"
            "  PLAN:\n"
            "    primary: dummy_reloaded\n"
        )
        server_module.reload_configuration()

        captured = {}

        async def fake_chat(*_: object, max_tokens: int, **__: object) -> ProviderChatResponse:
            captured["max_tokens"] = max_tokens
            return ProviderChatResponse(
                status_code=200,
                model="dummy-model",
                content="dummy:hi",
                finish_reason="stop",
            )

        monkeypatch.setattr(server_module.providers.get("dummy_reloaded"), "chat", fake_chat)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        assert response.headers["x-orch-provider"] == "dummy_reloaded"
        assert captured["max_tokens"] == 128
        assert server_module.cfg.router.defaults.max_tokens == 128


def test_config_refresh_loop_applies_reload_when_planner_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {
        "providers.dummy.toml": (
            "[dummy]\n"
            "type = \"dummy\"\n"
            "model = \"initial\"\n"
            "base_url = \"http://initial\"\n"
            "rpm = 60\n"
            "concurrency = 1\n"
        ),
        "router.yaml": (
            "defaults:\n"
            "  temperature: 0.2\n"
            "  max_tokens: 64\n"
            "  task_header: \"x-orch-task-kind\"\n"
            "  task_header_value: \"PLAN\"\n"
            "routes:\n"
            "  PLAN:\n"
            "    primary: dummy\n"
        ),
    }
    for name, content in configs.items():
        (tmp_path / name).write_text(content)
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))

    load_app("1")
    server_module = sys.modules["src.orch.server"]

    provider_before = server_module.providers.get("dummy")
    assert provider_before.defn.base_url == "http://initial"

    (tmp_path / "providers.dummy.toml").write_text(
        "[dummy]\n"
        "type = \"dummy\"\n"
        "model = \"updated\"\n"
        "base_url = \"http://updated\"\n"
        "rpm = 120\n"
        "concurrency = 2\n"
    )
    (tmp_path / "router.yaml").write_text(
        "defaults:\n"
        "  temperature: 0.5\n"
        "  max_tokens: 128\n"
        "  task_header: \"x-orch-task-kind\"\n"
        "  task_header_value: \"PLAN\"\n"
        "routes:\n"
        "  PLAN:\n"
        "    primary: dummy\n"
    )

    monkeypatch.setattr(server_module, "CONFIG_REFRESH_INTERVAL", 0.0, raising=False)

    refresh_calls = 0

    def fake_refresh() -> bool:
        nonlocal refresh_calls
        refresh_calls += 1
        return refresh_calls == 1

    monkeypatch.setattr(server_module.planner, "refresh", fake_refresh)

    async def _run_loop() -> None:
        reload_event = asyncio.Event()
        original_reload = server_module.reload_configuration

        def wrapped_reload_configuration() -> None:
            original_reload()
            reload_event.set()

        monkeypatch.setattr(server_module, "reload_configuration", wrapped_reload_configuration)

        task = asyncio.create_task(server_module._config_refresh_loop())
        try:
            await asyncio.wait_for(reload_event.wait(), timeout=1.0)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(_run_loop())

    provider_after = server_module.providers.get("dummy")
    assert provider_after.defn.base_url == "http://updated"
    assert server_module.cfg.router.defaults.temperature == 0.5
    assert server_module.planner.cfg.defaults.temperature == 0.5
    assert server_module.planner.providers["dummy"].base_url == "http://updated"
    assert refresh_calls >= 1


def test_config_refresh_loop_reloads_when_planner_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("src.orch.server", None)
    importlib.invalidate_caches()
    server_module = importlib.import_module("src.orch.server")

    monkeypatch.setattr(server_module, "CONFIG_REFRESH_INTERVAL", 0.0, raising=False)

    call_log: list[str] = []

    class DummyProviders:
        def __init__(self, label: str) -> None:
            self._label = label

        def get(self, name: str) -> tuple[str, str]:
            return name, self._label

    event = asyncio.Event()
    reload_calls = 0

    class DummyPlanner:
        def __init__(self, label: str, *, trigger_reload: bool) -> None:
            self.label = label
            self.trigger_reload = trigger_reload
            self.calls = 0

        def refresh(self) -> bool:
            self.calls += 1
            call_log.append(f"refresh:{self.label}")
            if self.trigger_reload:
                if self.calls > 1:
                    pytest.fail("stale planner used after reload")
                return True
            event.set()
            return False

    initial_planner = DummyPlanner("initial", trigger_reload=True)
    updated_planner = DummyPlanner("updated", trigger_reload=False)

    monkeypatch.setattr(server_module, "providers", DummyProviders("before"), raising=False)
    monkeypatch.setattr(server_module, "planner", initial_planner, raising=False)

    def fake_reload_configuration() -> None:
        nonlocal reload_calls
        reload_calls += 1
        call_log.append("reload")
        server_module.providers = DummyProviders("after")
        server_module.planner = updated_planner

    monkeypatch.setattr(server_module, "reload_configuration", fake_reload_configuration)

    async def _run_loop() -> None:
        task = asyncio.create_task(server_module._config_refresh_loop())
        try:
            await asyncio.wait_for(event.wait(), timeout=1.0)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(_run_loop())

    assert reload_calls == 1
    assert server_module.planner is updated_planner
    assert updated_planner.calls >= 1
    assert server_module.providers.get("dummy") == ("dummy", "after")
    assert call_log[:3] == ["refresh:initial", "reload", "refresh:updated"]

