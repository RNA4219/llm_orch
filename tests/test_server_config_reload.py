from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_server_routes import load_app


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
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ORCH_USE_DUMMY", "1")
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


def test_http_request_uses_reloaded_provider_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_bundle(
        provider: str, max_tokens: int, concurrency: int, timestamp: float
    ) -> None:
        providers_body = "\n".join(
            [
                f"[{provider}]",
                'type = "dummy"',
                'model = "dummy"',
                f'base_url = "http://{provider}"',
                "rpm = 60",
                f"concurrency = {concurrency}",
                "",
            ]
        )
        router_body = "\n".join(
            [
                "defaults:",
                "  temperature: 0.2",
                f"  max_tokens: {max_tokens}",
                '  task_header: "x-orch-task-kind"',
                '  task_header_value: "PLAN"',
                "routes:",
                "  PLAN:",
                f"    primary: {provider}",
                "",
            ]
        )
        for name, body in (
            ("providers.dummy.toml", providers_body),
            ("router.yaml", router_body),
        ):
            path = tmp_path / name
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(body)
            os.replace(temp, path)
            os.utime(path, (timestamp, timestamp))

    write_bundle("alpha", 64, 1, 1_000_000.0)
    for key, value in (
        ("ORCH_CONFIG_DIR", str(tmp_path)),
        ("ORCH_USE_DUMMY", "1"),
        ("ORCH_INBOUND_API_KEYS", "test-key"),
    ):
        monkeypatch.setenv(key, value)

    load_app("1")
    server_module = sys.modules["src.orch.server"]
    monkeypatch.setattr(server_module.planner, "refresh", lambda *_, **__: False)

    from src.orch.providers import DummyProvider
    from src.orch.types import ProviderChatResponse

    call_state = SimpleNamespace(value=None)

    async def fake_chat(
        self: DummyProvider,
        model: str,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> ProviderChatResponse:  # type: ignore[override]
        call_state.value = (self.defn.name, int(kwargs.get("max_tokens", 0)))
        return ProviderChatResponse(
            status_code=200,
            model="dummy",
            content=self.defn.name,
            finish_reason="stop",
        )

    monkeypatch.setattr(DummyProvider, "chat", fake_chat, raising=False)

    with TestClient(server_module.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "test-key"},
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200

        write_bundle("beta", 128, 2, 1_000_010.0)
        server_module.reload_configuration()

        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "test-key"},
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "after"}],
            },
        )
        assert resp.status_code == 200
        assert (resp.headers["x-orch-provider"], call_state.value) == (
            "beta",
            ("beta", 128),
        )


def test_config_refresh_loop_discards_old_planner_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("src.orch.server", None)
    importlib.invalidate_caches()
    server_module = importlib.import_module("src.orch.server")
    event = asyncio.Event()
    initial_state = SimpleNamespace(count=0)
    updated_state = SimpleNamespace(count=0)

    def initial_refresh() -> bool:
        initial_state.count += 1
        assert initial_state.count == 1, "stale planner refresh executed"
        return True

    def updated_refresh() -> bool:
        updated_state.count += 1
        event.set()
        return False

    server_module.planner = SimpleNamespace(refresh=initial_refresh)
    updated_planner = SimpleNamespace(refresh=updated_refresh)

    def fake_reload() -> None:
        server_module.planner = updated_planner

    monkeypatch.setattr(server_module, "reload_configuration", fake_reload)

    async def runner() -> None:
        task = asyncio.create_task(server_module._config_refresh_loop())
        try:
            await asyncio.wait_for(event.wait(), timeout=1.0)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(runner())

    assert initial_state.count == 1
    assert updated_state.count >= 1 and server_module.planner is updated_planner
