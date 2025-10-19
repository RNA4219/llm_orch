from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient


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
