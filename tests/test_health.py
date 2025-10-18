import importlib
import os
import sys
from pathlib import Path
import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("ORCH_CONFIG_DIR", "config")


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


def test_health():
    c = TestClient(load_app())
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_dummy():
    # use dummy providers file for offline test
    c = TestClient(load_app("1"))
    r = c.post("/v1/chat/completions", json={
        "model":"dummy",
        "messages":[{"role":"user","content":"hi"}]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_dummy_env_true_variants() -> None:
    for value in ("true", "TRUE", "yes", "Yes"):
        c = TestClient(load_app(value))
        r = c.get("/healthz")
        assert r.status_code == 200


def test_load_app_truthy_regression() -> None:
    for value in ("true", "yes"):
        app = load_app(value)
        assert isinstance(app, FastAPI)


def test_chat_missing_default_returns_400(tmp_path, monkeypatch) -> None:
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
routes:
  PLAN:
    primary: dummy
""".strip()
    )
    c = TestClient(load_app("1"))
    response = c.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert "no route configured" in response.json()["detail"]
