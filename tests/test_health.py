import importlib
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.orch.server import app
from src.orch import server as server_mod

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
    os.environ["ORCH_USE_DUMMY"] = "1"
    server_mod.init_dependencies(use_dummy=True)
    c = TestClient(app)
    c = TestClient(load_app("1"))
    r = c.post("/v1/chat/completions", json={
        "model":"dummy",
        "messages":[{"role":"user","content":"hi"}]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_reinitialize_dependencies_for_dummy_provider():
    os.environ["ORCH_USE_DUMMY"] = "0"
    server_mod.init_dependencies(use_dummy=False)
    os.environ["ORCH_USE_DUMMY"] = "1"
    server_mod.init_dependencies(use_dummy=True)
    c = TestClient(app)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["choices"][0]["message"]["content"].startswith("dummy:")
def test_dummy_env_true_variants() -> None:
    for value in ("true", "TRUE", "yes", "Yes"):
        c = TestClient(load_app(value))
        r = c.get("/healthz")
        assert r.status_code == 200
