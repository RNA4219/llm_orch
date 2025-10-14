import importlib
import os
import sys
from pathlib import Path

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
