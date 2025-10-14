import os
os.environ.setdefault("ORCH_CONFIG_DIR", "config")
from fastapi.testclient import TestClient
from src.orch.server import app
from src.orch import server as server_mod

def test_health():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_dummy():
    # use dummy providers file for offline test
    os.environ["ORCH_USE_DUMMY"] = "1"
    server_mod.init_dependencies(use_dummy=True)
    c = TestClient(app)
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


def test_dummy_env_true_variants():
    for value in ("1", "true", "True", "TRUE"):
        os.environ["ORCH_USE_DUMMY"] = value
        server_mod.init_dependencies(
            use_dummy=server_mod.resolve_dummy_flag_from_env()
        )
        c = TestClient(app)
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": value}],
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"].startswith("dummy:")
