import os
os.environ.setdefault("ORCH_CONFIG_DIR", "config")
from fastapi.testclient import TestClient
from src.orch.server import app

def test_health():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_dummy():
    # use dummy providers file for offline test
    os.environ["ORCH_USE_DUMMY"] = "1"
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={
        "model":"dummy",
        "messages":[{"role":"user","content":"hi"}]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
