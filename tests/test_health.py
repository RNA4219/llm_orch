import importlib
import os
import sys
import textwrap
from pathlib import Path

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


def test_load_app_with_undefined_provider(tmp_path: Path) -> None:
    config_dir = tmp_path
    (config_dir / "providers.toml").write_text(
        textwrap.dedent(
            """
            [known]
            type = "dummy"
            base_url = "http://example.test"
            model = "dummy"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "router.yaml").write_text(
        textwrap.dedent(
            """
            defaults: { temperature: 0.1, max_tokens: 1024, task_header: "x-orch-task-kind" }
            routes:
              DEFAULT: { primary: missing, fallback: [known] }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    prev_dir = os.environ.get("ORCH_CONFIG_DIR")
    os.environ["ORCH_CONFIG_DIR"] = str(config_dir)
    try:
        with pytest.raises(ValueError, match="Route 'DEFAULT' references undefined provider 'missing'"):
            load_app()
    finally:
        if prev_dir is None:
            os.environ.pop("ORCH_CONFIG_DIR", None)
        else:
            os.environ["ORCH_CONFIG_DIR"] = prev_dir
