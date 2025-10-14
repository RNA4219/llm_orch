import importlib
import os
import sys
import textwrap
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


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


def _write_dummy_provider_config(base: Path) -> None:
    (base / "providers.dummy.toml").write_text(
        textwrap.dedent(
            """
            [dummy]
            type = "dummy"
            model = "dummy"
            base_url = ""
            rpm = 60
            concurrency = 1
            """
        ).strip()
    )


def _write_router_config_without_default(base: Path) -> None:
    (base / "router.yaml").write_text(
        textwrap.dedent(
            """
            defaults:
              temperature: 0.2
              max_tokens: 64
              task_header: "x-orch-task-kind"
            routes:
              PLAN:
                primary: dummy
            """
        ).strip()
    )


def test_chat_missing_default_includes_available_routes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCH_CONFIG_DIR", str(tmp_path))
    _write_dummy_provider_config(tmp_path)
    _write_router_config_without_default(tmp_path)

    client = TestClient(load_app("1"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "no route configured" in detail
    assert "available routes" in detail
    assert "PLAN" in detail
