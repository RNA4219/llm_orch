from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from src.orch.router import RouteDef, RoutePlanner


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


@pytest.fixture(name="route_test_config")
def fixture_route_test_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
  task_header_value: "PLAN"
routes:
  PLAN:
    primary: dummy
""".strip()
    )
    return tmp_path


def capture_metric_records(
    server_module: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    async def capture(record: dict[str, object]) -> None:
        records.append(record)

    monkeypatch.setattr(server_module.metrics, "write", capture)
    return records


def assert_single_req_id(records: list[dict[str, object]]) -> None:
    assert records
    req_ids = {record.get("req_id") for record in records}
    assert len(req_ids) == 1
    req_id = req_ids.pop()
    assert isinstance(req_id, str)
    assert req_id


@pytest.mark.parametrize(
    "request_overrides",
    [
        {},
        {"temperature": None, "max_tokens": None},
    ],
)
def test_chat_applies_router_defaults_for_optional_fields(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch, request_overrides: dict[str, Any]
) -> None:
    router_file = route_test_config / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.55
  max_tokens: 512
  task_header: "x-orch-task-kind"
  task_header_value: "PLAN"
routes:
  PLAN:
    primary: dummy
""".strip()
    )

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.types import ProviderChatResponse

    provider_response = ProviderChatResponse(
        status_code=200,
        model="dummy",
        content="dummy:hi",
        usage_prompt_tokens=0,
        usage_completion_tokens=0,
    )
    chat_mock = AsyncMock(return_value=provider_response)

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = chat_mock

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    request_body: dict[str, Any] = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "hi"}],
    }
    request_body.update(request_overrides)

    response = client.post(
        "/v1/chat/completions",
        json=request_body,
    )

    assert response.status_code == 200
    assert chat_mock.await_count == 1
    args, kwargs = chat_mock.await_args
    assert args == ("dummy", [{"role": "user", "content": "hi"}])
    assert kwargs["temperature"] == 0.55
    assert kwargs["max_tokens"] == 512


def test_chat_missing_route_and_default_returns_400(route_test_config: Path) -> None:
    client = TestClient(load_app("1"))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "IDEATE"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "no route configured for task 'IDEATE' and no DEFAULT route defined in router configuration."
    )


def test_chat_missing_header_uses_default_task(route_test_config: Path) -> None:
    client = TestClient(load_app("1"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "dummy:hi"


def test_chat_metrics_records_response_model_when_provider_model_missing(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_file = route_test_config / "providers.dummy.toml"
    providers_file.write_text(
        """
[dummy]
type = "dummy"
model = ""
base_url = ""
rpm = 60
concurrency = 1
""".strip()
    )

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    provider_response = ProviderChatResponse(
        status_code=200,
        model="response-model",
        content="dummy:hi",
        usage_prompt_tokens=0,
        usage_completion_tokens=0,
    )
    chat_mock = AsyncMock(return_value=provider_response)

    class MockProvider:
        model = ""

        def __init__(self) -> None:
            self.chat = chat_mock

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "req-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert records
    assert records[-1]["ok"] is True
    assert records[-1]["model"] == "response-model"


def test_chat_metrics_records_request_model_on_failure_when_provider_model_missing(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_file = route_test_config / "providers.dummy.toml"
    providers_file.write_text(
        """
[dummy]
type = "dummy"
model = ""
base_url = ""
rpm = 60
concurrency = 1
""".strip()
    )

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    chat_mock = AsyncMock(side_effect=RuntimeError("boom"))

    class MockProvider:
        model = ""

        def __init__(self) -> None:
            self.chat = chat_mock

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())
    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "req-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 502
    assert records
    assert records[-1]["ok"] is False
    assert records[-1]["model"] == "req-model"


def test_chat_metrics_records_model_precedence(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_file = route_test_config / "providers.dummy.toml"
    providers_file.write_text(
        """
[dummy]
type = "dummy"
model = ""
base_url = ""
rpm = 60
concurrency = 1
""".strip()
    )

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    class SuccessfulProvider:
        model = ""

        def __init__(self) -> None:
            self.chat = AsyncMock(
                return_value=ProviderChatResponse(
                    status_code=200,
                    model="response-model",
                    content="dummy:hi",
                    usage_prompt_tokens=0,
                    usage_completion_tokens=0,
                )
            )

    monkeypatch.setitem(
        server_module.providers.providers,
        "dummy",
        SuccessfulProvider(),
    )

    client = TestClient(app)
    success_response = client.post(
        "/v1/chat/completions",
        json={
            "model": "req-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert success_response.status_code == 200
    assert records
    success_record = records[-1]
    assert success_record["ok"] is True
    assert success_record["model"] == "response-model"

    class FailingProvider:
        def __init__(self, model: str) -> None:
            self.model = model
            self.chat = AsyncMock(side_effect=RuntimeError("boom"))

    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)

    monkeypatch.setitem(
        server_module.providers.providers,
        "dummy",
        FailingProvider(model=""),
    )

    failure_response = client.post(
        "/v1/chat/completions",
        json={
            "model": "req-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert failure_response.status_code == 502
    assert len(records) >= 2
    failure_record = records[-1]
    assert failure_record["ok"] is False
    assert failure_record["model"] == "req-model"

    monkeypatch.setitem(
        server_module.providers.providers,
        "dummy",
        FailingProvider(model="prov-model"),
    )

    fallback_failure_response = client.post(
        "/v1/chat/completions",
        json={
            "model": "",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert fallback_failure_response.status_code == 502
    assert len(records) >= 3
    fallback_record = records[-1]
    assert fallback_record["ok"] is False
    assert fallback_record["model"] == "prov-model"


def test_chat_missing_header_routes_to_task_header_default(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(load_app("1"))
    from src.orch.router import RouteDef as RouterRouteDef, RoutePlanner as RouterRoutePlanner

    recorded_tasks: list[str] = []
    original_plan: Callable[[RouterRoutePlanner, str], RouterRouteDef] = RouterRoutePlanner.plan

    def recording_plan(self: RouterRoutePlanner, task: str) -> RouterRouteDef:
        recorded_tasks.append(task)
        return original_plan(self, task)

    monkeypatch.setattr(RouterRoutePlanner, "plan", recording_plan)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert recorded_tasks == ["PLAN"]


def test_chat_custom_task_header_routes_only_with_configured_header(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router_file = route_test_config / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.2
  max_tokens: 64
  task_header: "x-orch-custom-task"
  task_header_value: "PLAN"
routes:
  PLAN:
    primary: dummy
  CUSTOM:
    primary: dummy
""".strip()
    )

    client = TestClient(load_app("1"))
    from src.orch.router import RouteDef as RouterRouteDef, RoutePlanner as RouterRoutePlanner

    recorded_tasks: list[str] = []
    original_plan: Callable[[RouterRoutePlanner, str], RouterRouteDef] = RouterRoutePlanner.plan

    def recording_plan(self: RouterRoutePlanner, task: str) -> RouterRouteDef:
        recorded_tasks.append(task)
        return original_plan(self, task)

    monkeypatch.setattr(RouterRoutePlanner, "plan", recording_plan)

    response_default = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response_default.status_code == 200
    assert recorded_tasks == ["PLAN"]

    recorded_tasks.clear()

    response_custom = client.post(
        "/v1/chat/completions",
        headers={"x-orch-custom-task": "CUSTOM"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response_custom.status_code == 200
    assert recorded_tasks == ["CUSTOM"]


def test_chat_custom_task_header_rejects_default_header_name(
    route_test_config: Path,
) -> None:
    router_file = route_test_config / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.2
  max_tokens: 64
  task_header: "x-orch-alt-task"
  task_header_value: ""
routes:
  CUSTOM:
    primary: dummy
""".strip()
    )

    client = TestClient(load_app("1"))

    success = client.post(
        "/v1/chat/completions",
        headers={"x-orch-alt-task": "CUSTOM"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert success.status_code == 200

    failure = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "CUSTOM"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert failure.status_code == 400
    assert failure.json()["detail"] == (
        "no route configured for task 'DEFAULT' and no DEFAULT route defined in router configuration."
    )


def test_chat_metrics_success_includes_req_id(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert_single_req_id(records)


def test_chat_metrics_routing_error_includes_req_id(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "UNKNOWN"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert_single_req_id(records)


def test_chat_metrics_routing_error_usage_zero(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "UNKNOWN"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert records
    record = records[-1]
    assert record["usage_prompt"] == 0
    assert record["usage_completion"] == 0


def test_chat_metrics_provider_error_includes_req_id(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)
    async def no_sleep(_: float) -> None:
        return None

    class BoomProvider:
        model = "dummy"

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)
    monkeypatch.setitem(server_module.providers.providers, "dummy", BoomProvider())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 502
    assert_single_req_id(records)


def test_chat_metrics_provider_error_usage_zero(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    async def no_sleep(_: float) -> None:
        return None

    class BoomProvider:
        model = "dummy"

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)
    monkeypatch.setitem(server_module.providers.providers, "dummy", BoomProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 502
    assert records
    for record in records:
        assert record["usage_prompt"] == 0
        assert record["usage_completion"] == 0


def test_chat_metrics_transient_provider_error_usage_zero(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    async def no_sleep(_: float) -> None:
        return None

    from src.orch.types import ProviderChatResponse

    success_response = ProviderChatResponse(
        status_code=200,
        model="dummy",
        content="dummy:hi",
        usage_prompt_tokens=5,
        usage_completion_tokens=7,
    )

    chat_mock = AsyncMock(side_effect=[RuntimeError("boom"), success_response])

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = chat_mock

    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)
    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    failure_records = [record for record in records if record.get("ok") is False]
    assert failure_records
    failure_record = failure_records[0]
    assert failure_record["usage_prompt"] == 0
    assert failure_record["usage_completion"] == 0

    success_records = [record for record in records if record.get("ok") is True]
    assert success_records
    success_record = success_records[0]
    assert success_record["usage_prompt"] == 5
    assert success_record["usage_completion"] == 7


def test_chat_retries_success_after_transient_failures(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    async def no_sleep(_: float) -> None:
        return None

    class FlakyProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(
            self,
            model: str,
            messages: list[dict[str, str]],
            *,
            temperature: float = 0.2,
            max_tokens: int = 2048,
        ) -> "ProviderChatResponse":
            from src.orch.types import ProviderChatResponse

            self.calls += 1
            if self.calls < 3:
                raise RuntimeError(f"fail-{self.calls}")
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            return ProviderChatResponse(
                status_code=200,
                model=model,
                content=f"dummy:{last_user}",
            )

    provider = FlakyProvider()
    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)
    monkeypatch.setitem(server_module.providers.providers, "dummy", provider)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert provider.calls == 3
    success_records = [record for record in records if record.get("ok") is True]
    assert len(success_records) == 1
    assert success_records[0]["retries"] == 2
    failure_records = [record for record in records if record.get("ok") is False]
    assert [record["retries"] for record in failure_records] == [0, 1]


def test_chat_retries_uses_three_attempts_with_mock(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    async def no_sleep(_: float) -> None:
        return None

    from src.orch.types import ProviderChatResponse

    success_response = ProviderChatResponse(
        status_code=200,
        model="dummy",
        content="dummy:hi",
    )

    chat_mock = AsyncMock(
        side_effect=[RuntimeError("fail-1"), RuntimeError("fail-2"), success_response]
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = chat_mock

    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)
    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert chat_mock.await_count == 3
    success_records = [record for record in records if record.get("ok") is True]
    assert len(success_records) == 1
    assert success_records[0]["retries"] == 2
