from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import httpx
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


def ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


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
tpm = 6000

[dummy_alt]
type = "dummy"
model = "dummy"
base_url = ""
rpm = 60
concurrency = 1
tpm = 6000
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
    targets:
      - provider: dummy
        circuit_breaker:
          failure_threshold: 2
          recovery_time_s: 60
      - provider: dummy_alt
""".strip()
    )
    return tmp_path


def _write_single_provider_router(route_test_config: Path) -> None:
    router_file = route_test_config / "router.yaml"
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


def test_models_endpoint_returns_expected_shape(route_test_config: Path) -> None:
    app = load_app("1")
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200

    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)

    dummy_entry = next((item for item in payload["data"] if item["provider"] == "dummy"), None)
    assert dummy_entry is not None
    assert dummy_entry["id"] == "dummy"
    assert dummy_entry["object"] == "model"
    assert dummy_entry["owned_by"] == "dummy"
    assert dummy_entry["model"] == "dummy"
    assert "dummy_alt" in dummy_entry.get("aliases", [])


def test_route_planner_skips_provider_after_consecutive_failures(
    route_test_config: Path,
) -> None:
    ensure_project_root_on_path()
    from src.orch.router import RoutePlanner, load_config

    loaded = load_config(str(route_test_config), use_dummy=True)
    planner = RoutePlanner(loaded.router, loaded.providers)

    initial = planner.plan("PLAN", now=0.0)
    assert initial.primary == "dummy"

    planner.record_failure("dummy", now=1.0)
    planner.record_failure("dummy", now=2.0)

    rerouted = planner.plan("PLAN", now=3.0)
    assert rerouted.primary == "dummy_alt"
    assert rerouted.fallback and rerouted.fallback[0] == "dummy"


def test_route_planner_circuit_resets_after_recovery(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import import_module

    ensure_project_root_on_path()
    from src.orch.router import RoutePlanner, load_config

    loaded = load_config(str(route_test_config), use_dummy=True)
    planner = RoutePlanner(loaded.router, loaded.providers)

    router_module = import_module("src.orch.router")

    current_time = [100.0]

    def fake_monotonic() -> float:
        return current_time[0]

    monkeypatch.setattr(router_module.time, "monotonic", fake_monotonic)

    initial = planner.plan("PLAN")
    assert initial.primary == "dummy"

    planner.record_failure("dummy")
    current_time[0] += 1.0
    planner.record_failure("dummy")
    failure_time = current_time[0]

    current_time[0] = failure_time + 1.0
    blocked = planner.plan("PLAN")
    assert blocked.primary == "dummy_alt"

    current_time[0] = failure_time + 61.0
    recovered = planner.plan("PLAN")
    assert recovered.primary == "dummy"

    state = planner._circuit_states["dummy"]
    assert state.opened_until is None
    assert state.failure_count == 0


def capture_metric_records(
    server_module: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    async def capture(record: dict[str, object]) -> None:
        records.append(record)

    monkeypatch.setattr(server_module.metrics, "write", capture)
    return records


def assert_orch_headers(
    response: httpx.Response,
    *,
    provider: str,
    fallback_attempts: str,
) -> None:
    assert response.headers["x-orch-request-id"]
    assert response.headers["x-orch-provider"] == provider
    assert response.headers["x-orch-fallback-attempts"] == fallback_attempts


def test_chat_respects_sticky_headers(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    providers_file = route_test_config / "providers.dummy.toml"
    providers_file.write_text(
        """
[dummy_a]
type = "dummy"
model = "dummy"
base_url = ""
rpm = 60
concurrency = 1
tpm = 6000

[dummy_b]
type = "dummy"
model = "dummy"
base_url = ""
rpm = 60
concurrency = 1
tpm = 6000

[dummy_c]
type = "dummy"
model = "dummy"
base_url = ""
rpm = 60
concurrency = 1
tpm = 6000
""".strip()
    )
    router_file = route_test_config / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.2
  max_tokens: 64
  task_header: "x-orch-task-kind"
  task_header_value: "PLAN"
routes:
  PLAN:
    strategy: sticky
    sticky_ttl: 5
    targets:
      - provider: dummy_a
      - provider: dummy_b
      - provider: dummy_c
""".strip()
    )

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.types import ProviderChatResponse

    class FakePlanner:
        def __init__(self, providers: list[str], *, ttl: float) -> None:
            self.providers = providers
            self.ttl = ttl
            self.now = 0.0
            self._assignments: dict[str, tuple[str, float]] = {}
            self._rotation = 0
            self.plan_calls: list[tuple[str, str | None, float]] = []

        def _next_provider(self) -> str:
            provider = self.providers[self._rotation % len(self.providers)]
            self._rotation += 1
            return provider

        def plan(self, task: str, *, sticky_key: str | None = None) -> SimpleNamespace:
            self.plan_calls.append((task, sticky_key, self.now))
            if sticky_key:
                assignment = self._assignments.get(sticky_key)
                if assignment is not None and assignment[1] > self.now:
                    provider = assignment[0]
                else:
                    provider = self._next_provider()
                    self._assignments[sticky_key] = (provider, self.now + self.ttl)
            else:
                provider = self._next_provider()
            fallback = [name for name in self.providers if name != provider]
            return SimpleNamespace(primary=provider, fallback=fallback)

        def record_success(self, provider: str) -> None:
            pass

        def record_failure(self, provider: str, *, now: float | None = None) -> None:
            pass

    fake_planner = FakePlanner(["dummy_a", "dummy_b", "dummy_c"], ttl=5.0)
    monkeypatch.setattr(server_module, "planner", fake_planner, raising=False)

    @asynccontextmanager
    async def guard_ctx() -> Any:
        yield object()

    class FakeGuard:
        def acquire(self, **_: object):
            return guard_ctx()

        def record_usage(
            self,
            lease: object,
            *,
            usage_prompt_tokens: int,
            usage_completion_tokens: int,
        ) -> float:
            return 0.0

    class FakeGuards:
        def get(self, name: str) -> FakeGuard:
            return FakeGuard()

    monkeypatch.setattr(server_module, "guards", FakeGuards(), raising=False)

    server_module.providers.providers.clear()

    def make_provider(name: str) -> SimpleNamespace:
        async def chat(*_: object, **__: object) -> ProviderChatResponse:
            return ProviderChatResponse(status_code=200, model=name, content=name)

        return SimpleNamespace(model=name, chat=chat)

    for provider_name in ["dummy_a", "dummy_b", "dummy_c"]:
        monkeypatch.setitem(
            server_module.providers.providers,
            provider_name,
            make_provider(provider_name),
        )

    client = TestClient(app)
    request_body = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "hi"}],
    }

    def post_with(headers: dict[str, str], *, now: float) -> httpx.Response:
        fake_planner.now = now
        return client.post("/v1/chat/completions", headers=headers, json=request_body)

    first = post_with({"x-orch-sticky-key": "user-1"}, now=0.0)
    repeat = post_with({"x-orch-sticky-key": "user-1"}, now=1.0)
    other_key = post_with({"x-orch-sticky-key": "user-2"}, now=1.5)
    expired = post_with({"X-Orch-Session": "user-1"}, now=10.0)

    for response in (first, repeat, other_key, expired):
        assert response.status_code == 200

    assert first.headers["x-orch-provider"] == "dummy_a"
    assert repeat.headers["x-orch-provider"] == "dummy_a"
    assert other_key.headers["x-orch-provider"] == "dummy_b"
    assert expired.headers["x-orch-provider"] == "dummy_c"

    assert [call[1] for call in fake_planner.plan_calls] == [
        "user-1",
        "user-1",
        "user-2",
        "user-1",
    ]
    assert fake_planner.plan_calls[-1][2] == 10.0


def test_chat_uses_guard_estimates_and_records_usage(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.types import ProviderChatResponse

    class FakeGuard:
        def __init__(self) -> None:
            self.acquire_calls: list[int] = []
            self.record_usage_calls: list[tuple[object, int, int]] = []
            self._lease = object()

        def acquire(self, *, estimated_prompt_tokens: int):
            self.acquire_calls.append(estimated_prompt_tokens)
            lease = self._lease

            class FakeGuardContext:
                async def __aenter__(self_nonlocal) -> object:
                    return lease

                async def __aexit__(self_nonlocal, exc_type, exc, tb) -> None:
                    return None

            return FakeGuardContext()

        def record_usage(
            self,
            lease: object,
            *,
            usage_prompt_tokens: int,
            usage_completion_tokens: int,
        ) -> float:
            self.record_usage_calls.append((lease, usage_prompt_tokens, usage_completion_tokens))
            return 0.0

    class FakePlanner:
        def __init__(self) -> None:
            self.plan_calls: list[str] = []
            self.record_success_calls: list[str] = []
            self.record_failure_calls: list[str] = []

        def plan(self, task: str, *, sticky_key: str | None = None):
            self.plan_calls.append(task)

            class _Route:
                primary = "dummy"
                fallback: list[str] = []

            return _Route()

        def record_success(self, provider: str) -> None:
            self.record_success_calls.append(provider)

        def record_failure(self, provider: str, *, now: float | None = None) -> None:
            self.record_failure_calls.append(provider)

    fake_guard = FakeGuard()
    fake_planner = FakePlanner()
    monkeypatch.setattr(server_module, "guards", type("_Guards", (), {"get": lambda self, name: fake_guard})())
    monkeypatch.setattr(server_module, "planner", fake_planner, raising=False)

    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
            usage_prompt_tokens=42,
            usage_completion_tokens=11,
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain."},
    ]
    body = {
        "model": "dummy",
        "messages": messages,
        "max_tokens": 128,
    }
    expected_estimate = server_module._estimate_prompt_tokens(messages, body["max_tokens"])

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert_orch_headers(response, provider="dummy", fallback_attempts="0")
    assert fake_guard.acquire_calls == [expected_estimate]
    assert fake_guard.record_usage_calls
    lease, usage_prompt, usage_completion = fake_guard.record_usage_calls[0]
    assert usage_prompt == 42
    assert usage_completion == 11
    assert fake_planner.record_success_calls == ["dummy"]
    assert fake_planner.record_failure_calls == []


def test_stream_usage_events_record_guard_and_metrics(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    metric_records = capture_metric_records(server_module, monkeypatch)
    from src.orch.types import ProviderStreamChunk
    class FakeGuard:
        def __init__(self) -> None:
            self.lease = object()
            self.record_usage_calls: list[tuple[object, int, int]] = []

        async def __aenter__(self) -> object:  # pragma: no cover - trivial context manager
            return self.lease

        async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial context manager
            return None

        def record_usage(
            self,
            lease: object,
            *,
            usage_prompt_tokens: int,
            usage_completion_tokens: int,
        ) -> float:
            self.record_usage_calls.append((lease, usage_prompt_tokens, usage_completion_tokens))
            return 0.0
    route = SimpleNamespace(primary="dummy", fallback=[])
    planner_stub = SimpleNamespace(plan=lambda task: route, record_success=lambda provider: None, record_failure=lambda provider, *, now=None: None)
    async def _stream(*_args: object, **_kwargs: object):
        yield ProviderStreamChunk(event_type="delta", delta={"content": "hi"})
        yield ProviderStreamChunk(event_type="usage", usage={"prompt_tokens": 9, "completion_tokens": 5})
        yield ProviderStreamChunk(event_type="message_stop", finish_reason="stop")
    fake_guard = FakeGuard()
    provider_stub = SimpleNamespace(model="dummy", chat_stream=staticmethod(_stream))
    monkeypatch.setattr(server_module, "guards", SimpleNamespace(get=lambda name: fake_guard), raising=False)
    monkeypatch.setattr(server_module, "planner", planner_stub, raising=False)
    monkeypatch.setattr(server_module, "providers", SimpleNamespace(get=lambda name: provider_stub), raising=False)
    client = TestClient(app)
    body = {"model": "dummy", "messages": [{"role": "user", "content": "stream usage"}], "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        for _ in response.iter_text():
            pass
    assert fake_guard.record_usage_calls
    assert fake_guard.record_usage_calls[0] == (fake_guard.lease, 9, 5)
    assert metric_records
    record = metric_records[0]; assert record["usage_prompt"] == 9 and record["usage_completion"] == 5


@pytest.mark.parametrize(
    "exception_factory, expected_status",
    [
        (
            lambda: httpx.HTTPStatusError(
                "429",
                request=httpx.Request("POST", "https://dummy.test"),
                response=httpx.Response(429, request=httpx.Request("POST", "https://dummy.test")),
            ),
            429,
        ),
        (lambda: RuntimeError("boom"), 502),
    ],
)
def test_chat_records_planner_failure_on_guarded_errors(
    route_test_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: Callable[[], Exception],
    expected_status: int,
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    class FakeGuard:
        def __init__(self) -> None:
            self.acquire_calls: list[int] = []
            self.record_usage_calls: list[tuple[object, int, int]] = []

        def acquire(self, *, estimated_prompt_tokens: int):
            self.acquire_calls.append(estimated_prompt_tokens)
            lease = object()

            class _Context:
                async def __aenter__(self_inner) -> object:
                    return lease

                async def __aexit__(self_inner, exc_type, exc, tb) -> None:
                    return None

            return _Context()

        def record_usage(
            self,
            lease: object,
            *,
            usage_prompt_tokens: int,
            usage_completion_tokens: int,
        ) -> float:
            self.record_usage_calls.append((lease, usage_prompt_tokens, usage_completion_tokens))
            return 0.0

    class FakePlanner:
        def __init__(self) -> None:
            self.record_success_calls: list[str] = []
            self.record_failure_calls: list[str] = []

        def plan(self, task: str, *, sticky_key: str | None = None):

            class _Route:
                primary = "dummy"
                fallback: list[str] = []

            return _Route()

        def record_success(self, provider: str) -> None:
            self.record_success_calls.append(provider)

        def record_failure(self, provider: str, *, now: float | None = None) -> None:
            self.record_failure_calls.append(provider)

    fake_guard = FakeGuard()
    fake_planner = FakePlanner()
    monkeypatch.setattr(server_module, "guards", type("_Guards", (), {"get": lambda self, name: fake_guard})())
    monkeypatch.setattr(server_module, "planner", fake_planner, raising=False)

    class MockProvider:
        model = "dummy"

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise exception_factory()

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    messages = [{"role": "user", "content": "Hello"}]
    body = {"model": "dummy", "messages": messages, "max_tokens": 32}
    expected_estimate = server_module._estimate_prompt_tokens(messages, body["max_tokens"])

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == expected_status
    assert fake_guard.acquire_calls
    assert all(call == expected_estimate for call in fake_guard.acquire_calls)
    if expected_status == 429:
        assert fake_guard.acquire_calls == [expected_estimate]
    assert fake_guard.record_usage_calls == []
    assert fake_planner.record_success_calls == []
    assert fake_planner.record_failure_calls
    assert all(name == "dummy" for name in fake_planner.record_failure_calls)
    if expected_status == 429:
        assert fake_planner.record_failure_calls == ["dummy"]


@pytest.mark.parametrize(
    "exception_factory, expected_status",
    [
        (
            lambda: httpx.HTTPStatusError(
                "429",
                request=httpx.Request("POST", "https://dummy.test"),
                response=httpx.Response(429, request=httpx.Request("POST", "https://dummy.test")),
            ),
            429,
        ),
        (lambda: RuntimeError("boom"), 502),
    ],
)
def test_chat_releases_guard_usage_on_provider_exception(
    route_test_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: Callable[[], Exception],
    expected_status: int,
) -> None:
    _write_single_provider_router(route_test_config)
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.rate_limiter import Guard, GuardLease

    class TrackingGuard(Guard):
        def __init__(self) -> None:
            super().__init__(rpm=60, concurrency=1, tpm=128)
            self.record_usage_calls: list[tuple[int, int]] = []

        def record_usage(
            self,
            lease: GuardLease | None,
            *,
            usage_prompt_tokens: int,
            usage_completion_tokens: int,
        ) -> float:
            self.record_usage_calls.append((usage_prompt_tokens, usage_completion_tokens))
            return super().record_usage(
                lease, usage_prompt_tokens=usage_prompt_tokens, usage_completion_tokens=usage_completion_tokens
            )

    tracking_guard = TrackingGuard()

    class MockProviderGuards:
        def get(self, name: str) -> Guard:
            assert name == "dummy"
            return tracking_guard

    monkeypatch.setattr(server_module, "guards", MockProviderGuards(), raising=False)
    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)

    class MockProvider:
        model = "dummy"

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise exception_factory()

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    messages = [{"role": "user", "content": "Hello"}]
    response = client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "messages": messages, "max_tokens": 32},
    )

    assert response.status_code == expected_status
    assert tracking_guard.record_usage_calls
    for usage_prompt, usage_completion in tracking_guard.record_usage_calls:
        assert usage_prompt == 0
        assert usage_completion == 0
    bucket = tracking_guard._tpm_bucket
    assert bucket is not None
    assert bucket._total == 0


def test_chat_failure_response_includes_orch_headers_when_guard_blocks_fallback(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    @asynccontextmanager
    async def lease_ctx() -> Any:
        yield object()

    guard = SimpleNamespace(acquire=lambda **_kwargs: lease_ctx(), record_usage=lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(server_module, "guards", SimpleNamespace(get=lambda *_args, **_kwargs: guard), raising=False)
    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)

    route = SimpleNamespace(primary="primary", fallback=["fallback"])
    monkeypatch.setattr(server_module.planner, "plan", lambda _task, *, sticky_key=None: route)

    primary_failure = AsyncMock(side_effect=RuntimeError("boom"))
    fallback_failure = AsyncMock(side_effect=RuntimeError("boom"))
    primary_provider = SimpleNamespace(model="primary-model", chat=primary_failure)
    fallback_provider = SimpleNamespace(model="fallback-model", chat=fallback_failure)
    monkeypatch.setitem(server_module.providers.providers, "primary", primary_provider)
    monkeypatch.setitem(server_module.providers.providers, "fallback", fallback_provider)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "req-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    assert_orch_headers(response, provider="primary", fallback_attempts="0")
    assert fallback_failure.call_count == 0


def test_chat_failure_response_includes_headers_for_unroutable_task(
    route_test_config: Path,
) -> None:
    client = TestClient(load_app("1"))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-orch-task-kind": "UNKNOWN"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    error_body = response.json().get("error")
    assert isinstance(error_body, dict)
    assert response.headers["x-orch-request-id"]
    assert response.headers["x-orch-provider"] == "unknown"
    assert response.headers["x-orch-fallback-attempts"] == "0"


def assert_single_req_id(records: list[dict[str, object]]) -> None:
    assert records
    req_ids = {record.get("req_id") for record in records}
    assert len(req_ids) == 1
    req_id = req_ids.pop()
    assert isinstance(req_id, str)
    assert req_id


def test_chat_streams_events(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    async def stream_chat(*_: object, **__: object):
        yield {"event": "chunk", "data": {"choices": [{"delta": {"content": "hi"}}]}}
        yield {"event": "done", "data": {}}

    class MockProvider:
        model = "dummy"

        async def chat_stream(self, *args: object, **kwargs: object):
            async for item in stream_chat(*args, **kwargs):
                yield item

    monkeypatch.setitem(
        server_module.providers.providers, "dummy", MockProvider()
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert_orch_headers(response, provider="dummy", fallback_attempts="0")
    body = response.text
    assert "\"delta\"" in body
    assert "data: [DONE]" in body


def test_chat_stream_guard_uses_prompt_estimate_and_cancels_reservation(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_single_provider_router(route_test_config)
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.rate_limiter import Guard

    class TrackingGuard(Guard):
        def __init__(self) -> None:
            super().__init__(rpm=60, concurrency=1, tpm=128)
            self.acquire_calls: list[int] = []
            self.record_usage_calls: list[tuple[int, int]] = []

        def acquire(self, *, estimated_prompt_tokens: int):
            self.acquire_calls.append(estimated_prompt_tokens)
            return super().acquire(estimated_prompt_tokens=estimated_prompt_tokens)

        def record_usage(
            self,
            lease: object,
            *,
            usage_prompt_tokens: int,
            usage_completion_tokens: int,
        ) -> float:
            self.record_usage_calls.append(
                (usage_prompt_tokens, usage_completion_tokens)
            )
            return super().record_usage(
                lease,
                usage_prompt_tokens=usage_prompt_tokens,
                usage_completion_tokens=usage_completion_tokens,
            )

    tracking_guard = TrackingGuard()

    bucket = tracking_guard._tpm_bucket
    assert bucket is not None

    reserve_calls: list[int] = []
    cancel_calls: list[int | None] = []

    original_reserve = bucket.reserve

    def reserve_wrapper(self: object, tokens: int, now: float):
        reserve_calls.append(tokens)
        return original_reserve(tokens, now)

    monkeypatch.setattr(
        bucket, "reserve", MethodType(reserve_wrapper, bucket)
    )

    original_cancel = bucket.cancel

    def cancel_wrapper(self: object, reservation_id: int | None, now: float):
        cancel_calls.append(reservation_id)
        return original_cancel(reservation_id, now)

    monkeypatch.setattr(
        bucket, "cancel", MethodType(cancel_wrapper, bucket)
    )

    class MockProviderGuards:
        def get(self, name: str) -> Guard:
            assert name == "dummy"
            return tracking_guard

    monkeypatch.setattr(server_module, "guards", MockProviderGuards(), raising=False)

    async def stream_chat(*_: object, **__: object):
        yield {"event": "chunk", "data": {"choices": [{"delta": {"content": "hi"}}]}}
        yield {"event": "done", "data": {}}

    class MockProvider:
        model = "dummy"

        async def chat_stream(self, *args: object, **kwargs: object):
            async for item in stream_chat(*args, **kwargs):
                yield item

    monkeypatch.setitem(
        server_module.providers.providers,
        "dummy",
        MockProvider(),
    )

    client = TestClient(app)
    messages = [{"role": "user", "content": "hi"}]
    body = {
        "model": "dummy",
        "messages": messages,
        "max_tokens": 64,
        "stream": True,
    }
    expected_estimate = server_module._estimate_prompt_tokens(
        messages, body["max_tokens"]
    )

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert tracking_guard.acquire_calls == [expected_estimate]
    assert reserve_calls == [expected_estimate]
    assert cancel_calls
    assert cancel_calls[0] is not None
    assert tracking_guard.record_usage_calls == []
    assert bucket._total == 0


def test_chat_streams_provider_chunk_events(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.types import ProviderStreamChoice, ProviderStreamChunk

    async def stream_chat(*_: object, **__: object):
        yield ProviderStreamChunk(
            choices=[
                ProviderStreamChoice(
                    delta={"content": "hello"},
                )
            ]
        )

    class MockProvider:
        model = "dummy"

        async def chat_stream(self, *args: object, **kwargs: object):
            async for item in stream_chat(*args, **kwargs):
                yield item

    monkeypatch.setitem(
        server_module.providers.providers, "dummy", MockProvider()
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "data: {\"choices\":" in body
    assert "data: [DONE]" in body

def _make_http_status_error(status: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://dummy.test")
    headers: dict[str, str] = {} if retry_after is None else {"Retry-After": retry_after}
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_code"),
    [
        ("auth", 401, "invalid_api_key"),
        ("rate_limit", 429, "rate_limit"),
        ("server_error", 502, "provider_server_error"),
    ],
)
def test_chat_error_code_is_enumerated(
    route_test_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_status: int,
    expected_code: str,
) -> None:
    _write_single_provider_router(route_test_config)
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    client = TestClient(app)

    if scenario == "auth":
        monkeypatch.setattr(server_module, "INBOUND_API_KEYS", frozenset({"secret"}))
    else:
        status_map = {"rate_limit": 429, "server_error": 503}

        class ErroringProvider:
            model = "dummy"

            async def chat(self, *args: object, **kwargs: object) -> object:
                raise _make_http_status_error(status_map[scenario])

        monkeypatch.setitem(
            server_module.providers.providers, "dummy", ErroringProvider()
        )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == expected_status
    payload = response.json()
    error_body = payload.get("error")
    assert isinstance(error_body, dict)
    error_code = error_body.get("code")
    assert isinstance(error_code, str)
    enum_values = {member.value for member in server_module.ErrorCode}
    assert error_code in enum_values
    assert error_code == expected_code


@pytest.mark.parametrize(
    "retry_after_seconds, use_http_date",
    [(37, False), (90, True)],
)
def test_chat_stream_rate_limit_retry_after(
    route_test_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_after_seconds: int,
    use_http_date: bool,
) -> None:
    _write_single_provider_router(route_test_config)
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    header_value = str(retry_after_seconds)
    if use_http_date:
        freeze_now = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        header_value = format_datetime(freeze_now + timedelta(seconds=retry_after_seconds))
        monkeypatch.setattr(server_module, "datetime", SimpleNamespace(now=lambda *_: freeze_now))

    class MockProvider:
        model = "dummy"
        async def chat_stream(self, *args: object, **kwargs: object):
            raise _make_http_status_error(429, header_value)
    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 429
    assert_orch_headers(response, provider="dummy", fallback_attempts="0")
    payload = response.json()
    assert payload["error"]["retry_after"] == retry_after_seconds
    assert payload["error"]["type"] == "rate_limit"


def test_chat_stream_fallback_chain_returns_final_failure(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    guard = server_module.guards.get("dummy")
    monkeypatch.setattr(server_module.guards, "get", lambda *_args, **_kwargs: guard)
    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)
    route = SimpleNamespace(primary="primary", fallback=["secondary"])
    monkeypatch.setattr(server_module.planner, "plan", lambda *_: route)
    failure_calls: list[str] = []
    monkeypatch.setattr(
        server_module.planner,
        "record_failure",
        lambda provider: failure_calls.append(provider),
    )

    def _stream_error(exc: Exception):
        async def _gen() -> Any:
            raise exc
            if False:
                yield None
        return _gen()
    providers_map = {
        name: type(
            f"{name.title()}Provider",
            (),
            {
                "model": f"{name}-model",
                "chat_stream": lambda self, *args, _exc=exc, **kwargs: _stream_error(_exc),
            },
        )()
        for name, exc in (
            ("primary", _make_http_status_error(503)),
            ("secondary", RuntimeError("fallback failed")),
        )
    }
    for name, provider in providers_map.items():
        monkeypatch.setitem(server_module.providers.providers, name, provider)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "req-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 502
    assert_orch_headers(response, provider="secondary", fallback_attempts="1")
    payload = response.json()
    assert payload["error"]["message"] == "fallback failed"
    assert payload["error"]["type"] == "provider_server_error"
    assert payload["error"]["retry_after"] == server_module.DEFAULT_RETRY_AFTER_SECONDS
    assert failure_calls == ["primary", "secondary"]


def test_chat_accepts_tool_role_messages(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    tool_content = [{"type": "output_text", "text": "done"}]
    tool_message = {
        "role": "tool",
        "name": "browser",
        "tool_call_id": "call-1",
        "content": tool_content,
    }
    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {"role": "user", "content": "hi"},
                tool_message,
            ],
        },
    )

    assert response.status_code == 200
    provider_chat.assert_awaited_once()
    called_args = provider_chat.await_args.args
    assert called_args[0] == "dummy"
    assert called_args[1] == [
        {"role": "user", "content": "hi"},
        tool_message,
    ]
    assert records
    assert all(record.get("status") != 422 for record in records)
    assert records[-1]["status"] == 200


def test_chat_preserves_message_extra_fields(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    extra_message = {
        "role": "assistant",
        "content": "function call",
        "name": "planner",
        "tool_call_id": "call-123",
    }
    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {"role": "user", "content": "hi"},
                extra_message,
            ],
        },
    )

    assert response.status_code == 200
    provider_chat.assert_awaited_once()
    called_args = provider_chat.await_args.args
    assert called_args[0] == "dummy"
    assert called_args[1] == [
        {"role": "user", "content": "hi"},
        extra_message,
    ]
    assert records
    assert all(record.get("status") != 422 for record in records)
    assert records[-1]["status"] == 200


def test_chat_passes_typed_options_once(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.25,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.1,
            "logit_bias": {"42": 1.0},
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    provider_chat.assert_awaited_once()
    assert provider_chat.await_args.args == ("dummy", [{"role": "user", "content": "hi"}])
    called_kwargs = provider_chat.await_args.kwargs
    assert called_kwargs["top_p"] == 0.25
    assert called_kwargs["frequency_penalty"] == 0.5
    assert called_kwargs["presence_penalty"] == -0.1
    assert called_kwargs["logit_bias"] == {"42": 1.0}
    assert called_kwargs["response_format"] == {"type": "json_object"}
    assert records
    assert records[-1]["status"] == 200

def test_chat_returns_retry_after_on_429(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    async def failing_chat(*_: object, **__: object) -> None:
        response = httpx.Response(429, headers={"Retry-After": "3"}, request=httpx.Request("POST", "http://x"))
        raise httpx.HTTPStatusError("rate limit", request=response.request, response=response)

    class MockProvider:
        model = "dummy"

        async def chat(self, *args: object, **kwargs: object) -> None:
            await failing_chat(*args, **kwargs)

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 429
    payload = response.json()
    assert payload["error"]["retry_after"] == 3


def test_metrics_endpoint_returns_prometheus_text(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCH_INBOUND_API_KEYS", "")
    monkeypatch.setenv("ORCH_CORS_ALLOW_ORIGINS", "")
    app = load_app("1")
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP" in response.text


def test_chat_requires_api_key_when_configured(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCH_INBOUND_API_KEYS", "expected")
    monkeypatch.setenv("ORCH_CORS_ALLOW_ORIGINS", "https://example.com")
    app = load_app("1")
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Origin": "https://example.com"},
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 401


def test_logs_warning_when_api_key_not_configured(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ORCH_INBOUND_API_KEYS", "")
    monkeypatch.setenv("ORCH_CORS_ALLOW_ORIGINS", "")
    with caplog.at_level(logging.WARNING):
        app = load_app("1")
        client = TestClient(app)
        response = client.get("/metrics")

    assert response.status_code == 200
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("APIキー保護が無効" in message for message in warnings)

def test_chat_accepts_tool_choice_strings(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    client = TestClient(app)

    def run_case(tool_choice: str) -> None:
        provider_chat = AsyncMock(
            return_value=ProviderChatResponse(
                status_code=200,
                model="dummy",
                content="ok",
            )
        )

        class MockProvider:
            model = "dummy"

            def __init__(self) -> None:
                self.chat = provider_chat

        monkeypatch.setitem(
            server_module.providers.providers, "dummy", MockProvider()
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": tool_choice,
            },
        )

        assert response.status_code == 200
        provider_chat.assert_awaited_once()
        assert provider_chat.await_args.kwargs["tool_choice"] == tool_choice

    run_case("auto")
    run_case("none")

    assert records
    assert records[-1]["status"] == 200


def test_chat_forwards_tools_to_provider(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "lookup"}}
    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "tools": tools,
            "tool_choice": tool_choice,
        },
    )

    assert response.status_code == 200
    provider_chat.assert_awaited_once()
    assert provider_chat.await_args.kwargs["tools"] == tools
    assert provider_chat.await_args.kwargs["tool_choice"] == tool_choice
    assert records
    assert records[-1]["status"] == 200


def test_chat_response_preserves_all_provider_choices(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.types import ProviderChatResponse

    provider_response = ProviderChatResponse(
        status_code=200,
        model="dummy",
        choices=[
            {"index": 5, "message": {"role": "assistant", "content": "first"}},
            {"message": {"role": "assistant", "content": "second"}},
        ],
        usage_prompt_tokens=3,
        usage_completion_tokens=7,
    )
    provider_chat = AsyncMock(return_value=provider_response)

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

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
    body = response.json()
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "first"},
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {"role": "assistant", "content": "second"},
            "finish_reason": "stop",
        },
    ]


def test_chat_forwards_extra_options(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.types import ProviderChatResponse

    extra_options = {
        "top_p": 0.42,
        "frequency_penalty": 1.25,
        "presence_penalty": -0.5,
        "response_format": {"type": "json_object"},
    }

    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="ok",
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    request_body = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "hi"}],
        **extra_options,
    }
    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    provider_chat.assert_awaited_once()
    kwargs = provider_chat.await_args.kwargs
    assert kwargs["top_p"] == extra_options["top_p"]
    assert kwargs["frequency_penalty"] == extra_options["frequency_penalty"]
    assert kwargs["presence_penalty"] == extra_options["presence_penalty"]
    assert kwargs["response_format"] == extra_options["response_format"]

    from src.orch.providers import OpenAICompatProvider
    from src.orch.router import ProviderDef

    openai_calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        openai_calls.append({"url": url, "json": kwargs.get("json", {})})
        request = httpx.Request("POST", url, headers=kwargs.get("headers", {}))
        response_json = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return httpx.Response(status_code=200, json=response_json, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    server_module.providers.providers["dummy"] = OpenAICompatProvider(
        ProviderDef(
            name="dummy",
            type="openai",
            base_url="https://api.openai.com",
            model="gpt-4o",
            auth_env=None,
            rpm=60,
            concurrency=1,
        )
    )

    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    assert openai_calls
    payload = openai_calls[0]["json"]
    assert payload["top_p"] == extra_options["top_p"]
    assert payload["frequency_penalty"] == extra_options["frequency_penalty"]
    assert payload["presence_penalty"] == extra_options["presence_penalty"]
    assert payload["response_format"] == extra_options["response_format"]


def test_chat_forwards_tools_into_provider_http_payload(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    from src.orch.providers import AnthropicProvider, OpenAICompatProvider
    from src.orch.router import ProviderDef

    openai_calls: list[dict[str, Any]] = []
    anthropic_calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        payload = kwargs.get("json", {})
        captured = {"url": url, "json": json.loads(json.dumps(payload))}
        request = httpx.Request("POST", url, headers=kwargs.get("headers", {}))
        if "openai" in url:
            openai_calls.append(captured)
            response_json = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        else:
            anthropic_calls.append(captured)
            response_json = {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        return httpx.Response(status_code=200, json=response_json, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "lookup"}}

    server_module.providers.providers["dummy"] = OpenAICompatProvider(
        ProviderDef(
            name="dummy",
            type="openai",
            base_url="https://api.openai.com",
            model="gpt-4o",
            auth_env=None,
            rpm=60,
            concurrency=1,
        )
    )
    assert isinstance(
        server_module.providers.providers["dummy"], OpenAICompatProvider
    )

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": "dummy",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": json.loads(json.dumps(tools)),
        "tool_choice": json.loads(json.dumps(tool_choice)),
    })

    assert response.status_code == 200
    assert openai_calls
    openai_payload = openai_calls[0]["json"]
    assert openai_calls[0]["url"].endswith("/v1/chat/completions")
    assert "tools" in openai_payload
    assert openai_payload["tools"]
    openai_tool = openai_payload["tools"][0]
    if "function" in openai_tool:
        assert openai_tool["function"]["name"] == "lookup"
    else:
        assert openai_tool["name"] == "lookup"
    openai_choice = openai_payload.get("tool_choice")
    assert openai_choice is not None
    if openai_choice.get("type") == "function":
        assert openai_choice["function"]["name"] == "lookup"
    else:
        assert openai_choice.get("name") == "lookup"

    server_module.providers.providers["dummy"] = AnthropicProvider(
        ProviderDef(
            name="dummy",
            type="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-3-opus-20240229",
            auth_env=None,
            rpm=60,
            concurrency=1,
        )
    )
    assert isinstance(
        server_module.providers.providers["dummy"], AnthropicProvider
    )

    response = client.post("/v1/chat/completions", json={
        "model": "dummy",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": json.loads(json.dumps(tools)),
        "tool_choice": json.loads(json.dumps(tool_choice)),
    })

    assert response.status_code == 200
    assert anthropic_calls
    anthropic_payload = anthropic_calls[0]["json"]
    assert anthropic_payload["tools"] == [
        {
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
    assert anthropic_payload["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert records
    assert records[-1]["status"] == 200


def test_chat_response_propagates_finish_reason_and_tool_calls(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content=None,
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

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
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"] == tool_calls
    assert "content" not in body["choices"][0]["message"]


def test_chat_response_propagates_function_call(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    capture_metric_records(server_module, monkeypatch)

    from src.orch.types import ProviderChatResponse

    function_call = {"name": "lookup", "arguments": "{}"}
    provider_chat = AsyncMock(
        return_value=ProviderChatResponse(
            status_code=200,
            model="dummy",
            content=None,
            function_call=function_call,
        )
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

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
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["function_call"] == function_call
    assert "content" not in message


def test_chat_rejects_stream_requests(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    provider_chat = AsyncMock()

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["message"] == server_module.STREAMING_UNSUPPORTED_ERROR
    provider_chat.assert_not_awaited()
    assert records
    record = records[-1]
    assert record["ok"] is False
    assert record["status"] == 400
    assert record["error"] == server_module.STREAMING_UNSUPPORTED_ERROR


def test_chat_stream_requests_plan_route_and_record_metrics(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    provider_chat = AsyncMock()

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = provider_chat

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    class _Route:
        primary = "dummy"
        fallback: list[str] = []

    planner_plan = Mock(return_value=_Route())
    monkeypatch.setattr(server_module.planner, "plan", planner_plan)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    error_reason = server_module.STREAMING_UNSUPPORTED_ERROR
    assert response.json()["error"]["message"] == error_reason
    provider_chat.assert_not_awaited()
    planner_plan.assert_called_once()
    assert_single_req_id(records)
    assert records[-1]["ok"] is False
    assert records[-1]["status"] == 400
    assert records[-1]["error"] == error_reason
    assert records[-1]["provider"] == "unsupported"
    assert records[-1]["retries"] == 0


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
    assert response.json()["error"]["message"] == (
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
    _write_single_provider_router(route_test_config)

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
    _write_single_provider_router(route_test_config)

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


def test_dummy_config_missing_provider_warns(route_test_config: Path) -> None:
    ensure_project_root_on_path()
    providers_file = route_test_config / "providers.dummy.toml"
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

    from src.orch.router import RoutePlanner, load_config

    with pytest.warns(UserWarning, match="dummy_alt"):
        loaded = load_config(str(route_test_config), use_dummy=True)

    planner = RoutePlanner(loaded.router, loaded.providers)
    selection = planner.plan("PLAN")
    assert selection.primary == "dummy"
    assert "dummy_alt" not in loaded.providers


def test_chat_metrics_records_status_bad_gateway_on_total_failure(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_single_provider_router(route_test_config)
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    class FailingProvider:
        model = ""

        def __init__(self) -> None:
            self.chat = AsyncMock(side_effect=RuntimeError("boom"))

    monkeypatch.setitem(server_module.providers.providers, "dummy", FailingProvider())
    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "req-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert records
    failure_record = records[-1]
    assert failure_record["ok"] is False
    assert failure_record["status"] == server_module.BAD_GATEWAY_STATUS

    assert response.status_code == 502


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
    _write_single_provider_router(route_test_config)

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


def test_chat_logs_error_context(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
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
    _write_single_provider_router(route_test_config)

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    class FailingProvider:
        model = ""

        def __init__(self) -> None:
            self.chat = AsyncMock(side_effect=RuntimeError("boom"))

    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)
    monkeypatch.setitem(
        server_module.providers.providers,
        "dummy",
        FailingProvider(),
    )

    client = TestClient(app)
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 502
    req_id = response.headers["x-orch-request-id"]
    provider = response.headers["x-orch-provider"]
    attempts = int(response.headers["x-orch-fallback-attempts"]) + 1

    error_records = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert error_records
    last_record = error_records[-1]
    message = last_record.getMessage()
    assert f"req_id={req_id}" in message
    assert f"provider={provider}" in message
    assert f"attempts={attempts}" in message


def test_chat_missing_header_routes_to_task_header_default(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(load_app("1"))
    from src.orch.router import RouteDef as RouterRouteDef, RoutePlanner as RouterRoutePlanner

    recorded_tasks: list[str] = []
    original_plan = RouterRoutePlanner.plan

    def recording_plan(
        self: RouterRoutePlanner, task: str, *, sticky_key: str | None = None
    ) -> RouterRouteDef:
        recorded_tasks.append(task)
        return original_plan(self, task, sticky_key=sticky_key)

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
    original_plan = RouterRoutePlanner.plan

    def recording_plan(
        self: RouterRoutePlanner, task: str, *, sticky_key: str | None = None
    ) -> RouterRouteDef:
        recorded_tasks.append(task)
        return original_plan(self, task, sticky_key=sticky_key)

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
    assert failure.json()["error"]["message"] == (
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
    _write_single_provider_router(route_test_config)
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
    _write_single_provider_router(route_test_config)
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


def test_chat_metrics_provider_error_records_status_502(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_single_provider_router(route_test_config)
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
    monkeypatch.setattr(server_module, "MAX_PROVIDER_ATTEMPTS", 1)

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
    failure_record = records[-1]
    assert failure_record["ok"] is False
    assert failure_record["status"] == 502


def test_chat_metrics_retry_success_single_record(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)

    metrics_mock = AsyncMock()
    monkeypatch.setattr(server_module.metrics, "write", metrics_mock)

    from src.orch.types import ProviderChatResponse

    class FlakyProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = AsyncMock(
                side_effect=[
                    RuntimeError("boom"),
                    ProviderChatResponse(
                        status_code=200,
                        model="dummy",
                        content="dummy:hi",
                        usage_prompt_tokens=0,
                        usage_completion_tokens=0,
                    ),
                ]
            )

    monkeypatch.setitem(
        server_module.providers.providers,
        "dummy",
        FlakyProvider(),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert metrics_mock.await_count == 1
    record = metrics_mock.await_args.args[0]
    assert record["ok"] is True
    assert record["retries"] == 1
    assert isinstance(record["req_id"], str)


def test_chat_metrics_does_not_retry_on_http_client_error(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status_code=400, request=request, text="bad request")
    chat_mock = AsyncMock(
        side_effect=httpx.HTTPStatusError("bad request", request=request, response=response)
    )

    class MockProvider:
        model = "dummy"

        def __init__(self) -> None:
            self.chat = chat_mock

    monkeypatch.setitem(server_module.providers.providers, "dummy", MockProvider())

    client = TestClient(app)
    response_obj = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response_obj.status_code == 400
    body = response_obj.json()
    assert body["error"]["message"] == "bad request"
    assert chat_mock.await_count == 1
    assert records
    failure_record = records[-1]
    assert failure_record["retries"] == 0
    assert failure_record["status"] == response_obj.status_code
    assert failure_record["error"] == body["error"]["message"]


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
    assert len(records) == 1
    record = records[0]
    assert record["ok"] is True
    assert record["usage_prompt"] == 5
    assert record["usage_completion"] == 7
    assert record["retries"] == 1

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
            tools: list[dict[str, Any]] | None = None,
            tool_choice: dict[str, Any] | None = None,
            function_call: dict[str, Any] | None = None,
        ) -> "ProviderChatResponse":
            from src.orch.types import ProviderChatResponse

            _ = tools
            _ = tool_choice
            _ = function_call
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
    assert len(records) == 1
    record = records[0]
    assert record["ok"] is True
    assert record["retries"] == 2


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


def test_chat_metrics_retry_success_logs_final_result_once(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(server_module.asyncio, "sleep", no_sleep)

    metrics_mock = AsyncMock()
    monkeypatch.setattr(server_module.metrics, "write", metrics_mock)

    from src.orch.types import ProviderChatResponse

    responses: list[ProviderChatResponse | Exception] = [
        RuntimeError("boom"),
        ProviderChatResponse(
            status_code=200,
            model="dummy",
            content="dummy:hi",
            usage_prompt_tokens=0,
            usage_completion_tokens=0,
        ),
    ]

    class MockProvider:
        model = "dummy"

        async def chat(self, *args: object, **kwargs: object) -> ProviderChatResponse:
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

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
    assert metrics_mock.await_count == 1
    record = metrics_mock.await_args_list[0].args[0]
    assert record["ok"] is True
    assert record["retries"] == 1

def test_anthropic_rejects_image_url_block(
    route_test_config: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_file = route_test_config / "providers.dummy.toml"
    providers_file.write_text(
        """
[claude]
type = "anthropic"
model = "claude-3-opus"
base_url = "https://anthropic.invalid"
rpm = 60
concurrency = 1
""".strip()
    )
    router_file = route_test_config / "router.yaml"
    router_file.write_text(
        """
defaults:
  temperature: 0.2
  max_tokens: 64
  task_header: "x-orch-task-kind"
  task_header_value: "PLAN"
routes:
  PLAN:
    primary: claude
""".strip()
    )

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-3-opus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "error" in payload
    assert "message" in payload["error"]
    assert "image_url" in payload["error"]["message"]
    assert records
    assert_single_req_id(records)
    final_record = records[-1]
    assert final_record["status"] == 400
    assert final_record["provider"] == "claude"
    assert final_record["ok"] is False
    assert final_record["retries"] == 0
