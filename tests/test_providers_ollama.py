import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.providers import OllamaProvider  # noqa: E402
from src.orch.router import ProviderDef  # noqa: E402


class _DummyResponse:
    status_code = 200

    @staticmethod
    def json() -> dict[str, Any]:
        return {"message": {"content": "ok"}, "done": True}

    @staticmethod
    def raise_for_status() -> None:
        return None


def _make_provider() -> OllamaProvider:
    provider_def = ProviderDef(
        name="ollama",
        type="ollama",
        base_url="http://localhost:11434",
        model="llama3",
        auth_env=None,
        rpm=120,
        concurrency=1,
    )
    return OllamaProvider(provider_def)


def test_ollama_sets_format_json_for_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider()
    post_calls: list[dict[str, Any]] = []

    class _DummyAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs

        async def __aenter__(self) -> "_DummyAsyncClient":
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: Any,
        ) -> None:
            return None

        async def post(self, url: str, *, json: Any | None = None, **kwargs: Any) -> _DummyResponse:
            post_calls.append({"url": url, "json": json, "kwargs": kwargs})
            return _DummyResponse()

    monkeypatch.setattr("httpx.AsyncClient", _DummyAsyncClient)

    async def invoke() -> None:
        await provider.chat(
            model="llama3",
            messages=[{"role": "user", "content": "ping"}],
            response_format={"type": "json_object"},
        )

    asyncio.run(invoke())

    assert post_calls
    payload = post_calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload.get("format") == "json"
