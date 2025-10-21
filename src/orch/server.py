import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from typing_extensions import TypedDict

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


from .metrics import MetricsLogger
from .providers import ProviderRegistry, UnsupportedContentBlockError
from .rate_limiter import ProviderGuards
from .router import ProviderDef, RouteDef, RoutePlanner, load_config
from .types import ChatRequest, ProviderChatResponse, chat_response_from_provider

logger = logging.getLogger(__name__)

app = FastAPI(title="llm-orch")

CONFIG_DIR = os.environ.get("ORCH_CONFIG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config"))

TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
FALSY_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})


class _ModelInfo(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str


class _ModelListResponse(TypedDict):
    object: Literal["list"]
    data: list[_ModelInfo]


def _env_var_as_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    return default


USE_DUMMY: bool = _env_var_as_bool("ORCH_USE_DUMMY")
DEFAULT_RETRY_AFTER_SECONDS = int(os.environ.get("ORCH_RETRY_AFTER_SECONDS", "30"))


def _env_var_as_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    return value if value >= 0 else default


CONFIG_REFRESH_INTERVAL: float = _env_var_as_float(
    "ORCH_CONFIG_REFRESH_INTERVAL", default=30.0
)


def _parse_env_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


INBOUND_API_KEYS = frozenset(_parse_env_list(os.environ.get("ORCH_INBOUND_API_KEYS", "")))
API_KEY_HEADER = os.environ.get("ORCH_API_KEY_HEADER", "x-api-key")
ALLOWED_ORIGINS = _parse_env_list(os.environ.get("ORCH_CORS_ALLOW_ORIGINS", ""))
PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
HISTOGRAM_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

logger = logging.getLogger(__name__)


def _format_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _build_watch_files(
    mtimes: dict[str, float], watch_paths: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    keys = list(mtimes.keys())
    entries: list[tuple[str, str]] = []
    for index, path in enumerate(watch_paths):
        name = keys[index] if index < len(keys) else f"path_{index}"
        entries.append((name, path))
    return tuple(entries)


def _sanitize_watch_path(path: str) -> str:
    try:
        relative = os.path.relpath(path, CONFIG_DIR)
    except ValueError:
        relative = os.path.basename(path)
    else:
        if relative.startswith(".."):
            relative = os.path.basename(path)
    return relative.replace("\\", "/")


def _planner_watch_summary(
    watch_files: tuple[tuple[str, str], ...],
    watch_mtimes: dict[str, float],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for name, raw_path in watch_files:
        try:
            current_mtime = os.stat(raw_path).st_mtime
        except OSError:
            current_mtime = watch_mtimes.get(name)
        summary.append(
            {
                "name": name,
                "path": _sanitize_watch_path(raw_path),
                "last_modified_at": _format_timestamp(current_mtime),
            }
        )
    return summary


def _new_histogram_state() -> dict[str, Any]:
    return {"buckets": [0] * (len(HISTOGRAM_BUCKETS) + 1), "count": 0, "sum": 0.0}


PROM_COUNTER: defaultdict[tuple[str, str, str], int] = defaultdict(int)
PROM_HISTOGRAM: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
    _new_histogram_state
)


class ErrorCode(str, Enum):
    INVALID_API_KEY = "invalid_api_key"
    RATE_LIMIT = "rate_limit"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_SERVER_ERROR = "provider_server_error"
    ROUTING_ERROR = "routing_error"

    @classmethod
    def from_error_type(cls, error_type: str) -> "ErrorCode | None":
        try:
            return cls(error_type)
        except ValueError:
            return None


class _AliasProviderMap(MutableMapping[str, Any]):
    def __init__(self, data: dict[str, Any], aliases: dict[str, str]) -> None:
        self._data = data
        self._aliases = aliases

    def _canonical(self, key: str) -> str:
        return self._aliases.get(key, key)

    def __getitem__(self, key: str) -> Any:
        return self._data[self._canonical(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[self._canonical(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._data[self._canonical(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - defensive
        if not isinstance(key, str):
            return False
        return self._canonical(key) in self._data


def _build_alias_map(provider_defs: dict[str, ProviderDef]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    seen: dict[tuple[str, str, str], str] = {}
    for name, defn in provider_defs.items():
        signature = (
            (defn.type or "").strip(),
            defn.base_url or "",
            defn.model or "",
        )
        canonical = seen.setdefault(signature, name)
        if canonical != name:
            aliases[name] = canonical
    return aliases


def _apply_provider_aliases(
    provider_defs: dict[str, ProviderDef],
    registry: ProviderRegistry,
    guard_registry: ProviderGuards,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    aliases = _build_alias_map(provider_defs)
    if not aliases:
        return
    registry.providers = _AliasProviderMap(registry.providers, aliases)
    guard_registry.guards = _AliasProviderMap(guard_registry.guards, aliases)


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str
    provider: str
    model: str
    aliases: list[str] | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


def _make_response_headers(*, req_id: str, provider: str | None, attempts: int) -> dict[str, str]:
    fallback_attempts = max(attempts - 1, 0)
    provider_value = provider or "unknown"
    return {
        "x-orch-request-id": req_id,
        "x-orch-provider": provider_value,
        "x-orch-fallback-attempts": str(fallback_attempts),
    }


def _log_request_event(
    level: int,
    *,
    event: str,
    req_id: str,
    provider: str | None,
    attempts: int,
    detail: str | None = None,
) -> None:
    provider_value = provider or "unknown"
    message = f"{event} req_id={req_id} provider={provider_value} attempts={attempts}"
    if detail:
        message = f"{message} detail={detail}"
    logger.log(level, message)


def _estimate_text_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(len(normalized) // 4, 1)


def _estimate_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return _estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    total += _estimate_text_tokens(text)
        return total
    return 0


def _estimate_prompt_tokens(messages: list[dict[str, Any]], fallback: int) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        total += _estimate_content_tokens(message.get("content"))
    if total <= 0:
        return max(int(fallback), 0)
    return total


def _planner_supports_sticky(plan: Any) -> bool:
    try:
        signature = inspect.signature(plan)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    sticky_param = parameters.get("sticky_key")
    if sticky_param is None:
        return False
    return sticky_param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


cfg = load_config(CONFIG_DIR, use_dummy=USE_DUMMY)
providers = ProviderRegistry(cfg.providers)
guards = ProviderGuards(cfg.providers)
_apply_provider_aliases(cfg.providers, providers, guards, enabled=USE_DUMMY)
planner = RoutePlanner(
    cfg.router,
    cfg.providers,
    config_dir=CONFIG_DIR,
    use_dummy=USE_DUMMY,
    mtimes=cfg.mtimes,
)
planner_last_reload_at: float = time.time()
planner_watch_mtimes: dict[str, float] = dict(cfg.mtimes)
planner_watch_files: tuple[tuple[str, str], ...] = _build_watch_files(
    planner_watch_mtimes, cfg.watch_paths
)
metrics = MetricsLogger(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "metrics"))

_config_refresh_task: asyncio.Task[None] | None = None

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def _config_refresh_loop() -> None:
    try:
        while True:
            current_planner = planner
            needs_reload = current_planner.refresh()
            if needs_reload:
                reload_configuration()
                continue
            await asyncio.sleep(CONFIG_REFRESH_INTERVAL if CONFIG_REFRESH_INTERVAL > 0 else 0)
    except asyncio.CancelledError:
        raise


@app.on_event("startup")
async def _start_config_refresh() -> None:
    global _config_refresh_task
    if _config_refresh_task is None or _config_refresh_task.done():
        _config_refresh_task = asyncio.create_task(_config_refresh_loop())


@app.on_event("shutdown")
async def _stop_config_refresh() -> None:
    global _config_refresh_task
    task = _config_refresh_task
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    _config_refresh_task = None


def reload_configuration() -> None:
    global cfg, providers, guards, planner
    global planner_last_reload_at, planner_watch_mtimes, planner_watch_files
    new_cfg = load_config(CONFIG_DIR, use_dummy=USE_DUMMY)
    cfg = new_cfg
    providers = ProviderRegistry(new_cfg.providers)
    guards = ProviderGuards(new_cfg.providers)
    _apply_provider_aliases(new_cfg.providers, providers, guards, enabled=USE_DUMMY)
    planner = RoutePlanner(
        new_cfg.router,
        new_cfg.providers,
        config_dir=CONFIG_DIR,
        use_dummy=USE_DUMMY,
        mtimes=new_cfg.mtimes,
    )
    planner_last_reload_at = time.time()
    planner_watch_mtimes = dict(new_cfg.mtimes)
    planner_watch_files = _build_watch_files(planner_watch_mtimes, new_cfg.watch_paths)


def _http_status_error_details(exc: httpx.HTTPStatusError) -> tuple[int | None, str]:
    status: int | None = None
    message: str | None = None
    response = exc.response
    if response is not None:
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error_field = payload.get("error")
            if isinstance(error_field, dict):
                error_message = error_field.get("message")
                if isinstance(error_message, str) and error_message:
                    message = error_message
            if message is None:
                nested_message = payload.get("message")
                if isinstance(nested_message, str) and nested_message:
                    message = nested_message
        if message is None:
            text = response.text
            if text:
                message = text
        if message is None:
            reason = response.reason_phrase
            if reason:
                message = reason
    if message is None:
        message = str(exc)
    return status, message


def _retry_after_seconds(response: httpx.Response | None) -> int | None:
    if response is None:
        return None
    header = response.headers.get("Retry-After")
    if not header:
        return None
    value = header.strip()
    if not value:
        return None
    if value.isdigit():
        return max(int(value), 0)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = (parsed - datetime.now(timezone.utc)).total_seconds()
    return max(int(delta), 0)


def _error_type_from_status(status: int | None) -> str:
    if status == 429:
        return "rate_limit"
    if status is not None and status >= 500:
        return "provider_server_error"
    return "provider_error"


def _resolve_error_code(
    *, status_code: int, error_type: str, explicit: "ErrorCode | str | None"
) -> str:
    if explicit is not None:
        if isinstance(explicit, ErrorCode):
            return explicit.value
        return str(explicit)
    if status_code == 401:
        return ErrorCode.INVALID_API_KEY.value
    if status_code == 429:
        return ErrorCode.RATE_LIMIT.value
    if status_code >= 500:
        return ErrorCode.PROVIDER_SERVER_ERROR.value
    member = ErrorCode.from_error_type(error_type)
    if member is not None:
        return member.value
    return error_type


def _make_error_body(
    *,
    status_code: int,
    message: str,
    error_type: str,
    retry_after: int | None = None,
    code: "ErrorCode | str | None" = None,
) -> dict[str, Any]:
    resolved_code = _resolve_error_code(
        status_code=status_code, error_type=error_type, explicit=code
    )
    payload: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "code": resolved_code,
    }
    if retry_after is not None:
        payload["retry_after"] = retry_after
    return {"error": payload}


def _require_api_key(req: Request) -> None:
    if not INBOUND_API_KEYS:
        logger.warning(
            "APIキー保護が無効: ORCH_INBOUND_API_KEYS が未設定"
        )
        return
    candidate = req.headers.get(API_KEY_HEADER)
    if candidate is None:
        auth_header = req.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            candidate = auth_header[7:]
    if candidate and candidate in INBOUND_API_KEYS:
        return
    raise HTTPException(status_code=401, detail="missing or invalid api key")


async def _log_metrics(record: dict[str, Any]) -> None:
    await metrics.write(record)
    provider = str(record.get("provider") or "unknown")
    status = str(record.get("status") or "0")
    ok_label = "true" if record.get("ok") else "false"
    PROM_COUNTER[(provider, status, ok_label)] += 1
    latency_seconds = max(float(record.get("latency_ms") or 0) / 1000.0, 0.0)
    hist_entry = PROM_HISTOGRAM[(provider, ok_label)]
    buckets = hist_entry["buckets"]
    for idx, bound in enumerate(HISTOGRAM_BUCKETS):
        if latency_seconds <= bound:
            buckets[idx] += 1
    buckets[-1] += 1
    hist_entry["count"] += 1
    hist_entry["sum"] += latency_seconds


def _render_prometheus() -> bytes:
    lines: list[str] = [
        "# HELP orch_requests_total Total number of orchestrator requests",
        "# TYPE orch_requests_total counter",
    ]
    for (provider, status, ok_label), value in sorted(PROM_COUNTER.items()):
        lines.append(
            f'orch_requests_total{{provider="{provider}",status="{status}",ok="{ok_label}"}} {value}'
        )
    lines.append(
        "# HELP orch_request_latency_seconds Request latency for orchestrated requests"
    )
    lines.append("# TYPE orch_request_latency_seconds histogram")
    for (provider, ok_label), state in sorted(PROM_HISTOGRAM.items()):
        buckets = state["buckets"]
        for idx, bound in enumerate(HISTOGRAM_BUCKETS):
            le_value = format(bound, ".6g")
            count = buckets[idx]
            lines.append(
                f'orch_request_latency_seconds_bucket{{provider="{provider}",ok="{ok_label}",le="{le_value}"}} {count}'
            )
        lines.append(
            f'orch_request_latency_seconds_bucket{{provider="{provider}",ok="{ok_label}",le="+Inf"}} {buckets[-1]}'
        )
        lines.append(
            f'orch_request_latency_seconds_count{{provider="{provider}",ok="{ok_label}"}} {state["count"]}'
        )
        lines.append(
            f'orch_request_latency_seconds_sum{{provider="{provider}",ok="{ok_label}"}} {state["sum"]}'
        )
    return ("\n".join(lines) + "\n").encode("utf-8")

MAX_PROVIDER_ATTEMPTS = 3
BAD_GATEWAY_STATUS = 502
STREAMING_UNSUPPORTED_ERROR = "streaming responses are not supported"


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    alias_map = _build_alias_map(cfg.providers) if USE_DUMMY else {}
    alias_groups: dict[str, list[str]] = {}
    for alias, canonical in alias_map.items():
        alias_groups.setdefault(canonical, []).append(alias)

    models: list[ModelInfo] = []
    for name, provider_def in sorted(cfg.providers.items()):
        if name in alias_map:
            continue
        alias_list = sorted(alias_groups.get(name, ()))
        models.append(
            ModelInfo(
                id=provider_def.model or name,
                owned_by=provider_def.type,
                provider=name,
                model=provider_def.model,
                aliases=alias_list or None,
            )
        )

    return ModelListResponse(data=models)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    planner_summary = {
        "last_reload_at": _format_timestamp(planner_last_reload_at),
        "watch": _planner_watch_summary(planner_watch_files, planner_watch_mtimes),
    }
    return {
        "status": "ok",
        "providers": list(cfg.providers.keys()),
        "planner": planner_summary,
    }

@app.get("/metrics")
async def metrics_endpoint(req: Request) -> Response:
    _require_api_key(req)
    return Response(_render_prometheus(), media_type=PROM_CONTENT_TYPE)


@app.get("/v1/models")
async def list_models() -> _ModelListResponse:
    models: list[_ModelInfo] = []
    for name, definition in sorted(cfg.providers.items()):
        owner = definition.type or name
        model_entry: _ModelInfo = {
            "id": name,
            "object": "model",
            "owned_by": owner,
        }
        models.append(model_entry)
    payload: _ModelListResponse = {"object": "list", "data": models}
    return payload


@app.post("/v1/chat/completions")
async def chat_completions(req: Request, body: ChatRequest):
    try:
        _require_api_key(req)
    except HTTPException as exc:
        if exc.status_code != 401:
            raise
        req_id = str(uuid.uuid4())
        headers = _make_response_headers(req_id=req_id, provider=None, attempts=0)
        detail = exc.detail if isinstance(exc.detail, str) else "missing or invalid api key"
        error_body = _make_error_body(
            status_code=exc.status_code,
            message=detail,
            error_type="authentication_error",
            code=ErrorCode.INVALID_API_KEY,
        )
        return JSONResponse(error_body, status_code=exc.status_code, headers=headers)
    header_value = (
        req.headers.get(cfg.router.defaults.task_header)
        if cfg.router.defaults.task_header
        else None
    )
    task = header_value or cfg.router.defaults.task_header_value or "DEFAULT"
    start = time.perf_counter()
    req_id = str(uuid.uuid4())
    normalized_messages = [
        message.model_dump(mode="json", exclude_none=True)
        for message in body.messages
    ]
    function_call = getattr(body, "function_call", None)
    additional_options: dict[str, Any] = {}
    typed_options: dict[str, Any] = {}
    typed_option_fields = (
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "response_format",
    )
    for field in typed_option_fields:
        if field in body.model_fields_set:
            value = getattr(body, field)
            if value is not None:
                typed_options[field] = value
    extra_options_source = getattr(body, "model_extra", None)
    if isinstance(extra_options_source, dict):
        for key, value in extra_options_source.items():
            if (
                key == "function_call"
                or value is None
                or key in additional_options
                or key in typed_option_fields
            ):
                continue
            additional_options[key] = value
    if "temperature" in body.model_fields_set and body.temperature is not None:
        temperature = body.temperature
    else:
        temperature = cfg.router.defaults.temperature
    if "max_tokens" in body.model_fields_set and body.max_tokens is not None:
        max_tokens = body.max_tokens
    else:
        max_tokens = cfg.router.defaults.max_tokens

    provider_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": body.tools,
        "tool_choice": body.tool_choice,
        "function_call": function_call,
        **typed_options,
        **additional_options,
    }

    estimated_prompt_tokens = _estimate_prompt_tokens(normalized_messages, max_tokens)

    sticky_key_header = req.headers.get("x-orch-sticky-key")
    if not sticky_key_header:
        sticky_key_header = req.headers.get("X-Orch-Session")
    sticky_key = sticky_key_header.strip() if sticky_key_header else None

    try:
        plan_fn = planner.plan
        if _planner_supports_sticky(plan_fn):
            route = plan_fn(task, sticky_key=sticky_key)
        else:
            route = plan_fn(task)
    except ValueError as exc:
        detail = str(exc) or "routing unavailable"
        await _log_metrics({
            "req_id": req_id,
            "ts": time.time(),
            "task": task,
            "provider": "unroutable",
            "model": body.model,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "ok": False,
            "status": 400,
            "error": detail,
            "usage_prompt": 0,
            "usage_completion": 0,
            "retries": 0,
        })
        headers = _make_response_headers(req_id=req_id, provider=None, attempts=0)
        error_body = _make_error_body(
            status_code=400,
            message=detail,
            error_type="routing_error",
        )
        return JSONResponse(error_body, status_code=400, headers=headers)
    if body.stream:
        return await _stream_chat_response(
            model=body.model,
            route=route,
            task=task,
            req_id=req_id,
            start=start,
            normalized_messages=normalized_messages,
            provider_kwargs=provider_kwargs,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
    last_err: str | None = None
    usage_prompt = 0
    usage_completion = 0
    attempt_count = 0
    last_provider = route.primary
    last_model = body.model
    last_error_type: str | None = None
    last_retry_after: int | None = None
    success_response: ProviderChatResponse | None = None
    success_record: dict[str, object] | None = None

    abort_processing = False
    abort_status: int | None = None
    abort_error: str | None = None
    abort_error_type: str | None = None
    abort_retry_after: int | None = None
    for provider_name in [route.primary] + route.fallback:
        if attempt_count >= MAX_PROVIDER_ATTEMPTS:
            break
        try:
            prov = providers.get(provider_name)
        except KeyError:
            continue
        try:
            guard = guards.get(provider_name)
        except (AssertionError, KeyError):
            continue
        should_abort = False
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            if attempt_count >= MAX_PROVIDER_ATTEMPTS:
                break
            should_abort = False
            async with guard.acquire(
                estimated_prompt_tokens=estimated_prompt_tokens
            ) as lease:
                attempt_count += 1
                try:
                    resp = await prov.chat(
                        body.model,
                        normalized_messages,
                        **provider_kwargs,
                    )
                except Exception as exc:
                    if getattr(guard, "_tpm_bucket", None) is not None:
                        guard.record_usage(
                            lease,
                            usage_prompt_tokens=0,
                            usage_completion_tokens=0,
                        )
                    planner.record_failure(provider_name)
                    last_err = str(exc)
                    last_provider = provider_name
                    last_model = prov.model or body.model
                    last_error_type = "provider_error"
                    if isinstance(exc, UnsupportedContentBlockError):
                        abort_error = last_err or "unsupported content block"
                        abort_status = 400
                        abort_error_type = "provider_error"
                        should_abort = True
                    elif isinstance(exc, httpx.HTTPStatusError):
                        status, message = _http_status_error_details(exc)
                        retry_after = _retry_after_seconds(exc.response)
                        error_type = _error_type_from_status(status)
                        last_error_type = error_type
                        if retry_after is not None:
                            last_retry_after = retry_after
                        if status == 429:
                            abort_error = message
                            abort_status = status
                            abort_error_type = error_type
                            abort_retry_after = (
                                retry_after if retry_after is not None else DEFAULT_RETRY_AFTER_SECONDS
                            )
                            should_abort = True
                        elif status is not None and 400 <= status < 500:
                            abort_error = message
                            last_err = abort_error
                            abort_status = status
                            abort_error_type = error_type
                            should_abort = True
                        elif status is not None and status >= 500 and last_retry_after is None:
                            last_retry_after = (
                                retry_after if retry_after is not None else DEFAULT_RETRY_AFTER_SECONDS
                            )
                else:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    usage_prompt = resp.usage_prompt_tokens or 0
                    usage_completion = resp.usage_completion_tokens or 0
                    guard.record_usage(
                        lease,
                        usage_prompt_tokens=usage_prompt,
                        usage_completion_tokens=usage_completion,
                    )
                    planner.record_success(provider_name)
                    success_response = resp
                    last_provider = provider_name
                    last_model = resp.model or body.model or prov.model
                    success_record = {
                        "req_id": req_id,
                        "ts": time.time(),
                        "task": task,
                        "provider": provider_name,
                        "model": last_model,
                        "latency_ms": latency_ms,
                        "ok": True,
                        "status": resp.status_code,
                        "retries": attempt_count - 1,
                        "usage_prompt": usage_prompt,
                        "usage_completion": usage_completion,
                    }
                    break

            if should_abort:
                abort_processing = True
                break

            if (
                attempt < MAX_PROVIDER_ATTEMPTS
                and attempt_count < MAX_PROVIDER_ATTEMPTS
            ):
                await asyncio.sleep(min(0.25 * attempt, 2.0))  # simple backoff

        if (
            success_record is not None
            or abort_processing
            or attempt_count >= MAX_PROVIDER_ATTEMPTS
        ):
            break

    if success_response is not None and success_record is not None:
        await _log_metrics(success_record)
        log_level = logging.WARNING if attempt_count > 1 else logging.INFO
        event_name = (
            "chat.completions fallback" if attempt_count > 1 else "chat.completions success"
        )
        _log_request_event(
            log_level,
            event=event_name,
            req_id=req_id,
            provider=last_provider,
            attempts=attempt_count,
        )
        headers = _make_response_headers(
            req_id=req_id, provider=last_provider, attempts=attempt_count
        )
        return JSONResponse(
            chat_response_from_provider(success_response), headers=headers
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    failure_status = BAD_GATEWAY_STATUS
    failure_error = last_err or "all providers failed"
    if abort_processing and abort_status is not None:
        failure_status = abort_status
        failure_error = abort_error or failure_error
    failure_error_type = (
        abort_error_type or last_error_type or _error_type_from_status(failure_status)
    )
    failure_retry_after = abort_retry_after if abort_processing else last_retry_after
    if failure_retry_after is None and (
        failure_status == 429 or failure_status >= 500
    ):
        failure_retry_after = DEFAULT_RETRY_AFTER_SECONDS
    failure_record = {
        "req_id": req_id,
        "ts": time.time(),
        "task": task,
        "provider": last_provider,
        "model": last_model,
        "latency_ms": latency_ms,
        "ok": False,
        "status": failure_status,
        "error": failure_error,
        "usage_prompt": 0,
        "usage_completion": 0,
        "retries": max(attempt_count - 1, 0),
    }
    if failure_retry_after is not None:
        failure_record["retry_after"] = failure_retry_after
    await _log_metrics(failure_record)
    _log_request_event(
        logging.ERROR,
        event="chat.completions failure",
        req_id=req_id,
        provider=last_provider,
        attempts=attempt_count,
        detail=failure_error,
    )
    error_body = _make_error_body(
        status_code=failure_status,
        message=failure_error,
        error_type=failure_error_type,
        retry_after=failure_retry_after,
    )
    headers = _make_response_headers(
        req_id=req_id, provider=last_provider, attempts=attempt_count
    )
    return JSONResponse(error_body, status_code=failure_status, headers=headers)


@asynccontextmanager
async def _guard_context(
    guard: Any | None,
    *,
    estimated_prompt_tokens: int,
) -> AsyncIterator[Any]:
    if guard is None:
        yield None
        return
    acquire = getattr(guard, "acquire", None)
    if callable(acquire):
        try:
            context = acquire(estimated_prompt_tokens=estimated_prompt_tokens)
        except TypeError:
            context = acquire()
        async with context as lease:
            yield lease
            return
    if hasattr(guard, "__aenter__") and hasattr(guard, "__aexit__"):
        async with guard as lease:
            yield lease
            return
    yield None


async def _stream_chat_response(
    *,
    model: str,
    route: RouteDef,
    task: str,
    req_id: str,
    start: float,
    normalized_messages: list[dict[str, Any]],
    provider_kwargs: dict[str, Any],
    estimated_prompt_tokens: int,
) -> JSONResponse | StreamingResponse:
    providers_to_try = [route.primary] + route.fallback
    if not providers_to_try:
        headers = _make_response_headers(
            req_id=req_id, provider=route.primary or "unroutable", attempts=0
        )
        return JSONResponse(
            {"error": {"message": "routing unavailable"}},
            status_code=400,
            headers=headers,
        )

    def _normalize_event(raw_event: Any) -> tuple[str | None, Any, bool]:
        def _extract_event(source: Any) -> tuple[str | None, Any]:
            if isinstance(source, dict):
                name = source.get("event") or source.get("event_type")
                payload = source.get("data") if "data" in source else {
                    key: value for key, value in source.items() if key not in {"event", "event_type"}
                }
                return name if isinstance(name, str) else None, payload
            name_attr = getattr(source, "event_type", None) or getattr(source, "event", None)
            name = name_attr.strip() if isinstance(name_attr, str) else None
            payload = getattr(source, "data", source)
            return name, payload

        def _map_event(name: str | None) -> tuple[str | None, bool]:
            if not name:
                return None, False
            normalized = name.strip()
            if not normalized:
                return None, False
            if normalized in {"chat.completion.chunk", "telemetry.usage", "done"}:
                return normalized, normalized == "done"
            lowered = normalized.lower()
            if lowered in {"usage"} or lowered.endswith(".usage"):
                return "telemetry.usage", False
            if lowered in {
                "message_stop",
                "response.stop",
                "response_completed",
                "response.completed",
                "stop",
                "done",
            } or lowered.endswith("_stop") or lowered.endswith(".stop"):
                return "done", True
            if lowered in {
                "chunk",
                "delta",
                "message",
                "message_start",
                "message_delta",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "response",
            } or lowered.endswith("_delta") or lowered.endswith("_start") or lowered.endswith("_chunk") or lowered.endswith(".delta") or lowered.endswith(".chunk"):
                return "chat.completion.chunk", False
            return normalized, False

        def _coerce_payload(payload: Any) -> Any:
            if payload is None or isinstance(payload, (str, bytes)):
                return payload
            model_dump = getattr(payload, "model_dump", None)
            if callable(model_dump):
                try:
                    return model_dump(mode="json", exclude_none=True)
                except TypeError:
                    return model_dump()
            if is_dataclass(payload):
                return asdict(payload)
            return payload

        event_name, payload = _extract_event(raw_event)
        mapped_event, is_terminal = _map_event(event_name)
        payload = _coerce_payload(payload)

        if isinstance(payload, dict):
            payload = dict(payload)
            payload.pop("event_type", None)
            payload.pop("event", None)
            payload.pop("raw", None)

        if is_terminal:
            payload = {}

        return mapped_event, payload, is_terminal

    def _encode_normalized(mapped_event: str | None, payload: Any) -> bytes:
        if isinstance(payload, bytes):
            data_text = payload.decode("utf-8", errors="ignore")
        elif isinstance(payload, str):
            data_text = payload
        elif payload is None:
            data_text = ""
        else:
            data_text = json.dumps(payload)

        lines = []
        if mapped_event:
            lines.append(f"event: {mapped_event}")
        lines.append(f"data: {data_text}")
        return ("\n".join(lines) + "\n\n").encode("utf-8")

    attempts = 0
    last_error: str | None = None
    last_status: int | None = None
    last_error_type: str | None = None
    last_retry_after: int | None = None
    last_provider = providers_to_try[0]
    last_model = model

    for provider_name in providers_to_try:
        try:
            provider = providers.get(provider_name)
        except KeyError:
            continue
        attempts += 1
        provider_model = provider.model or model
        if not hasattr(provider, "chat_stream"):
            planner.record_failure(provider_name)
            failure_record = {
                "req_id": req_id,
                "ts": time.time(),
                "task": task,
                "provider": provider_name,
                "model": provider_model,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "ok": False,
                "status": 400,
                "error": STREAMING_UNSUPPORTED_ERROR,
                "retries": attempts - 1,
                "usage_prompt": 0,
                "usage_completion": 0,
            }
            await _log_metrics(failure_record)
            last_provider = "unsupported"
            last_model = provider_model
            last_status = 400
            last_error = STREAMING_UNSUPPORTED_ERROR
            last_error_type = "provider_error"
            last_retry_after = None
            continue
        try:
            guard = guards.get(provider_name)
        except (AssertionError, KeyError):
            guard = None
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        usage_prompt_tokens = usage_completion_tokens = 0
        usage_recorded = False
        guard_lease: object | None = None
        def _handle_usage_event(normalized: tuple[str | None, Any, bool]) -> None:
            nonlocal usage_prompt_tokens, usage_completion_tokens, usage_recorded, guard_lease
            event_name, payload, _ = normalized
            if event_name != "telemetry.usage" or not isinstance(payload, dict):
                return
            usage_payload = payload.get("usage")
            data = usage_payload if isinstance(usage_payload, dict) else payload
            if not isinstance(data, dict):
                return
            prompt_value = data.get("prompt_tokens")
            if isinstance(prompt_value, int) and prompt_value > usage_prompt_tokens:
                usage_prompt_tokens = prompt_value
            completion_value = data.get("completion_tokens")
            if isinstance(completion_value, int) and completion_value > usage_completion_tokens:
                usage_completion_tokens = completion_value
            if (
                guard is not None
                and guard_lease is not None
                and not usage_recorded
                and (isinstance(prompt_value, int) or isinstance(completion_value, int))
            ):
                guard.record_usage(
                    guard_lease,
                    usage_prompt_tokens=usage_prompt_tokens,
                    usage_completion_tokens=usage_completion_tokens,
                )
                usage_recorded = True

        async def _write_metrics(
            *,
            ok: bool,
            status: int,
            latency_ms: int,
            retries: int,
            error: str | None = None,
            retry_after: int | None = None,
        ) -> None:
            record = {
                "req_id": req_id,
                "ts": time.time(),
                "task": task,
                "provider": provider_name,
                "model": provider_model,
                "latency_ms": latency_ms,
                "ok": ok,
                "status": status,
                "retries": retries,
                "usage_prompt": usage_prompt_tokens,
                "usage_completion": usage_completion_tokens,
            }
            if error is not None:
                record["error"] = error
            if retry_after is not None:
                record["retry_after"] = retry_after
            await _log_metrics(record)

        async def _handle_http_status_error(exc: httpx.HTTPStatusError) -> None:
            planner.record_failure(provider_name)
            status, message = _http_status_error_details(exc)
            status_code = status or BAD_GATEWAY_STATUS
            retry_after = _retry_after_seconds(exc.response)
            error_type = _error_type_from_status(status_code)
            if status_code == 429 or (400 <= status_code < 500):
                if retry_after is None and status_code == 429:
                    retry_after = DEFAULT_RETRY_AFTER_SECONDS
                await queue.put(
                    (
                        "error",
                        {
                            "status": status_code,
                            "message": message,
                            "type": error_type,
                            "retry_after": retry_after,
                        },
                    )
                )
                return
            if retry_after is None and status_code >= 500:
                retry_after = DEFAULT_RETRY_AFTER_SECONDS
            await queue.put(
                (
                    "fallback",
                    {
                        "status": status_code,
                        "message": message,
                        "type": error_type,
                        "retry_after": retry_after,
                    },
                )
            )

        async def _emit_generic_failure(exc: Exception) -> None:
            planner.record_failure(provider_name)
            await queue.put(
                (
                    "fallback",
                    {
                        "status": BAD_GATEWAY_STATUS,
                        "message": str(exc) or "provider error",
                        "type": "provider_server_error",
                        "retry_after": DEFAULT_RETRY_AFTER_SECONDS,
                    },
                )
            )

        async def producer() -> None:
            nonlocal guard_lease
            try:
                async with _guard_context(
                    guard, estimated_prompt_tokens=estimated_prompt_tokens
                ) as lease:
                    guard_lease = lease
                    try:
                        stream_iter = provider.chat_stream(
                            model,
                            normalized_messages,
                            **provider_kwargs,
                        )
                        if inspect.isawaitable(stream_iter) and not hasattr(
                            stream_iter, "__anext__"
                        ):
                            stream_iter = await stream_iter
                    except UnsupportedContentBlockError as exc:
                        planner.record_failure(provider_name)
                        await queue.put(
                            (
                                "error",
                                {
                                    "status": 400,
                                    "message": str(exc) or "unsupported content block",
                                    "type": "provider_error",
                                    "retry_after": None,
                                },
                            )
                        )
                        return
                    except httpx.HTTPStatusError as exc:
                        await _handle_http_status_error(exc)
                        return
                    except Exception as exc:
                        await _emit_generic_failure(exc)
                        return
                    try:
                        first_event = await anext(stream_iter, None)
                    except UnsupportedContentBlockError as exc:
                        planner.record_failure(provider_name)
                        await queue.put(
                            (
                                "error",
                                {
                                    "status": 400,
                                    "message": str(exc) or "unsupported content block",
                                    "type": "provider_error",
                                    "retry_after": None,
                                },
                            )
                        )
                        return
                    except httpx.HTTPStatusError as exc:
                        await _handle_http_status_error(exc)
                        return
                    except Exception as exc:
                        await _emit_generic_failure(exc)
                        return
                    if first_event is not None:
                        normalized_first = _normalize_event(first_event)
                        _handle_usage_event(normalized_first)
                        mapped, payload, _ = normalized_first
                        await queue.put(("data", _encode_normalized(mapped, payload)))
                    async for raw_event in stream_iter:
                        normalized_event = _normalize_event(raw_event)
                        _handle_usage_event(normalized_event)
                        mapped, payload, _ = normalized_event
                        await queue.put(("data", _encode_normalized(mapped, payload)))
                    planner.record_success(provider_name)
                    await _write_metrics(
                        ok=True,
                        status=200,
                        latency_ms=int((time.perf_counter() - start) * 1000),
                        retries=attempts - 1,
                    )
                    await queue.put(("done", None))
            except asyncio.CancelledError:
                raise
            except UnsupportedContentBlockError as exc:
                planner.record_failure(provider_name)
                await queue.put(
                    (
                        "error",
                        {
                            "status": 400,
                            "message": str(exc) or "unsupported content block",
                            "type": "provider_error",
                            "retry_after": None,
                        },
                    )
                )
            except httpx.HTTPStatusError as exc:
                await _handle_http_status_error(exc)
            except Exception as exc:
                await _emit_generic_failure(exc)
            finally:
                guard_lease = None

        producer_task = asyncio.create_task(producer())
        first_kind, first_payload = await queue.get()
        buffered_events: list[tuple[str, Any]] = []
        while True:
            try:
                next_kind, next_payload = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if next_kind == "fallback":
                first_kind = "fallback"
                first_payload = next_payload
                buffered_events.clear()
                break
            buffered_events.append((next_kind, next_payload))
        if first_kind == "error":
            await producer_task
            status_code = int(first_payload["status"])
            message = str(first_payload["message"]) if first_payload["message"] else ""
            retry_after = first_payload.get("retry_after")
            await _write_metrics(
                ok=False,
                status=status_code,
                latency_ms=int((time.perf_counter() - start) * 1000),
                retries=attempts - 1,
                error=message,
                retry_after=retry_after,
            )
            error_payload: dict[str, Any] = {
                "message": message,
                "type": str(first_payload.get("type") or _error_type_from_status(status_code)),
            }
            if retry_after is not None:
                error_payload["retry_after"] = retry_after
            headers = _make_response_headers(
                req_id=req_id, provider=provider_name, attempts=attempts
            )
            return JSONResponse(
                {"error": error_payload}, status_code=status_code, headers=headers
            )
        if first_kind == "fallback":
            await producer_task
            status_code = int(first_payload["status"])
            message = str(first_payload.get("message") or "provider error")
            retry_after = first_payload.get("retry_after")
            await _write_metrics(
                ok=False,
                status=status_code,
                latency_ms=int((time.perf_counter() - start) * 1000),
                retries=attempts - 1,
                error=message,
                retry_after=retry_after,
            )
            last_provider = provider_name
            last_model = provider_model
            last_status = status_code
            last_error = message
            last_error_type = str(first_payload.get("type") or _error_type_from_status(last_status))
            last_retry_after = retry_after
            if last_retry_after is None and last_status >= 500:
                last_retry_after = DEFAULT_RETRY_AFTER_SECONDS
            continue
        if first_kind not in {"data", "done"}:
            await producer_task
            raise RuntimeError("unexpected stream signal")
        initial_chunks: list[bytes] = []
        initial_done = False
        if first_kind == "data":
            initial_chunks.append(first_payload)
        elif first_kind == "done":
            initial_done = True
        for buffered_kind, buffered_payload in buffered_events:
            if buffered_kind == "data":
                initial_chunks.append(buffered_payload)
            elif buffered_kind == "done":
                initial_done = True
                break

        done_frame = b"data: [DONE]\n\n"

        async def event_source() -> Any:
            done_sent = False

            def _next_done() -> bytes | None:
                nonlocal done_sent
                if done_sent:
                    return None
                done_sent = True
                return done_frame

            try:
                for chunk in initial_chunks:
                    yield chunk
                if initial_done:
                    done_payload = _next_done()
                    if done_payload is not None:
                        yield done_payload
                    return
                while True:
                    kind, payload = await queue.get()
                    if kind == "data":
                        yield payload
                    elif kind == "done":
                        done_payload = _next_done()
                        if done_payload is not None:
                            yield done_payload
                        break
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

        response = StreamingResponse(event_source(), media_type="text/event-stream")
        response.headers.update(
            _make_response_headers(
                req_id=req_id,
                provider=provider_name,
                attempts=attempts,
            )
        )
        return response

    latency_ms = int((time.perf_counter() - start) * 1000)
    failure_status = last_status or BAD_GATEWAY_STATUS
    failure_error = last_error or "all providers failed"
    failure_error_type = last_error_type or _error_type_from_status(failure_status)
    if last_retry_after is None and (failure_status == 429 or failure_status >= 500):
        last_retry_after = DEFAULT_RETRY_AFTER_SECONDS
    failure_record: dict[str, Any] = {
        "req_id": req_id,
        "ts": time.time(),
        "task": task,
        "provider": last_provider,
        "model": last_model,
        "latency_ms": latency_ms,
        "ok": False,
        "status": failure_status,
        "error": failure_error,
        "retries": max(attempts - 1, 0),
        "usage_prompt": 0,
        "usage_completion": 0,
    }
    if last_retry_after is not None:
        failure_record["retry_after"] = last_retry_after
    await _log_metrics(failure_record)
    error_payload = {"message": failure_error, "type": failure_error_type}
    if last_retry_after is not None:
        error_payload["retry_after"] = last_retry_after
    headers = _make_response_headers(
        req_id=req_id, provider=last_provider, attempts=attempts
    )
    return JSONResponse(
        {"error": error_payload}, status_code=failure_status, headers=headers
    )
