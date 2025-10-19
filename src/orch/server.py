import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


from .metrics import MetricsLogger
from .providers import ProviderRegistry, UnsupportedContentBlockError
from .rate_limiter import ProviderGuards
from .router import RouteDef, RoutePlanner, load_config
from .types import ChatRequest, ProviderChatResponse, chat_response_from_provider

app = FastAPI(title="llm-orch")

CONFIG_DIR = os.environ.get("ORCH_CONFIG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config"))

TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
FALSY_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})


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


def _new_histogram_state() -> dict[str, Any]:
    return {"buckets": [0] * (len(HISTOGRAM_BUCKETS) + 1), "count": 0, "sum": 0.0}


PROM_COUNTER: defaultdict[tuple[str, str, str], int] = defaultdict(int)
PROM_HISTOGRAM: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
    _new_histogram_state
)


def _make_response_headers(*, req_id: str, provider: str | None, attempts: int) -> dict[str, str]:
    fallback_attempts = max(attempts - 1, 0)
    provider_value = provider or "unknown"
    return {
        "x-orch-request-id": req_id,
        "x-orch-provider": provider_value,
        "x-orch-fallback-attempts": str(fallback_attempts),
    }


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


cfg = load_config(CONFIG_DIR, use_dummy=USE_DUMMY)
providers = ProviderRegistry(cfg.providers)
guards = ProviderGuards(cfg.providers)
planner = RoutePlanner(cfg.router, cfg.providers)
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
            planner.refresh()
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


def _require_api_key(req: Request) -> None:
    if not INBOUND_API_KEYS:
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

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "providers": list(cfg.providers.keys())}

@app.get("/metrics")
async def metrics_endpoint(req: Request) -> Response:
    _require_api_key(req)
    return Response(_render_prometheus(), media_type=PROM_CONTENT_TYPE)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request, body: ChatRequest):
    _require_api_key(req)
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

    try:
        route = planner.plan(task)
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
        raise HTTPException(status_code=400, detail=detail)
    if body.stream:
        return await _stream_chat_response(
            model=body.model,
            route=route,
            task=task,
            req_id=req_id,
            start=start,
            normalized_messages=normalized_messages,
            provider_kwargs=provider_kwargs,
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
        prov = providers.get(provider_name)
        guard = guards.get(provider_name)
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
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

            if attempt < MAX_PROVIDER_ATTEMPTS:
                await asyncio.sleep(min(0.25 * attempt, 2.0))  # simple backoff

        if success_record is not None or abort_processing:
            break

    if success_response is not None and success_record is not None:
        await _log_metrics(success_record)
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
    error_payload: dict[str, Any] = {
        "message": failure_error,
        "type": failure_error_type,
    }
    if failure_retry_after is not None:
        error_payload["retry_after"] = failure_retry_after
    headers = _make_response_headers(
        req_id=req_id, provider=last_provider, attempts=attempt_count
    )
    return JSONResponse(
        {"error": error_payload}, status_code=failure_status, headers=headers
    )


async def _stream_chat_response(
    *,
    model: str,
    route: RouteDef,
    task: str,
    req_id: str,
    start: float,
    normalized_messages: list[dict[str, Any]],
    provider_kwargs: dict[str, Any],
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

    def _encode_event(raw_event: Any) -> bytes:
        if isinstance(raw_event, dict):
            event_name = raw_event.get("event")
            data_field = raw_event.get("data")
        else:
            event_name = None
            data_field = raw_event
        if not isinstance(event_name, str):
            event_name = None
        if data_field is not None and not isinstance(data_field, str):
            model_dump = getattr(data_field, "model_dump", None)
            if callable(model_dump):
                try:
                    data_field = model_dump(mode="json", exclude_none=True)
                except TypeError:
                    data_field = model_dump()
            elif is_dataclass(data_field):
                data_field = asdict(data_field)
        if isinstance(data_field, str):
            data_text = data_field
        elif data_field is None:
            data_text = ""
        else:
            data_text = json.dumps(data_field)
        lines = []
        if event_name:
            lines.append(f"event: {event_name}")
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
        attempts += 1
        provider = providers.get(provider_name)
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
        guard = guards.get(provider_name)
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

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
                "usage_prompt": 0,
                "usage_completion": 0,
            }
            if error is not None:
                record["error"] = error
            if retry_after is not None:
                record["retry_after"] = retry_after
            await _log_metrics(record)

        async def producer() -> None:
            try:
                async with guard:
                    stream_iter = provider.chat_stream(
                        model,
                        normalized_messages,
                        **provider_kwargs,
                    )
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
                        return
                    except Exception as exc:
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
                        return
                    if first_event is not None:
                        await queue.put(("data", _encode_event(first_event)))
                    async for raw_event in stream_iter:
                        await queue.put(("data", _encode_event(raw_event)))
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
            except Exception as exc:
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

        async def event_source() -> Any:
            try:
                for chunk in initial_chunks:
                    yield chunk
                if initial_done:
                    yield b"data: [DONE]\n\n"
                    return
                if initial_done:
                    yield b"data: [DONE]\n\n"
                    return
                while True:
                    kind, payload = await queue.get()
                    if kind == "data":
                        yield payload
                    elif kind == "done":
                        yield b"data: [DONE]\n\n"
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
