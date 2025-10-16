import asyncio
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .metrics import MetricsLogger
from .providers import ProviderRegistry
from .rate_limiter import ProviderGuards
from .router import RoutePlanner, load_config
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

cfg = load_config(CONFIG_DIR, use_dummy=USE_DUMMY)
providers = ProviderRegistry(cfg.providers)
guards = ProviderGuards(cfg.providers)
planner = RoutePlanner(cfg.router, cfg.providers)
metrics = MetricsLogger(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "metrics"))


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

MAX_PROVIDER_ATTEMPTS = 3
BAD_GATEWAY_STATUS = 502
STREAMING_UNSUPPORTED_ERROR = "streaming responses are not supported"

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "providers": list(cfg.providers.keys())}

@app.post("/v1/chat/completions")
async def chat_completions(req: Request, body: ChatRequest):
    header_value = (
        req.headers.get(cfg.router.defaults.task_header)
        if cfg.router.defaults.task_header
        else None
    )
    task = header_value or cfg.router.defaults.task_header_value or "DEFAULT"
    start = time.perf_counter()
    req_id = str(uuid.uuid4())
    if body.stream:
        reason = STREAMING_UNSUPPORTED_ERROR
        await metrics.write(
            {
                "req_id": req_id,
                "ts": time.time(),
                "task": task,
                "provider": "unsupported",
                "model": body.model,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "ok": False,
                "status": 400,
                "error": reason,
                "usage_prompt": 0,
                "usage_completion": 0,
                "retries": 0,
            }
        )
        return JSONResponse({"error": {"message": reason}}, status_code=400)
    try:
        route = planner.plan(task)
    except ValueError as exc:
        detail = str(exc) or "routing unavailable"
        await metrics.write({
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
    last_err: str | None = None
    usage_prompt = 0
    usage_completion = 0
    attempt_count = 0
    last_provider = route.primary
    last_model = body.model
    success_response: ProviderChatResponse | None = None
    success_record: dict[str, object] | None = None
    normalized_messages = [
        message.model_dump(mode="json", exclude_none=True)
        for message in body.messages
    ]
    if "temperature" in body.model_fields_set and body.temperature is not None:
        temperature = body.temperature
    else:
        temperature = cfg.router.defaults.temperature
    if "max_tokens" in body.model_fields_set and body.max_tokens is not None:
        max_tokens = body.max_tokens
    else:
        max_tokens = cfg.router.defaults.max_tokens

    abort_processing = False
    abort_status: int | None = None
    abort_error: str | None = None
    for provider_name in [route.primary] + route.fallback:
        prov = providers.get(provider_name)
        guard = guards.get(provider_name)
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            async with guard:
                attempt_count += 1
                should_abort = False
                try:
                    resp = await prov.chat(
                        body.model,
                        normalized_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    last_err = str(exc)
                    last_provider = provider_name
                    last_model = prov.model or body.model
                    if isinstance(exc, httpx.HTTPStatusError):
                        status, message = _http_status_error_details(exc)
                        if status is not None and status != 429 and 400 <= status < 500:
                            abort_error = message
                            last_err = abort_error
                            abort_status = status
                            should_abort = True
                else:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    usage_prompt = resp.usage_prompt_tokens or 0
                    usage_completion = resp.usage_completion_tokens or 0
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
        await metrics.write(success_record)
        return JSONResponse(chat_response_from_provider(success_response))

    latency_ms = int((time.perf_counter() - start) * 1000)
    failure_status = BAD_GATEWAY_STATUS
    failure_error = last_err or "all providers failed"
    if abort_processing and abort_status is not None:
        failure_status = abort_status
        failure_error = abort_error or failure_error
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
    await metrics.write(failure_record)
    return JSONResponse(
        {"error": {"message": failure_error}}, status_code=failure_status
    )
