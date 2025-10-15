import asyncio
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .metrics import MetricsLogger
from .providers import ProviderRegistry
from .rate_limiter import ProviderGuards
from .router import RoutePlanner, load_config
from .types import ChatRequest, chat_response_from_provider

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

MAX_PROVIDER_ATTEMPTS = 3

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
    normalized_messages = [{"role": m.role, "content": m.content} for m in body.messages]

    for provider_name in [route.primary] + route.fallback:
        prov = providers.get(provider_name)
        guard = guards.get(provider_name)
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            async with guard:
                try:
                    resp = await prov.chat(
                        body.model,
                        normalized_messages,
                        temperature=body.temperature,
                        max_tokens=body.max_tokens,
                    )
                except Exception as exc:
                    last_err = str(exc)
                    await metrics.write(
                        {
                            "req_id": req_id,
                            "ts": time.time(),
                            "task": task,
                            "provider": provider_name,
                            "model": prov.model,
                            "latency_ms": int((time.perf_counter() - start) * 1000),
                            "ok": False,
                            "status": 0,
                            "error": last_err,
                            "usage_prompt": 0,
                            "usage_completion": 0,
                            "retries": attempt - 1,
                        }
                    )
                else:
                    latency = int((time.perf_counter() - start) * 1000)
                    usage_prompt = resp.usage_prompt_tokens or 0
                    usage_completion = resp.usage_completion_tokens or 0
                    await metrics.write(
                        {
                            "req_id": req_id,
                            "ts": time.time(),
                            "task": task,
                            "provider": provider_name,
                            "model": prov.model,
                            "latency_ms": latency,
                            "ok": True,
                            "status": resp.status_code,
                            "retries": attempt - 1,
                            "usage_prompt": usage_prompt,
                            "usage_completion": usage_completion,
                        }
                    )
                    return JSONResponse(chat_response_from_provider(resp))

            if attempt < MAX_PROVIDER_ATTEMPTS:
                await asyncio.sleep(min(0.25 * attempt, 2.0))  # simple backoff

    return JSONResponse({"error": {"message": last_err or "all providers failed"}}, status_code=502)
