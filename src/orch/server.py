import asyncio
import os
import time
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from .router import LoadedConfig, RoutePlanner, load_config
from .metrics import MetricsLogger
from .rate_limiter import ProviderGuards
from .types import ChatRequest, chat_response_from_provider
from .providers import ProviderRegistry

app = FastAPI(title="llm-orch")

CONFIG_DIR = os.environ.get("ORCH_CONFIG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config"))

TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}


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


METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "metrics")

cfg: LoadedConfig
providers: ProviderRegistry
guards: ProviderGuards
planner: RoutePlanner
metrics: MetricsLogger


def init_dependencies(*, use_dummy: bool) -> None:
    global cfg, providers, guards, planner, metrics

    cfg = load_config(CONFIG_DIR, use_dummy=use_dummy)
    providers = ProviderRegistry(cfg.providers)
    guards = ProviderGuards(cfg.providers)
    planner = RoutePlanner(cfg.router, cfg.providers)
    metrics = MetricsLogger(METRICS_DIR)


init_dependencies(use_dummy=_env_var_as_bool("ORCH_USE_DUMMY"))

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "providers": list(cfg.providers.keys())}

@app.post("/v1/chat/completions")
async def chat_completions(req: Request, body: ChatRequest, x_orch_task_kind: str | None = Header(default=None)):
    task = x_orch_task_kind or cfg.router.defaults.task_header_value or "DEFAULT"
    route = planner.plan(task)
    attempt = 0
    start = time.perf_counter()
    last_err: str | None = None
    usage_prompt = 0
    usage_completion = 0
    normalized_messages = [{"role": m.role, "content": m.content} for m in body.messages]

    for provider_name in [route.primary] + route.fallback:
        attempt += 1
        prov = providers.get(provider_name)
        guard = guards.get(provider_name)
        async with guard:
            try:
                resp = await prov.chat(body.model, normalized_messages, temperature=body.temperature, max_tokens=body.max_tokens)
                latency = int((time.perf_counter() - start) * 1000)
                usage_prompt = resp.usage_prompt_tokens or 0
                usage_completion = resp.usage_completion_tokens or 0
                await metrics.write({
                    "ts": time.time(),
                    "task": task,
                    "provider": provider_name,
                    "model": prov.model,
                    "latency_ms": latency,
                    "ok": True,
                    "status": resp.status_code,
                    "retries": attempt - 1,
                    "usage_prompt": usage_prompt,
                    "usage_completion": usage_completion
                })
                return JSONResponse(chat_response_from_provider(resp))
            except Exception as e:
                last_err = str(e)
                await metrics.write({
                    "ts": time.time(),
                    "task": task,
                    "provider": provider_name,
                    "model": prov.model,
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                    "ok": False,
                    "status": 0,
                    "error": last_err,
                    "retries": attempt - 1
                })
                await asyncio.sleep(min(0.25 * attempt, 2.0))  # simple backoff

    return JSONResponse({"error": {"message": last_err or "all providers failed"}}, status_code=502)
