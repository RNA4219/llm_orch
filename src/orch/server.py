import asyncio
from collections.abc import Mapping
import os
import time

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .metrics import MetricsLogger
from .providers import ProviderRegistry
from .rate_limiter import ProviderGuards
from .router import LoadedConfig, RoutePlanner, load_config
from .types import ChatRequest, chat_response_from_provider

app = FastAPI(title="llm-orch")

CONFIG_DIR = os.environ.get(
    "ORCH_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config"),
)
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "metrics")

cfg: LoadedConfig
providers: ProviderRegistry
guards: ProviderGuards
planner: RoutePlanner
metrics: MetricsLogger


def resolve_dummy_flag_from_env(env: Mapping[str, str] | None = None) -> bool:
    source = env or os.environ
    raw_value = source.get("ORCH_USE_DUMMY", "0")
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def init_dependencies(*, use_dummy: bool) -> None:
    global cfg, providers, guards, planner, metrics

    cfg = load_config(CONFIG_DIR, use_dummy=use_dummy)
    providers = ProviderRegistry(cfg.providers)
    guards = ProviderGuards(cfg.providers)
    planner = RoutePlanner(cfg.router, cfg.providers)
    metrics = MetricsLogger(METRICS_DIR)


@app.on_event("startup")
async def _startup_init_dependencies() -> None:
    init_dependencies(use_dummy=resolve_dummy_flag_from_env())


init_dependencies(use_dummy=resolve_dummy_flag_from_env())

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

    payload_messages = [message.model_dump() for message in body.messages]

    for provider_name in [route.primary] + route.fallback:
        attempt += 1
        prov = providers.get(provider_name)
        guard = guards.get(provider_name)
        async with guard:
            try:
                resp = await prov.chat(
                    body.model,
                    payload_messages,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                )
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
