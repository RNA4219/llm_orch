import os
from dataclasses import dataclass
from typing import Dict, Any
import tomllib
import yaml

@dataclass
class ProviderDef:
    name: str
    type: str
    base_url: str
    model: str
    auth_env: str | None
    rpm: int
    concurrency: int

@dataclass
class RouterDefaults:
    temperature: float
    max_tokens: int
    task_header: str
    task_header_value: str | None = None

@dataclass
class RouteDef:
    primary: str
    fallback: list[str]

@dataclass
class RouterConfig:
    defaults: RouterDefaults
    routes: Dict[str, RouteDef]

@dataclass
class LoadedConfig:
    providers: Dict[str, ProviderDef]
    router: RouterConfig

def load_config(config_dir: str, use_dummy: bool=False) -> LoadedConfig:
    prov_path = os.path.join(config_dir, "providers.dummy.toml" if use_dummy else "providers.toml")
    with open(prov_path, "rb") as f:
        prov_data = tomllib.load(f)
    providers: Dict[str, ProviderDef] = {}
    for name, d in prov_data.items():
        providers[name] = ProviderDef(
            name=name,
            type=d.get("type", "openai"),
            base_url=d.get("base_url", ""),
            model=d.get("model", ""),
            auth_env=d.get("auth_env"),
            rpm=int(d.get("rpm", 60)),
            concurrency=int(d.get("concurrency", 4)),
        )
    with open(os.path.join(config_dir, "router.yaml"), "r", encoding="utf-8") as f:
        rdata = yaml.safe_load(f)
    defs = rdata.get("defaults", {})
    routes_cfg = {}
    for k, v in rdata.get("routes", {}).items():
        fallback_raw = v.get("fallback")
        if fallback_raw is None:
            fallback_list: list[str] = []
        elif isinstance(fallback_raw, list):
            fallback_list = [str(item) for item in fallback_raw]
        else:
            fallback_list = [str(fallback_raw)]
        routes_cfg[k] = RouteDef(primary=v["primary"], fallback=fallback_list)
    router = RouterConfig(
        defaults=RouterDefaults(
            temperature=float(defs.get("temperature", 0.2)),
            max_tokens=int(defs.get("max_tokens", 2048)),
            task_header=str(defs.get("task_header", "x-orch-task-kind")),
            task_header_value=None
        ),
        routes=routes_cfg
    )
    for route_name, route in routes_cfg.items():
        referenced = [(route.primary, "primary"), *[(name, "fallback") for name in route.fallback]]
        for provider_name, origin in referenced:
            if provider_name not in providers:
                raise ValueError(
                    f"Route '{route_name}' references undefined provider '{provider_name}' in {origin}"
                )
    return LoadedConfig(providers=providers, router=router)

class RoutePlanner:
    def __init__(self, cfg: RouterConfig, providers: Dict[str, ProviderDef]):
        self.cfg = cfg
        self.providers = providers

    def plan(self, task: str) -> RouteDef:
        route = self.cfg.routes.get(task)
        if route is not None:
            return route
        default_route = self.cfg.routes.get("DEFAULT")
        if default_route is not None:
            return default_route
        raise ValueError(f"no route configured for task '{task}' and no DEFAULT route")
