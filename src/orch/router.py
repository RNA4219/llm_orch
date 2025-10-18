import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised via tests
    import tomli as tomllib

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, ValidationError, model_validator

@dataclass
class ProviderDef:
    name: str
    type: str
    base_url: str
    model: str
    auth_env: str | None
    rpm: int
    concurrency: int
    tpm: int | None = None

@dataclass
class RouterDefaults:
    temperature: float
    max_tokens: int
    task_header: str
    task_header_value: str | None = None

@dataclass(frozen=True)
class CircuitBreakerSettings:
    failure_threshold: int
    recovery_time_s: float


@dataclass
class RouteTarget:
    provider: str
    weight: int | None = None
    circuit_breaker: CircuitBreakerSettings | None = None


@dataclass
class RouteDef:
    name: str
    strategy: Literal["priority", "weighted", "sticky"]
    targets: list[RouteTarget]
    sticky_ttl: float | None = None
    _order: tuple[str, ...] = field(default_factory=tuple, repr=False)

    @property
    def primary(self) -> str:
        if self._order:
            return self._order[0]
        return self.targets[0].provider

    @property
    def fallback(self) -> list[str]:
        order = self._order or tuple(target.provider for target in self.targets)
        return list(order[1:])

    def ordered(self, providers: Sequence[str]) -> "RouteDef":
        return RouteDef(
            name=self.name,
            strategy=self.strategy,
            targets=self.targets,
            sticky_ttl=self.sticky_ttl,
            _order=tuple(providers),
        )

@dataclass
class RouterConfig:
    defaults: RouterDefaults
    routes: Dict[str, RouteDef]

@dataclass
class LoadedConfig:
    providers: Dict[str, ProviderDef]
    router: RouterConfig
    mtimes: dict[str, float] = field(default_factory=dict)
    watch_paths: tuple[str, ...] = field(default_factory=tuple)


class _CircuitBreakerModel(BaseModel):
    failure_threshold: PositiveInt = Field(default=3)
    recovery_time_s: PositiveFloat = Field(default=30.0)

    model_config = ConfigDict(extra="forbid")


class _TargetModel(BaseModel):
    provider: str
    weight: PositiveInt | None = None
    circuit_breaker: _CircuitBreakerModel | None = None

    model_config = ConfigDict(extra="forbid")


class _RouteModel(BaseModel):
    strategy: Literal["priority", "weighted", "sticky"] | None = None
    primary: str | None = None
    fallback: list[str] = Field(default_factory=list)
    targets: list[_TargetModel] = Field(default_factory=list)
    sticky_ttl: PositiveFloat | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise TypeError("route definition must be a mapping")
        normalized = dict(data)
        fallback = normalized.get("fallback")
        if fallback is None:
            normalized["fallback"] = []
        elif isinstance(fallback, list):
            normalized["fallback"] = [str(item) for item in fallback]
        else:
            normalized["fallback"] = [str(fallback)]
        return normalized

    @model_validator(mode="after")
    def _finalize(self) -> "_RouteModel":
        if not self.targets:
            providers: list[str] = []
            if self.primary:
                providers.append(self.primary)
            providers.extend(self.fallback)
            if not providers:
                raise ValueError("route must specify at least one provider")
            self.targets = [_TargetModel(provider=name) for name in providers]
        if self.strategy is None:
            if self.sticky_ttl is not None:
                self.strategy = "sticky"
            elif any(target.weight is not None for target in self.targets):
                self.strategy = "weighted"
            else:
                self.strategy = "priority"
        if self.strategy == "weighted":
            missing = [t.provider for t in self.targets if t.weight is None]
            if missing:
                raise ValueError(
                    "weighted strategy requires weights for all targets; missing: {targets}".format(
                        targets=", ".join(missing)
                    )
                )
        if self.strategy == "sticky" and self.sticky_ttl is None:
            raise ValueError("sticky strategy requires 'sticky_ttl'")
        return self


class _DefaultsModel(BaseModel):
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=2048, ge=1)
    task_header: str = Field(default="x-orch-task-kind")
    task_header_value: str | None = None

    model_config = ConfigDict(extra="forbid")


class _RouterModel(BaseModel):
    defaults: _DefaultsModel = Field(default_factory=_DefaultsModel)
    routes: Dict[str, _RouteModel]

    model_config = ConfigDict(extra="forbid")


def _read_concurrency(name: str, raw_value: object) -> int:
    concurrency = int(raw_value)
    if concurrency < 1:
        raise ValueError(
            "Provider '{name}' defines invalid concurrency {value}; must be >= 1.".format(
                name=name,
                value=concurrency,
            )
        )
    return concurrency


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
            concurrency=_read_concurrency(name, d.get("concurrency", 4)),
            tpm=int(d["tpm"]) if d.get("tpm") is not None else None,
        )
    router_path = os.path.join(config_dir, "router.yaml")
    with open(router_path, "r", encoding="utf-8") as f:
        rdata = yaml.safe_load(f) or {}
    try:
        parsed = _RouterModel.model_validate(rdata)
    except ValidationError as exc:  # pragma: no cover - exercised via explicit tests
        problems = []
        for error in exc.errors():
            location = " -> ".join(str(item) for item in error.get("loc", ())) or "<root>"
            problems.append(f"{location}: {error.get('msg', 'invalid value')}")
        raise ValueError("; ".join(problems)) from exc
    defs = parsed.defaults
    routes_cfg: Dict[str, RouteDef] = {}
    for name, route_model in parsed.routes.items():
        targets = [
            RouteTarget(
                provider=target.provider,
                weight=int(target.weight) if target.weight is not None else None,
                circuit_breaker=(
                    CircuitBreakerSettings(
                        failure_threshold=int(target.circuit_breaker.failure_threshold),
                        recovery_time_s=float(target.circuit_breaker.recovery_time_s),
                    )
                    if target.circuit_breaker
                    else None
                ),
            )
            for target in route_model.targets
        ]
        routes_cfg[name] = RouteDef(
            name=name,
            strategy=route_model.strategy,
            targets=targets,
            sticky_ttl=float(route_model.sticky_ttl) if route_model.sticky_ttl is not None else None,
        )
    router = RouterConfig(
        defaults=RouterDefaults(
            temperature=float(defs.temperature),
            max_tokens=int(defs.max_tokens),
            task_header=str(defs.task_header),
            task_header_value=str(defs.task_header_value)
            if defs.task_header_value is not None
            else None,
        ),
        routes=routes_cfg,
    )
    validate_router_config(router, providers)
    mtimes = {
        "providers": os.stat(prov_path).st_mtime,
        "router": os.stat(router_path).st_mtime,
    }
    return LoadedConfig(
        providers=providers,
        router=router,
        mtimes=mtimes,
        watch_paths=(prov_path, router_path),
    )


def validate_router_config(router: RouterConfig, providers: Dict[str, ProviderDef]) -> None:
    for route_name, route in router.routes.items():
        if not route.targets:
            raise ValueError(f"Route '{route_name}' must specify at least one provider")
        for target in route.targets:
            provider_name = target.provider
            if provider_name not in providers:
                available = ", ".join(sorted(providers)) or "<none>"
                raise ValueError(
                    "Route '{route}' references undefined provider '{provider}'. Available providers: {available}".format(
                        route=route_name,
                        provider=provider_name,
                        available=available,
                    )
                )

class RoutePlanner:
    def __init__(
        self,
        cfg: RouterConfig,
        providers: Dict[str, ProviderDef],
        *,
        config_dir: str | None = None,
        use_dummy: bool = False,
        mtimes: dict[str, float] | None = None,
    ):
        self.cfg = cfg
        self.providers = providers
        self._config_dir = config_dir
        self._use_dummy = use_dummy
        self._mtimes = dict(mtimes or {})
        self._sticky_assignments: dict[str, dict[str, tuple[str, float]]] = {}
        self._circuit_states: dict[str, _CircuitBreakerState] = {}
        self._rebuild_states()

    def _rebuild_states(self) -> None:
        self._sticky_assignments.clear()
        circuit_states: dict[str, _CircuitBreakerState] = {}
        for route in self.cfg.routes.values():
            for target in route.targets:
                settings = target.circuit_breaker
                if settings is None:
                    continue
                existing = circuit_states.get(target.provider)
                if existing is None or existing.settings != settings:
                    circuit_states[target.provider] = _CircuitBreakerState(settings)
        self._circuit_states = circuit_states

    def refresh(self) -> bool:
        if self._config_dir is None:
            return False
        prov_filename = "providers.dummy.toml" if self._use_dummy else "providers.toml"
        prov_path = os.path.join(self._config_dir, prov_filename)
        router_path = os.path.join(self._config_dir, "router.yaml")
        try:
            providers_mtime = os.stat(prov_path).st_mtime
            router_mtime = os.stat(router_path).st_mtime
        except FileNotFoundError:
            return False
        if (
            providers_mtime == self._mtimes.get("providers")
            and router_mtime == self._mtimes.get("router")
        ):
            return False
        loaded = load_config(self._config_dir, use_dummy=self._use_dummy)
        self.providers = loaded.providers
        self.cfg = loaded.router
        self._mtimes = dict(loaded.mtimes)
        self._rebuild_states()
        return True

    def record_failure(self, provider: str, *, now: float | None = None) -> None:
        state = self._circuit_states.get(provider)
        if state is None:
            return
        state.record_failure(time.monotonic() if now is None else now)

    def record_success(self, provider: str) -> None:
        state = self._circuit_states.get(provider)
        if state is None:
            return
        state.record_success()

    def plan(
        self,
        task: str,
        *,
        sticky_key: str | None = None,
        now: float | None = None,
        rand: random.Random | None = None,
    ) -> RouteDef:
        route = self.cfg.routes.get(task)
        if route is None:
            route = self.cfg.routes.get("DEFAULT")
        if route is None:
            raise ValueError(
                f"no route configured for task '{task}' and no DEFAULT route defined in router configuration."
            )
        current_time = time.monotonic() if now is None else now
        order = self._resolve_order(route, sticky_key=sticky_key, now=current_time, rand=rand)
        return route.ordered(order)

    def _resolve_order(
        self,
        route: RouteDef,
        *,
        sticky_key: str | None,
        now: float,
        rand: random.Random | None,
    ) -> list[str]:
        if not route.targets:
            raise ValueError(f"Route '{route.name}' defines no targets")
        active: list[RouteTarget] = []
        blocked: list[RouteTarget] = []
        for target in route.targets:
            state = self._circuit_states.get(target.provider)
            if state is not None and state.is_open(now):
                blocked.append(target)
            else:
                active.append(target)
        candidates = active if active else blocked or route.targets
        selected: str | None = None
        if route.strategy == "weighted":
            selected = self._choose_weighted(candidates, rand=rand)
        elif route.strategy == "sticky" and sticky_key:
            selected = self._select_sticky(route, sticky_key, candidates, now)
        else:
            selected = candidates[0].provider
        order: list[str] = [selected]
        for target in route.targets:
            if target.provider == selected:
                continue
            state = self._circuit_states.get(target.provider)
            if state is not None and state.is_open(now):
                continue
            order.append(target.provider)
        for target in route.targets:
            if target.provider in order:
                continue
            order.append(target.provider)
        if route.strategy == "sticky" and sticky_key:
            self._remember_sticky(route.name, sticky_key, selected, now, route.sticky_ttl)
        return order

    def _choose_weighted(
        self, targets: Sequence[RouteTarget], *, rand: random.Random | None
    ) -> str:
        total = 0
        weights: list[tuple[str, int]] = []
        for target in targets:
            weight = target.weight if target.weight is not None else 1
            total += weight
            weights.append((target.provider, weight))
        if total <= 0:  # pragma: no cover - defensive
            return targets[0].provider
        draw = rand.random() if rand is not None else random.random()
        threshold = draw * total
        cumulative = 0.0
        for provider, weight in weights:
            cumulative += weight
            if threshold < cumulative:
                return provider
        return weights[-1][0]

    def _select_sticky(
        self,
        route: RouteDef,
        sticky_key: str,
        candidates: Sequence[RouteTarget],
        now: float,
    ) -> str:
        self._prune_sticky(route.name, now)
        assignments = self._sticky_assignments.get(route.name)
        if assignments is not None:
            assigned = assignments.get(sticky_key)
            if assigned is not None:
                provider, expiry = assigned
                if expiry > now and any(t.provider == provider for t in candidates):
                    state = self._circuit_states.get(provider)
                    if state is None or not state.is_open(now):
                        return provider
                assignments.pop(sticky_key, None)
                if not assignments:
                    self._sticky_assignments.pop(route.name, None)
        return candidates[0].provider

    def _remember_sticky(
        self,
        route_name: str,
        sticky_key: str,
        provider: str,
        now: float,
        ttl: float | None,
    ) -> None:
        if ttl is None:
            return
        expiry = now + ttl
        assignments = self._sticky_assignments.setdefault(route_name, {})
        assignments[sticky_key] = (provider, expiry)

    def _prune_sticky(self, route_name: str, now: float) -> None:
        assignments = self._sticky_assignments.get(route_name)
        if not assignments:
            return
        expired = [key for key, (_, expiry) in assignments.items() if expiry <= now]
        for key in expired:
            assignments.pop(key, None)
        if not assignments:
            self._sticky_assignments.pop(route_name, None)


@dataclass
class _CircuitBreakerState:
    settings: CircuitBreakerSettings
    failure_count: int = 0
    opened_until: float | None = None

    def is_open(self, now: float) -> bool:
        if self.opened_until is None:
            return False
        if now >= self.opened_until:
            self.opened_until = None
            self.failure_count = 0
            return False
        return True

    def record_failure(self, now: float) -> None:
        if self.is_open(now):
            return
        self.failure_count += 1
        if self.failure_count >= self.settings.failure_threshold:
            self.opened_until = now + self.settings.recovery_time_s
            self.failure_count = 0

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_until = None
