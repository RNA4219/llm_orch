import os
import re
from urllib.parse import urlparse, urlunparse
from typing import Dict, Any, List

import httpx

from .router import ProviderDef
from .types import ProviderChatResponse

class BaseProvider:
    def __init__(self, defn: ProviderDef):
        self.defn = defn
        self.model = defn.model

    async def chat(self, model: str, messages: List[dict[str, Any]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        raise NotImplementedError

class OpenAICompatProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, Any]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        raw_base = self.defn.base_url.strip()
        parsed = urlparse(raw_base)
        path = parsed.path or ""
        normalized_path = path.rstrip("/")
        path_segments = [segment for segment in normalized_path.split("/") if segment]
        hostname = (parsed.hostname or "").lower()
        azure_compat_suffixes = ("openai.azure.com", "cognitiveservices.azure.com")

        def _matches_suffix(host: str, suffix: str) -> bool:
            return host == suffix or host.endswith(f".{suffix}")

        is_openai_host = hostname.endswith("openai.com")
        is_azure_openai_host = any(_matches_suffix(hostname, suffix) for suffix in azure_compat_suffixes)

        def is_version_segment(segment: str) -> bool:
            if not segment:
                return False
            lowered = segment.lower()
            if not lowered.startswith("v"):
                return False
            suffix = lowered[1:]
            return bool(suffix) and suffix[0].isdigit()

        has_openai_segment = any(segment == "openai" for segment in path_segments)
        openai_is_last_segment = bool(path_segments and path_segments[-1] == "openai")

        should_append_v1 = True

        if not normalized_path:
            should_append_v1 = is_openai_host
        elif has_openai_segment and not openai_is_last_segment:
            should_append_v1 = False
        elif path_segments and is_version_segment(path_segments[-1]):
            should_append_v1 = False

        normalized_segments = list(path_segments)
        if should_append_v1:
            normalized_segments.append("v1")
        normalized_segments.extend(["chat", "completions"])
        new_path = "/" + "/".join(normalized_segments)
        rebuilt = parsed._replace(path=new_path)
        url = urlunparse(rebuilt)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_env = self.defn.auth_env
        if auth_env:
            key = os.environ.get(auth_env, "")
            if key:
                if is_azure_openai_host:
                    headers["api-key"] = key
                else:
                    headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.defn.model or model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
        tool_calls = message.get("tool_calls")
        usage = data.get("usage") or {}
        response_model = data.get("model") or self.defn.model or model
        return ProviderChatResponse(
            status_code=r.status_code,
            model=response_model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_completion_tokens=usage.get("completion_tokens", 0),
        )

class AnthropicProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, Any]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        base = self.defn.base_url.strip()
        parsed = urlparse(base)
        path = parsed.path or ""
        path_segments = [segment for segment in path.split("/") if segment]

        def is_version_segment(segment: str) -> bool:
            if not segment:
                return False
            lowered = segment.lower()
            if not lowered.startswith("v"):
                return False
            suffix = lowered[1:]
            return bool(suffix) and suffix[0].isdigit()

        normalized_segments = list(path_segments)
        has_version_segment = any(is_version_segment(segment) for segment in normalized_segments)

        if normalized_segments:
            if normalized_segments[-1].lower() != "messages":
                if not has_version_segment:
                    normalized_segments.append("v1")
                normalized_segments.append("messages")
        else:
            normalized_segments = ["v1", "messages"]

        normalized_path = "/" + "/".join(normalized_segments)
        url = urlunparse(parsed._replace(path=normalized_path))
        headers: dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        auth_env = self.defn.auth_env
        if auth_env:
            raw_key = os.environ.get(auth_env)
            if raw_key:
                key = raw_key.strip()
                if key:
                    headers["x-api-key"] = key
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        mapped: list[dict[str, Any]] = []
        for message in messages:
            if message["role"] not in ("user", "assistant"):
                continue
            mapped.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )
        payload: dict[str, Any] = {
            "model": self.defn.model or model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": mapped,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content_blocks = data.get("content") or []
        content = "".join(
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        response_model = data.get("model") or self.defn.model or model
        return ProviderChatResponse(
            status_code=r.status_code,
            model=response_model,
            content=content,
            usage_prompt_tokens=usage.get("input_tokens", 0),
            usage_completion_tokens=usage.get("output_tokens", 0),
        )

class OllamaProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, Any]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        url = f"{self.defn.base_url.rstrip('/')}/api/chat"
        payload = {"model": self.defn.model or model, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        # Ollama returns {"message":{"content":...}, "done":true, ...}
        content = (data.get("message") or {}).get("content", "")
        return ProviderChatResponse(status_code=r.status_code, model=self.defn.model or model, content=content)

class DummyProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, Any]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        # simple echo-ish behavior for tests
        last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "ping")
        return ProviderChatResponse(status_code=200, model="dummy", content=f"dummy:{last_user}")

class ProviderRegistry:
    _PROVIDER_FACTORIES: dict[str, type[BaseProvider]] = {
        "openai": OpenAICompatProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "dummy": DummyProvider,
    }

    def __init__(self, providers: Dict[str, ProviderDef]):
        self.providers = {}
        for name, d in providers.items():
            provider_type_raw = d.type
            provider_type = provider_type_raw if provider_type_raw is not None else "openai"

            if isinstance(provider_type_raw, str) and not provider_type_raw.strip():
                raise ValueError(
                    f"Unknown provider type '<missing>' for provider '{name}'"
                )

            factory = self._PROVIDER_FACTORIES.get(provider_type)
            if factory is None:
                display_type: str
                if isinstance(provider_type_raw, str):
                    stripped = provider_type_raw.strip()
                    display_type = stripped if stripped else "<missing>"
                else:
                    display_type = provider_type
                raise ValueError(
                    f"Unknown provider type '{display_type}' for provider '{name}'"
                )
            self.providers[name] = factory(d)

    def get(self, name: str) -> BaseProvider:
        return self.providers[name]
