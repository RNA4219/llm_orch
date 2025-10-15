import os
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List

import httpx

from .router import ProviderDef
from .types import ProviderChatResponse

class BaseProvider:
    def __init__(self, defn: ProviderDef):
        self.defn = defn
        self.model = defn.model

    async def chat(self, model: str, messages: List[dict[str, str]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        raise NotImplementedError

class OpenAICompatProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, str]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        base = self.defn.base_url.rstrip("/")
        parsed = urlparse(base)
        path = parsed.path or ""
        normalized_path = path.rstrip("/")
        path_segments = [segment for segment in normalized_path.split("/") if segment]
        hostname = (parsed.netloc or "").lower()
        is_openai_host = hostname.endswith("openai.com")

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

        base_for_join = f"{base}/v1" if should_append_v1 else base

        url = urljoin(f"{base_for_join.rstrip('/')}/", "chat/completions")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_env = self.defn.auth_env
        if auth_env:
            key = os.environ.get(auth_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        payload = {"model": self.defn.model or model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        response_model = data.get("model") or self.defn.model or model
        return ProviderChatResponse(
            status_code=r.status_code,
            model=response_model,
            content=content,
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_completion_tokens=usage.get("completion_tokens", 0),
        )

class AnthropicProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, str]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        url = f"{self.defn.base_url.rstrip('/')}/v1/messages"
        headers: dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        auth_env = self.defn.auth_env
        if auth_env:
            key = os.environ.get(auth_env, "")
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
    async def chat(self, model: str, messages: List[dict[str, str]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
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
    async def chat(self, model: str, messages: List[dict[str, str]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        # simple echo-ish behavior for tests
        last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "ping")
        return ProviderChatResponse(status_code=200, model="dummy", content=f"dummy:{last_user}")

class ProviderRegistry:
    def __init__(self, providers: Dict[str, ProviderDef]):
        self.providers = {}
        for name, d in providers.items():
            if d.type == "openai":
                self.providers[name] = OpenAICompatProvider(d)
            elif d.type == "anthropic":
                self.providers[name] = AnthropicProvider(d)
            elif d.type == "ollama":
                self.providers[name] = OllamaProvider(d)
            elif d.type == "dummy":
                self.providers[name] = DummyProvider(d)
            else:
                self.providers[name] = OpenAICompatProvider(d)  # default

    def get(self, name: str) -> BaseProvider:
        return self.providers[name]
