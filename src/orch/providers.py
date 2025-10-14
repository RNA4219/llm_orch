import os
from typing import Dict, Any, List, Iterable
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
        base = self.defn.base_url.rstrip('/')
        url = f"{base}/chat/completions" if "/openai/" in base else f"{base}/v1/chat/completions"
        key = os.environ.get(self.defn.auth_env or "", "")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": self.defn.model or model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return ProviderChatResponse(status_code=r.status_code, model=data.get("model", self.defn.model), content=content,
                                    usage_prompt_tokens=usage.get("prompt_tokens", 0),
                                    usage_completion_tokens=usage.get("completion_tokens", 0))

class AnthropicProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, str]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        url = f"{self.defn.base_url.rstrip('/')}/v1/messages"
        key = os.environ.get(self.defn.auth_env or "", "")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

        def to_text_blocks(content: Any) -> List[Dict[str, str]]:
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            if isinstance(content, Iterable) and not isinstance(content, dict):
                blocks: List[Dict[str, str]] = []
                for item in content:
                    if isinstance(item, dict):
                        text_val = item.get("text")
                        if text_val is None:
                            continue
                        blocks.append({"type": item.get("type", "text"), "text": str(text_val)})
                    else:
                        blocks.append({"type": "text", "text": str(item)})
                return blocks or [{"type": "text", "text": ""}]
            return [{"type": "text", "text": str(content)}]

        def flatten_to_text(content: Any) -> str:
            blocks = to_text_blocks(content)
            texts = [block.get("text", "") for block in blocks if block.get("text")]
            return "\n\n".join(texts)

        system_segments = [m["content"] for m in messages if m.get("role") == "system"]
        mapped: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            mapped.append({"role": role, "content": to_text_blocks(msg.get("content", ""))})

        payload: Dict[str, Any] = {
            "model": self.defn.model or model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": mapped,
        }
        if system_segments:
            flattened = [flatten_to_text(seg) for seg in system_segments]
            payload["system"] = "\n\n".join(filter(None, flattened))
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content = "".join(block.get("text", "") for block in (data.get("content") or []) if isinstance(block, dict))
        usage = data.get("usage") or {}
        return ProviderChatResponse(status_code=r.status_code, model=data.get("model", self.defn.model), content=content,
                                    usage_prompt_tokens=usage.get("input_tokens", 0),
                                    usage_completion_tokens=usage.get("output_tokens", 0))

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
