import json
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
        function_call = message.get("function_call")
        usage = data.get("usage") or {}
        response_model = data.get("model") or self.defn.model or model
        return ProviderChatResponse(
            status_code=r.status_code,
            model=response_model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            function_call=function_call,
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
        def normalize_text_content(raw_content: Any) -> str:
            if raw_content is None:
                return ""
            if isinstance(raw_content, str):
                return raw_content
            if isinstance(raw_content, list):
                parts: list[str] = []
                for block in raw_content:
                    if not isinstance(block, dict):
                        raise ValueError(
                            "Anthropic content lists must contain dict blocks with 'type' and 'text'."
                        )
                    block_type = block.get("type")
                    if not isinstance(block_type, str) or not block_type:
                        raise ValueError(
                            "Anthropic content blocks require a non-empty string 'type'."
                        )
                    block_text = block.get("text")
                    if not isinstance(block_text, str):
                        raise ValueError(
                            "Anthropic text-like blocks must include string 'text' values."
                        )
                    parts.append(block_text)
                return "".join(parts)
            raise ValueError("Anthropic messages must provide string or list content values.")

        def map_tool_call(tool_call: Any) -> dict[str, Any]:
            if not isinstance(tool_call, dict):
                raise ValueError("Anthropic tool calls must be dictionaries.")
            identifier = tool_call.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Anthropic tool calls require a non-empty string 'id'.")
            call_type = tool_call.get("type")
            if call_type not in (None, "function"):
                raise ValueError("Unsupported Anthropic tool call type. Only 'function' is allowed.")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("Anthropic tool calls require a 'function' definition.")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Anthropic tool call functions require a non-empty string 'name'.")
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    input_payload: Any = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError as exc:
                    raise ValueError("Anthropic tool call arguments must be valid JSON strings.") from exc
            elif isinstance(raw_arguments, dict):
                input_payload = raw_arguments
            elif raw_arguments is None:
                input_payload = {}
            else:
                raise ValueError("Anthropic tool call arguments must be provided as JSON strings or dicts.")
            return {
                "type": "tool_use",
                "id": identifier,
                "name": name,
                "input": input_payload,
            }

        def map_tool_result(message: dict[str, Any]) -> dict[str, Any]:
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("Anthropic tool messages require a 'tool_call_id'.")
            if "content" not in message:
                raise ValueError("Anthropic tool messages must include 'content'.")
            raw_content = message["content"]
            if isinstance(raw_content, list):
                normalized_blocks: list[dict[str, Any]] = []
                for block in raw_content:
                    if not isinstance(block, dict):
                        raise ValueError(
                            "Anthropic tool result content lists must contain block dictionaries."
                        )
                    block_type = block.get("type")
                    if not isinstance(block_type, str) or not block_type:
                        raise ValueError(
                            "Anthropic tool result blocks require a non-empty string 'type'."
                        )
                    normalized_blocks.append(block)
                result_content: str | list[dict[str, Any]] = normalized_blocks
            else:
                result_content = normalize_text_content(raw_content)
            return {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result_content,
            }

        system_messages: list[str] = []
        mapped: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                if "content" not in message:
                    raise ValueError("Anthropic system messages must include 'content'.")
                system_messages.append(normalize_text_content(message["content"]))
                continue
            if role == "tool":
                mapped.append(
                    {
                        "role": "user",
                        "content": [map_tool_result(message)],
                    }
                )
                continue
            if role not in ("user", "assistant"):
                continue
            content_blocks: list[dict[str, Any]] = []
            if "content" in message:
                text_content = normalize_text_content(message["content"])
                if text_content:
                    content_blocks.append({"type": "text", "text": text_content})
            tool_calls = message.get("tool_calls")
            if tool_calls:
                if not isinstance(tool_calls, list):
                    raise ValueError("Anthropic tool calls must be provided as a list.")
                for tool_call in tool_calls:
                    content_blocks.append(map_tool_call(tool_call))
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            mapped.append({"role": role, "content": content_blocks})
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
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_value = block.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
            elif block_type == "tool_use":
                identifier = block.get("id")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError("Anthropic tool_use blocks require a non-empty string 'id'.")
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Anthropic tool_use blocks require a non-empty string 'name'.")
                raw_input = block.get("input")
                if raw_input is None:
                    input_payload: dict[str, Any] = {}
                elif isinstance(raw_input, dict):
                    input_payload = raw_input
                else:
                    raise ValueError("Anthropic tool_use blocks must provide dict 'input' payloads.")
                tool_calls.append(
                    {
                        "id": identifier,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(input_payload),
                        },
                    }
                )
            elif block_type == "tool_result":
                result_content = block.get("content")
                text_parts.append(normalize_text_content(result_content))

        content = "".join(text_parts)
        finish_reason_raw = data.get("stop_reason")
        finish_reason = "tool_calls" if finish_reason_raw == "tool_use" else finish_reason_raw
        normalized_tool_calls = tool_calls or None
        usage = data.get("usage") or {}
        response_model = data.get("model") or self.defn.model or model
        return ProviderChatResponse(
            status_code=r.status_code,
            model=response_model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=normalized_tool_calls,
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
        message = data.get("message") or {}
        content = message.get("content")
        finish_reason = data.get("finish_reason") or data.get("done_reason")
        tool_calls = message.get("tool_calls")
        return ProviderChatResponse(
            status_code=r.status_code,
            model=self.defn.model or model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
        )

class DummyProvider(BaseProvider):
    async def chat(self, model: str, messages: List[dict[str, Any]], temperature=0.2, max_tokens=2048) -> ProviderChatResponse:
        # simple echo-ish behavior for tests
        last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "ping")
        return ProviderChatResponse(
            status_code=200,
            model="dummy",
            content=f"dummy:{last_user}",
            finish_reason="stop",
        )

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
