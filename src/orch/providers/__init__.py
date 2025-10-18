"""Provider implementations and registry."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, List

import httpx

from ..router import ProviderDef
from ..types import ProviderChatResponse


@dataclass(slots=True)
class ProviderStreamChunk:
    event_type: str
    index: int | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    error: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


class BaseProvider:
    _RESERVED_OPTION_KEYS: frozenset[str] = frozenset(
        {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "tools",
            "tool_choice",
            "function_call",
            "stream",
        }
    )

    def __init__(self, defn: ProviderDef):
        self.defn = defn
        self.model = defn.model

    async def chat(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature=0.2,
        max_tokens=2048,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        function_call: dict[str, Any] | str | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        response_format: dict[str, Any] | None = None,
        **extra_options: Any,
    ) -> ProviderChatResponse:
        raise NotImplementedError

    @classmethod
    def _merge_extra_options(
        cls, payload: dict[str, Any], extra_options: dict[str, Any] | None
    ) -> None:
        if not extra_options:
            return
        for key, value in extra_options.items():
            if key in cls._RESERVED_OPTION_KEYS:
                continue
            if value is None:
                continue
            payload[key] = value


class OpenAICompatProvider(BaseProvider):
    async def chat(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature=0.2,
        max_tokens=2048,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        function_call: dict[str, Any] | str | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        response_format: dict[str, Any] | None = None,
        **extra_options: Any,
    ) -> ProviderChatResponse:
        raw_base = self.defn.base_url.strip()
        parsed = urlparse(raw_base)
        path = parsed.path or ""
        normalized_path = path.rstrip("/")
        path_segments = [segment for segment in normalized_path.split("/") if segment]
        lowered_segments = [segment.lower() for segment in path_segments]
        has_chat_completions_suffix = bool(
            len(lowered_segments) >= 2
            and lowered_segments[-2:] == ["chat", "completions"]
        )
        has_chat_suffix = bool(
            not has_chat_completions_suffix
            and lowered_segments
            and lowered_segments[-1] == "chat"
        )
        if has_chat_completions_suffix:
            segments_for_evaluation = path_segments[:-2]
            preserved_tail = path_segments[-2:]
            suffix_segments = []
        elif has_chat_suffix:
            segments_for_evaluation = path_segments[:-1]
            preserved_tail = path_segments[-1:]
            suffix_segments = ["completions"]
        else:
            segments_for_evaluation = path_segments
            preserved_tail = []
            suffix_segments = ["chat", "completions"]
        hostname = (parsed.hostname or "").lower()
        azure_compat_suffixes = (
            "openai.azure.com",
            "openai.azure.us",
            "openai.azure.cn",
            "cognitiveservices.azure.com",
            "cognitiveservices.azure.us",
            "cognitiveservices.azure.cn",
        )

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

        has_openai_segment = any(
            segment.lower() == "openai" for segment in segments_for_evaluation
        )
        last_segment = segments_for_evaluation[-1] if segments_for_evaluation else None
        openai_is_last_segment = bool(
            last_segment is not None and last_segment.lower() == "openai"
        )

        should_append_v1 = True

        if not segments_for_evaluation:
            should_append_v1 = is_openai_host
        elif has_openai_segment and not openai_is_last_segment:
            should_append_v1 = False
        elif segments_for_evaluation and is_version_segment(segments_for_evaluation[-1]):
            should_append_v1 = False

        normalized_segments = list(segments_for_evaluation)
        if should_append_v1:
            normalized_segments.append("v1")
        normalized_segments.extend(preserved_tail)
        normalized_segments.extend(suffix_segments)

        normalized_path = "/".join(normalized_segments)
        normalized_parsed = parsed._replace(path=f"/{normalized_path}" if normalized_path else "")
        normalized_base = urlunparse(normalized_parsed)
        url = f"{normalized_base.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.defn.auth_key:
            headers["Authorization"] = f"Bearer {self.defn.auth_key}"
        if self.defn.auth_header:
            headers[self.defn.auth_header] = self.defn.auth_key or ""

        payload: dict[str, Any] = {
            "model": self.defn.model or model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if function_call is not None:
            payload["function_call"] = function_call
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias
        if response_format is not None:
            payload["response_format"] = response_format
        self._merge_extra_options(payload, extra_options)
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        raw_choices = data.get("choices") or []
        normalized_choices: list[dict[str, Any]] = []
        for index, raw_choice in enumerate(raw_choices):
            if isinstance(raw_choice, dict):
                normalized_choice = dict(raw_choice)
                normalized_choice["index"] = raw_choice.get("index", index)
                raw_message = raw_choice.get("message")
                if isinstance(raw_message, dict):
                    normalized_message = {
                        key: value for key, value in raw_message.items() if value is not None
                    }
                elif raw_message is None:
                    normalized_message = None
                else:
                    normalized_message = {"content": raw_message}
                if normalized_message is not None:
                    normalized_choice["message"] = normalized_message
                elif "message" in normalized_choice:
                    normalized_choice["message"] = None
                normalized_choices.append(normalized_choice)
            else:
                normalized_choices.append({"index": index, "message": raw_choice})
        first_choice: dict[str, Any] = normalized_choices[0] if normalized_choices else {}
        first_message_raw = first_choice.get("message") if isinstance(first_choice, dict) else None
        first_message = first_message_raw if isinstance(first_message_raw, dict) else {}
        content = first_message.get("content")
        finish_reason = first_choice.get("finish_reason")
        tool_calls = first_message.get("tool_calls")
        function_call = first_message.get("function_call")
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
            choices=normalized_choices or None,
        )


class OllamaProvider(BaseProvider):
    async def chat(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature=0.2,
        max_tokens=2048,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        function_call: dict[str, Any] | str | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        response_format: dict[str, Any] | None = None,
        **extra_options: Any,
    ) -> ProviderChatResponse:
        url = f"{self.defn.base_url.rstrip('/')}/api/chat"
        _ = tools
        _ = tool_choice
        _ = function_call
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        payload = {
            "model": self.defn.model or model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            if not isinstance(response_format, dict):
                raise ValueError(
                    "OllamaProvider requires response_format to be a dictionary."
                )
            format_type = response_format.get("type")
            if format_type == "json_object":
                payload["format"] = "json"
            else:
                raise ValueError(
                    "OllamaProvider only supports response_format type 'json_object'."
                )
        cleaned_options: dict[str, Any] = {
            key: value
            for key, value in extra_options.items()
            if key not in self._RESERVED_OPTION_KEYS and value is not None
        }
        if top_p is not None:
            options["top_p"] = top_p
            cleaned_options.pop("top_p", None)
        if cleaned_options:
            options.update(cleaned_options)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
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
    async def chat(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature=0.2,
        max_tokens=2048,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        function_call: dict[str, Any] | str | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        response_format: dict[str, Any] | None = None,
        **extra_options: Any,
    ) -> ProviderChatResponse:
        _ = tools
        _ = tool_choice
        _ = function_call
        _ = extra_options
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "ping")
        return ProviderChatResponse(
            status_code=200,
            model="dummy",
            content=f"dummy:{last_user}",
            finish_reason="stop",
        )


from .anthropic import AnthropicProvider, UnsupportedContentBlockError


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

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "DummyProvider",
    "OpenAICompatProvider",
    "ProviderRegistry",
    "ProviderStreamChunk",
    "UnsupportedContentBlockError",
]
