from __future__ import annotations

import os
from typing import Any, List
from urllib.parse import urlparse, urlunparse

import httpx

from ..types import ProviderChatResponse
from . import BaseProvider


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
            suffix_segments: list[str] = []
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
        is_azure_openai_host = any(
            _matches_suffix(hostname, suffix) for suffix in azure_compat_suffixes
        )

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
        payload: dict[str, Any] = {
            "model": self.defn.model or model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
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
