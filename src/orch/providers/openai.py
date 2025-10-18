from __future__ import annotations

import json
import math
import os
from collections.abc import AsyncIterator
from typing import Any, List
from urllib.parse import urlparse, urlunparse

import httpx

from ..types import (
    ProviderChatResponse,
    ProviderStreamChoice,
    ProviderStreamChunk as ProviderStreamChunkModel,
)
from . import BaseProvider


class OpenAICompatProvider(BaseProvider):
    def _build_chat_request(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | str | None,
        function_call: dict[str, Any] | str | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        logit_bias: dict[str, float] | None,
        response_format: dict[str, Any] | None,
        extra_options: dict[str, Any],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
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
            "stream": stream,
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
        return url, headers, payload

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
        url, headers, payload = self._build_chat_request(
            model,
            messages,
            temperature,
            max_tokens,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            function_call=function_call,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            extra_options=dict(extra_options),
        )
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

    @staticmethod
    def _parse_retry_after(value: str | None) -> int | None:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            return None
        if seconds < 0:
            return None
        return int(math.ceil(seconds))

    @staticmethod
    def _normalize_stream_choice(
        raw_choice: dict[str, Any],
        index_fallback: int,
    ) -> tuple[ProviderStreamChoice, dict[str, Any] | str | None, str | None]:
        index_value = raw_choice.get("index")
        index = index_value if isinstance(index_value, int) else index_fallback
        delta_field = raw_choice.get("delta")
        role: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        function_call: dict[str, Any] | None = None
        normalized_delta: dict[str, Any] | str | None
        if isinstance(delta_field, dict):
            normalized_delta = {
                key: value for key, value in delta_field.items() if value is not None
            }
            role_candidate = normalized_delta.get("role")
            if isinstance(role_candidate, str) and role_candidate:
                role = role_candidate
            raw_tool_calls = normalized_delta.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                tool_calls = raw_tool_calls
            raw_function_call = normalized_delta.get("function_call")
            if isinstance(raw_function_call, dict):
                function_call = raw_function_call
        elif isinstance(delta_field, str):
            normalized_delta = delta_field
        else:
            normalized_delta = None
        message_field = raw_choice.get("message")
        if isinstance(message_field, dict):
            message_payload = {
                key: value for key, value in message_field.items() if value is not None
            }
        else:
            message_payload = None
        finish_reason = raw_choice.get("finish_reason")
        if not isinstance(finish_reason, str):
            finish_reason = None
        choice_model = ProviderStreamChoice(
            index=index,
            delta=normalized_delta,
            role=role,
            content=None,
            tool_calls=tool_calls,
            function_call=function_call,
            finish_reason=finish_reason,
            message=message_payload,
        )
        return choice_model, normalized_delta, finish_reason

    def _map_stream_payload(
        self,
        payload: dict[str, Any],
        *,
        event_name: str | None,
        retry_after: int | None,
    ) -> ProviderStreamChunkModel | None:
        choices_payload = payload.get("choices")
        usage_payload = payload.get("usage")
        error_payload = payload.get("error")
        normalized_choices: list[ProviderStreamChoice] = []
        chunk_index: int | None = None
        chunk_delta: dict[str, Any] | str | None = None
        chunk_finish: str | None = None
        if isinstance(choices_payload, list):
            for position, raw_choice in enumerate(choices_payload):
                if not isinstance(raw_choice, dict):
                    continue
                choice_model, choice_delta, finish = self._normalize_stream_choice(
                    raw_choice, position
                )
                normalized_choices.append(choice_model)
                if chunk_index is None:
                    chunk_index = choice_model.index
                if chunk_delta is None and choice_delta is not None:
                    chunk_delta = choice_delta
                if chunk_finish is None and finish is not None:
                    chunk_finish = finish
        usage: dict[str, int] | None = None
        if isinstance(usage_payload, dict):
            prompt_tokens = usage_payload.get("prompt_tokens")
            completion_tokens = usage_payload.get("completion_tokens")
            usage_values: dict[str, int] = {}
            if isinstance(prompt_tokens, int):
                usage_values["prompt_tokens"] = prompt_tokens
            if isinstance(completion_tokens, int):
                usage_values["completion_tokens"] = completion_tokens
            if usage_values:
                usage = usage_values
        error: dict[str, Any] | None = None
        if isinstance(error_payload, dict):
            error = {key: value for key, value in error_payload.items() if value is not None}
            if retry_after is not None and "retry_after" not in error:
                error["retry_after"] = retry_after
        event_type: str | None
        if normalized_choices:
            event_type = event_name or "chunk"
        elif usage is not None:
            event_type = event_name or "usage"
        elif error is not None:
            event_type = event_name or "error"
        else:
            event_type = event_name
        if not normalized_choices and usage is None and error is None:
            return None
        return ProviderStreamChunkModel(
            event_type=event_type,
            index=chunk_index,
            delta=chunk_delta,
            finish_reason=chunk_finish,
            choices=normalized_choices,
            usage=usage,
            error=error,
            raw=payload,
        )

    async def chat_stream(
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
    ) -> AsyncIterator[ProviderStreamChunkModel]:
        url, headers, payload = self._build_chat_request(
            model,
            messages,
            temperature,
            max_tokens,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            function_call=function_call,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            extra_options=dict(extra_options),
        )
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                retry_after = self._parse_retry_after(response.headers.get("retry-after"))
                data_lines: list[str] = []
                event_name: str | None = None
                async for raw_line in response.aiter_lines():
                    if raw_line is None:
                        continue
                    line = raw_line.strip("\r")
                    if line == "":
                        if not data_lines:
                            event_name = None
                            continue
                        data_text = "\n".join(data_lines)
                        data_lines.clear()
                        if data_text == "[DONE]":
                            break
                        try:
                            payload_data = json.loads(data_text)
                        except json.JSONDecodeError:
                            event_name = None
                            continue
                        if not isinstance(payload_data, dict):
                            event_name = None
                            continue
                        chunk = self._map_stream_payload(
                            payload_data,
                            event_name=event_name,
                            retry_after=retry_after,
                        )
                        event_name = None
                        if chunk is not None:
                            yield chunk
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_value = line[6:].strip()
                        event_name = event_value or None
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        continue
                if data_lines:
                    data_text = "\n".join(data_lines)
                    if data_text and data_text != "[DONE]":
                        try:
                            payload_data = json.loads(data_text)
                        except json.JSONDecodeError:
                            return
                        if isinstance(payload_data, dict):
                            chunk = self._map_stream_payload(
                                payload_data,
                                event_name=event_name,
                                retry_after=retry_after,
                            )
                            if chunk is not None:
                                yield chunk
