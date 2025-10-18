import json
import os
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse, urlunparse
from typing import Any, AsyncIterator, Dict, List

import httpx

from ..router import ProviderDef
from ..types import ProviderChatResponse

# [ ] openai移行完了



class UnsupportedContentBlockError(ValueError):
    """Raised when a request includes a content block unsupported by a provider."""


@dataclass(slots=True)
class ProviderStreamChunk:
    event_type: str
    index: int | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    error: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None

def _normalize_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    tool_type = tool.get("type")
    if tool_type is None:
        return dict(tool)
    if tool_type != "function":
        raise ValueError("Anthropic tools only support OpenAI function tool definitions.")
    function = tool.get("function")
    if not isinstance(function, dict):
        raise ValueError("Anthropic function tools require a 'function' dictionary definition.")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Anthropic tools require a non-empty function name.")
    description_field = function.get("description")
    if description_field is None:
        description_field = tool.get("description")
    if description_field is not None and not isinstance(description_field, str):
        raise ValueError("Anthropic tool descriptions must be strings when provided.")
    parameters = function.get("parameters")
    if parameters is None:
        input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    else:
        if not isinstance(parameters, dict):
            raise ValueError("Anthropic tool parameters must be provided as dictionaries.")
        input_schema = parameters
    normalized: dict[str, Any] = {"name": name, "input_schema": input_schema}
    if description_field:
        normalized["description"] = description_field
    return normalized


def _normalize_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_anthropic_tool(tool) for tool in tools]


def _normalize_anthropic_tool_choice(
    tool_choice: dict[str, Any] | str,
) -> dict[str, Any] | str:
    if isinstance(tool_choice, str):
        return tool_choice
    choice_type = tool_choice.get("type")
    if choice_type != "function":
        return dict(tool_choice)
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        raise ValueError("Anthropic function tool_choice requires a 'function' dictionary.")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Anthropic tool_choice requires a non-empty function name.")
    return {"type": "tool", "name": name}


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


class AnthropicProvider(BaseProvider):
    _TEXTUAL_BLOCK_TYPES: frozenset[str] = frozenset({"text", "output_text"})

    @classmethod
    def _normalize_text_block(cls, block: Any) -> dict[str, Any]:
        if not isinstance(block, dict):
            raise ValueError(
                "Anthropic content blocks must be dictionaries with 'type' and 'text'."
            )
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type:
            raise ValueError("Anthropic content blocks require a non-empty string 'type'.")
        if block_type not in cls._TEXTUAL_BLOCK_TYPES:
            raise UnsupportedContentBlockError(
                f"Anthropic provider does not support content block type '{block_type}'."
            )
        block_text = block.get("text")
        if not isinstance(block_text, str):
            raise ValueError(
                "Anthropic text-like blocks must include string 'text' values."
            )
        return block

    @classmethod
    def _normalize_text_content(cls, raw_content: Any) -> str:
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, dict):
            block = cls._normalize_text_block(raw_content)
            return block["text"]
        if isinstance(raw_content, list):
            parts: list[str] = []
            for block in raw_content:
                normalized_block = cls._normalize_text_block(block)
                parts.append(normalized_block["text"])
            return "".join(parts)
        raise ValueError("Anthropic messages must provide string or list content values.")

    @classmethod
    def _normalize_tool_result_blocks(cls, raw_content: Any) -> list[dict[str, Any]]:
        if isinstance(raw_content, list):
            blocks_source = raw_content
        else:
            blocks_source = [raw_content]

        normalized_blocks: list[dict[str, Any]] = []
        for block in blocks_source:
            if not isinstance(block, dict):
                raise ValueError(
                    "Anthropic tool result content must be provided as dictionaries."
                )
            block_type = block.get("type")
            if not isinstance(block_type, str) or not block_type:
                raise ValueError(
                    "Anthropic tool result blocks require a non-empty string 'type'."
                )
            if block_type in cls._TEXTUAL_BLOCK_TYPES:
                cls._normalize_text_block(block)
            normalized_blocks.append(block)

        return normalized_blocks

    @classmethod
    def _map_tool_call(cls, tool_call: Any) -> dict[str, Any]:
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

    @classmethod
    def _map_tool_result(cls, message: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("Anthropic tool messages require a 'tool_call_id'.")
        if "content" not in message:
            raise ValueError("Anthropic tool messages must include 'content'.")
        raw_content = message["content"]
        if isinstance(raw_content, (list, dict)):
            result_content = cls._normalize_tool_result_blocks(raw_content)
        else:
            text_content = cls._normalize_text_content(raw_content)
            result_content = cls._normalize_tool_result_blocks({"type": "text", "text": text_content})
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result_content,
        }

    def _prepare_chat_request(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | str | None,
        function_call: dict[str, Any] | str | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        logit_bias: dict[str, float] | None,
        response_format: dict[str, Any] | None,
        extra_options: dict[str, Any] | None,
        stream: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        _ = frequency_penalty
        _ = presence_penalty
        _ = logit_bias
        _ = response_format
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
            ends_with_messages = normalized_segments[-1].lower() == "messages"
            if not has_version_segment:
                insert_index = len(normalized_segments) - 1 if ends_with_messages else len(normalized_segments)
                normalized_segments.insert(insert_index, "v1")
            if not ends_with_messages:
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
        system_messages: list[str] = []
        mapped: list[dict[str, Any]] = []
        function_call_counter = 0
        for message in messages:
            role = message.get("role")
            if role == "system":
                if "content" not in message:
                    raise ValueError("Anthropic system messages must include 'content'.")
                system_messages.append(self._normalize_text_content(message["content"]))
                continue
            if role == "tool":
                mapped.append(
                    {
                        "role": "user",
                        "content": [self._map_tool_result(message)],
                    }
                )
                continue
            if role not in ("user", "assistant"):
                continue
            content_blocks: list[dict[str, Any]] = []
            if "content" in message:
                text_content = self._normalize_text_content(message["content"])
                if text_content:
                    content_blocks.append({"type": "text", "text": text_content})
            tool_calls = message.get("tool_calls")
            if tool_calls:
                if not isinstance(tool_calls, list):
                    raise ValueError("Anthropic tool calls must be provided as a list.")
                for tool_call in tool_calls:
                    content_blocks.append(self._map_tool_call(tool_call))
            function_call_field = message.get("function_call")
            if function_call_field is not None:
                if not isinstance(function_call_field, dict):
                    raise ValueError("Anthropic function_call must be provided as a dictionary.")
                function_call_counter += 1
                identifier_candidate = function_call_field.get("id")
                identifier: str | None
                if isinstance(identifier_candidate, str) and identifier_candidate:
                    identifier = identifier_candidate
                else:
                    message_identifier = message.get("id")
                    if isinstance(message_identifier, str) and message_identifier:
                        identifier = message_identifier
                    else:
                        identifier = f"function_call_{function_call_counter}"
                synthetic_tool_call = {
                    "id": identifier,
                    "type": "function",
                    "function": dict(function_call_field),
                }
                content_blocks.append(self._map_tool_call(synthetic_tool_call))
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            mapped.append({"role": role, "content": content_blocks})
        function_call_mode: str | None
        if isinstance(function_call, str):
            function_call_mode = function_call
        else:
            function_call_mode = None

        disable_tools = function_call_mode == "none"

        payload: dict[str, Any] = {
            "model": self.defn.model or model,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "messages": mapped,
            "stream": stream,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        if tools is not None and not disable_tools:
            payload["tools"] = _normalize_anthropic_tools(tools)

        normalized_tool_choice: dict[str, Any] | str | None = None
        if disable_tools:
            normalized_tool_choice = "none"
        elif tool_choice is not None:
            if isinstance(tool_choice, str):
                normalized_tool_choice = tool_choice
            elif isinstance(tool_choice, dict):
                normalized_tool_choice = _normalize_anthropic_tool_choice(tool_choice)
            else:
                raise ValueError("Anthropic tool_choice must be a string or dictionary.")
        elif isinstance(function_call, dict):
            name_candidate = function_call.get("name")
            if isinstance(name_candidate, str) and name_candidate:
                normalized_tool_choice = _normalize_anthropic_tool_choice(
                    {"type": "function", "function": {"name": name_candidate}}
                )
        elif function_call_mode is not None:
            normalized_tool_choice = function_call_mode

        if normalized_tool_choice is not None:
            payload["tool_choice"] = normalized_tool_choice
        cleaned_extra_options = dict(extra_options or {})
        extra_top_p = cleaned_extra_options.pop("top_p", None)
        unsupported_option_names = (
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "response_format",
        )
        for option_name in unsupported_option_names:
            cleaned_extra_options.pop(option_name, None)
        effective_top_p = top_p if top_p is not None else extra_top_p
        if effective_top_p is not None:
            payload["top_p"] = effective_top_p
        self._merge_extra_options(payload, cleaned_extra_options)
        return url, headers, payload

    @staticmethod
    def _map_stop_reason(raw: str | None) -> str | None:
        if raw is None:
            return None
        if raw == "tool_use":
            return "tool_calls"
        if raw in {"max_tokens", "message_limit"}:
            return "length"
        if raw in {"end_turn", "stop_sequence"}:
            return "stop"
        return raw

    async def _normalize_stream_events(
        self, events: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[ProviderStreamChunk]:
        stop_reason: str | None = None
        async for event in events:
            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue
            if event_type == "message_start":
                stop_reason = None
                message = event.get("message")
                role = "assistant"
                if isinstance(message, dict):
                    raw_role = message.get("role")
                    if isinstance(raw_role, str) and raw_role:
                        role = raw_role
                yield ProviderStreamChunk(
                    event_type="message_start",
                    delta={"role": role},
                    raw=event,
                )
                continue
            if event_type == "content_block_delta":
                index_value = event.get("index")
                block_delta = event.get("delta")
                if not isinstance(block_delta, dict):
                    continue
                delta_type = block_delta.get("type")
                if delta_type == "text_delta":
                    text_value = block_delta.get("text")
                    if isinstance(text_value, str) and text_value:
                        normalized_index = index_value if isinstance(index_value, int) else 0
                        yield ProviderStreamChunk(
                            event_type="delta",
                            index=normalized_index,
                            delta={"content": text_value},
                            raw=event,
                        )
                continue
            if event_type == "message_delta":
                delta_payload = event.get("delta")
                if not isinstance(delta_payload, dict):
                    continue
                stop_candidate = delta_payload.get("stop_reason")
                if isinstance(stop_candidate, str):
                    stop_reason = stop_candidate
                usage_payload = delta_payload.get("usage")
                if isinstance(usage_payload, dict):
                    usage: dict[str, int] = {}
                    prompt_tokens = usage_payload.get("input_tokens")
                    completion_tokens = usage_payload.get("output_tokens")
                    if isinstance(prompt_tokens, int):
                        usage["input_tokens"] = prompt_tokens
                    if isinstance(completion_tokens, int):
                        usage["output_tokens"] = completion_tokens
                    if usage:
                        yield ProviderStreamChunk(
                            event_type="usage",
                            usage=usage,
                            raw=event,
                        )
                continue
            if event_type == "message_stop":
                yield ProviderStreamChunk(
                    event_type="message_stop",
                    finish_reason=self._map_stop_reason(stop_reason),
                    raw=event,
                )
                stop_reason = None
                continue
            if event_type == "error":
                error_info = event.get("error")
                error_payload = dict(error_info) if isinstance(error_info, dict) else {}
                retry_after = event.get("retry_after")
                if isinstance(retry_after, (int, float)):
                    error_payload.setdefault("retry_after", float(retry_after))
                yield ProviderStreamChunk(
                    event_type="error",
                    error=error_payload or None,
                    raw=event,
                )
                continue

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
        url, headers, payload = self._prepare_chat_request(
            model,
            messages,
            temperature,
            max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            function_call=function_call,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            extra_options=extra_options,
            stream=False,
        )
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
                text_parts.append(self._normalize_text_content(result_content))

        if text_parts:
            joined_content = "".join(text_parts)
            content: str | None = joined_content if joined_content else None
        else:
            content = None
        finish_reason_raw = data.get("stop_reason")
        finish_reason = self._map_stop_reason(
            finish_reason_raw if isinstance(finish_reason_raw, str) else None
        )
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
    ) -> AsyncIterator[dict[str, Any]]:
        url, headers, payload = self._prepare_chat_request(
            model,
            messages,
            temperature,
            max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            function_call=function_call,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            extra_options=extra_options,
            stream=True,
        )

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()

                async def iter_events() -> AsyncIterator[dict[str, Any]]:
                    buffer: list[str] = []
                    async for line in response.aiter_lines():
                        if line is None:
                            continue
                        stripped = line.strip()
                        if not stripped:
                            if not buffer:
                                continue
                            data_text = "\n".join(buffer).strip()
                            buffer.clear()
                            if not data_text or data_text == "[DONE]":
                                continue
                            try:
                                parsed = json.loads(data_text)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(parsed, dict):
                                yield parsed
                            continue
                        if stripped.startswith("data:"):
                            buffer.append(stripped[5:].lstrip())
                    if buffer:
                        data_text = "\n".join(buffer).strip()
                        if data_text and data_text != "[DONE]":
                            try:
                                parsed = json.loads(data_text)
                            except json.JSONDecodeError:
                                parsed = None
                            if isinstance(parsed, dict):
                                yield parsed

                async for chunk in self._normalize_stream_events(iter_events()):
                    event_name = chunk.event_type or "message"
                    data_payload = asdict(chunk)
                    data_payload.pop("event_type", None)
                    yield {"event": event_name, "data": data_payload}

class OllamaProvider(BaseProvider):
    def _build_chat_request(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        response_format: dict[str, Any] | None,
        top_p: float | None,
        extra_options: dict[str, Any],
        stream: bool,
    ) -> tuple[str, dict[str, Any]]:
        url = f"{self.defn.base_url.rstrip('/')}/api/chat"
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        payload = {
            "model": self.defn.model or model,
            "messages": messages,
            "stream": stream,
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
        return url, payload

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
        url, payload = self._build_chat_request(
            model,
            messages,
            temperature,
            max_tokens,
            response_format=response_format,
            top_p=top_p,
            extra_options=extra_options,
            stream=False,
        )
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
    ) -> AsyncIterator[dict[str, Any]]:
        _ = tools
        _ = tool_choice
        _ = function_call
        _ = frequency_penalty
        _ = presence_penalty
        _ = logit_bias
        url, payload = self._build_chat_request(
            model,
            messages,
            temperature,
            max_tokens,
            response_format=response_format,
            top_p=top_p,
            extra_options=extra_options,
            stream=True,
        )

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                role_emitted = False
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        payload_line = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload_line, dict):
                        continue
                    message = payload_line.get("message")
                    if not isinstance(message, dict):
                        message = {}
                    choice: dict[str, Any] = {"index": 0}
                    delta_payload: dict[str, Any] = {}
                    role = message.get("role")
                    if isinstance(role, str) and role and not role_emitted:
                        delta_payload["role"] = role
                        role_emitted = True
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        delta_payload["content"] = content
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        delta_payload["tool_calls"] = tool_calls
                    function_delta = message.get("function_call")
                    if isinstance(function_delta, dict) and function_delta:
                        delta_payload["function_call"] = function_delta
                    if delta_payload:
                        choice["delta"] = delta_payload
                    done_flag = payload_line.get("done")
                    finish_reason = payload_line.get("done_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        choice["finish_reason"] = finish_reason
                    elif done_flag:
                        choice["finish_reason"] = "stop"
                    usage_payload = payload_line.get("usage")
                    if not isinstance(usage_payload, dict):
                        usage_payload = {}
                        prompt_eval = payload_line.get("prompt_eval_count")
                        eval_count = payload_line.get("eval_count")
                        if isinstance(prompt_eval, int):
                            usage_payload["prompt_tokens"] = prompt_eval
                        if isinstance(eval_count, int):
                            usage_payload["completion_tokens"] = eval_count
                        if not usage_payload:
                            usage_payload = None
                    chunk_payload: dict[str, Any] = {
                        "choices": [choice],
                        "model": payload_line.get("model") or self.defn.model or model,
                    }
                    if usage_payload:
                        chunk_payload["usage"] = usage_payload
                    if "delta" not in choice and "finish_reason" not in choice and "usage" not in chunk_payload:
                        continue
                    yield {"event": "chunk", "data": chunk_payload}

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
        # simple echo-ish behavior for tests
        _ = tools
        _ = tool_choice
        _ = function_call
        _ = extra_options
        last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "ping")
        return ProviderChatResponse(
            status_code=200,
            model="dummy",
            content=f"dummy:{last_user}",
            finish_reason="stop",
        )


from .openai import OpenAICompatProvider


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
    "UnsupportedContentBlockError",
    "ProviderStreamChunk",
    "BaseProvider",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "DummyProvider",
    "ProviderRegistry",
]
