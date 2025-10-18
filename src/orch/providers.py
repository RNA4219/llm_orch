import json
import os
import re
from urllib.parse import urlparse, urlunparse
from typing import Dict, Any, List

import httpx

from .router import ProviderDef
from .types import ProviderChatResponse


class UnsupportedContentBlockError(ValueError):
    """Raised when a request includes a content block unsupported by a provider."""

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

class AnthropicProvider(BaseProvider):
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
        textual_block_types: set[str] = {"text", "output_text"}

        def _normalize_text_block(block: Any) -> dict[str, Any]:
            if not isinstance(block, dict):
                raise ValueError(
                    "Anthropic content blocks must be dictionaries with 'type' and 'text'."
                )
            block_type = block.get("type")
            if not isinstance(block_type, str) or not block_type:
                raise ValueError(
                    "Anthropic content blocks require a non-empty string 'type'."
                )
            if block_type not in textual_block_types:
                raise UnsupportedContentBlockError(
                    f"Anthropic provider does not support content block type '{block_type}'."
                )
            block_text = block.get("text")
            if not isinstance(block_text, str):
                raise ValueError(
                    "Anthropic text-like blocks must include string 'text' values."
                )
            return block

        def normalize_text_content(raw_content: Any) -> str:
            if raw_content is None:
                return ""
            if isinstance(raw_content, str):
                return raw_content
            if isinstance(raw_content, dict):
                block = _normalize_text_block(raw_content)
                return block["text"]
            if isinstance(raw_content, list):
                parts: list[str] = []
                for block in raw_content:
                    normalized_block = _normalize_text_block(block)
                    parts.append(normalized_block["text"])
                return "".join(parts)
            raise ValueError("Anthropic messages must provide string or list content values.")

        def normalize_tool_result_blocks(raw_content: Any) -> list[dict[str, Any]]:
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
                if block_type in textual_block_types:
                    _normalize_text_block(block)
                normalized_blocks.append(block)

            return normalized_blocks

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
            if isinstance(raw_content, (list, dict)):
                result_content = normalize_tool_result_blocks(raw_content)
            elif raw_content is None or isinstance(raw_content, str):
                text_content = normalize_text_content(raw_content)
                result_content = normalize_tool_result_blocks(
                    {"type": "text", "text": text_content}
                )
            else:
                text_content = normalize_text_content(raw_content)
                result_content = normalize_tool_result_blocks(
                    {"type": "text", "text": text_content}
                )
            return {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result_content,
            }

        system_messages: list[str] = []
        mapped: list[dict[str, Any]] = []
        function_call_counter = 0
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
                content_blocks.append(map_tool_call(synthetic_tool_call))
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
        cleaned_extra_options = dict(extra_options)
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

        if text_parts:
            joined_content = "".join(text_parts)
            content: str | None = joined_content if joined_content else None
        else:
            content = None
        finish_reason_raw = data.get("stop_reason")
        finish_reason: str | None
        if isinstance(finish_reason_raw, str):
            if finish_reason_raw == "tool_use":
                finish_reason = "tool_calls"
            elif finish_reason_raw in {"max_tokens", "message_limit"}:
                finish_reason = "length"
            elif finish_reason_raw in {"end_turn", "stop_sequence"}:
                finish_reason = "stop"
            else:
                finish_reason = finish_reason_raw
        else:
            finish_reason = None
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
        payload = {
            "model": self.defn.model or model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                payload["format"] = "json"
            else:
                raise ValueError(
                    "OllamaProvider only supports response_format type 'json_object'."
                )
        cleaned_options = {
            key: value
            for key, value in extra_options.items()
            if key not in self._RESERVED_OPTION_KEYS and value is not None
        }
        if cleaned_options:
            payload["options"].update(cleaned_options)
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
