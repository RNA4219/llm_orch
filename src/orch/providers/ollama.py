from __future__ import annotations

from typing import Any, List

import httpx

from src.orch._providers_legacy import BaseProvider
from ..types import ProviderChatResponse

__all__ = ["OllamaProvider"]


class OllamaProvider(BaseProvider):
    async def chat(
        self,
        model: str,
        messages: List[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
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
        _ = frequency_penalty
        _ = presence_penalty
        _ = logit_bias
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
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        finish_reason = data.get("finish_reason") or data.get("done_reason")
        tool_calls = message.get("tool_calls")
        return ProviderChatResponse(
            status_code=response.status_code,
            model=self.defn.model or model,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
        )
