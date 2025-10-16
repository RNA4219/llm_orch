from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None


class ProviderChatResponse(BaseModel):
    status_code: int = 200
    model: str
    content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    usage_prompt_tokens: Optional[int] = 0
    usage_completion_tokens: Optional[int] = 0


def chat_response_from_provider(p: ProviderChatResponse) -> dict[str, Any]:
    import time
    import uuid

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": p.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    key: value
                    for key, value in {
                        "role": "assistant",
                        "content": p.content,
                        "tool_calls": p.tool_calls,
                        "function_call": p.function_call,
                    }.items()
                    if value is not None
                },
                "finish_reason": p.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p.usage_prompt_tokens or 0,
            "completion_tokens": p.usage_completion_tokens or 0,
            "total_tokens": (p.usage_prompt_tokens or 0)
            + (p.usage_completion_tokens or 0),
        },
    }
