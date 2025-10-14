from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class ProviderChatResponse(BaseModel):
    status_code: int = 200
    model: str
    content: str
    usage_prompt_tokens: Optional[int] = 0
    usage_completion_tokens: Optional[int] = 0

def chat_response_from_provider(p: ProviderChatResponse) -> dict[str, Any]:
    import time, uuid
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": p.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": p.content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": p.usage_prompt_tokens or 0,
            "completion_tokens": p.usage_completion_tokens or 0,
            "total_tokens": (p.usage_prompt_tokens or 0) + (p.usage_completion_tokens or 0)
        }
    }
