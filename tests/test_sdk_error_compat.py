from __future__ import annotations

import json, shutil, subprocess, sys
from pathlib import Path
from typing import Any

import httpx, pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch import server as orch_server

_PARAMS: list[tuple[int, str, str, dict[str, Any]]] = [
    (400, "provider_error", "bad request", {}),
    (401, "authentication_error", "missing or invalid api key", {"code": "invalid_api_key"}),
    (429, "rate_limit", "rate limited", {"retry_after": orch_server.DEFAULT_RETRY_AFTER_SECONDS}),
    (502, "provider_server_error", "upstream failed", {}),
]


@pytest.mark.parametrize("status, kind, message, extra", _PARAMS)
def test_python_openai_sdk(status: int, kind: str, message: str, extra: dict[str, Any]) -> None:
    openai = pytest.importorskip("openai")
    body = orch_server._make_error_body(status_code=status, message=message, error_type=kind, **extra)
    client = openai.OpenAI(
        api_key="test",
        base_url="https://mock.local/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(status_code=status, json=body)),
            base_url="https://mock.local",
        ),
    )
    with pytest.raises(openai.APIStatusError) as exc:
        client.chat.completions.create(model="dummy", messages=[{"role": "user", "content": "hi"}])
    assert exc.value.response.status_code == status
    assert exc.value.response.json() == body


def test_node_openai_sdk() -> None:
    if shutil.which("node") is None:
        pytest.skip("node runtime is unavailable")
    probe = subprocess.run(
        ["node", "--input-type=module", "-e", "import('openai').then(()=>0,()=>process.exit(1));"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("openai package is not installed for node")
    cases = json.dumps([(status, orch_server._make_error_body(status_code=status, message=message, error_type=kind, **extra)) for status, kind, message, extra in _PARAMS])
    script = (
        "import assert from'node:assert/strict';import OpenAI from'openai';import{MockAgent,Agent,setGlobalDispatcher}from'undici';"
        f"const cases={cases};"
        "(async()=>{for(const [status,payload]of cases){const agent=new MockAgent();agent.disableNetConnect();const pool=agent.get('https://mock.local');"
        "pool.intercept({method:'POST',path:'/v1/chat/completions'}).reply(status,payload);setGlobalDispatcher(agent);const client=new OpenAI({apiKey:'test',baseURL:'https://mock.local/v1'});"
        "let threw=false;try{await client.chat.completions.create({model:'dummy',messages:[{role:'user',content:'hi'}]});}"
        "catch(err){threw=true;assert.equal(err.status,status);const body=err.error??err;assert.deepEqual(body,payload.error??payload);}"
        "assert.ok(threw,'expected API error');await agent.close();}})().then(()=>setGlobalDispatcher(new Agent())).catch(err=>{console.error(err);process.exit(1);});"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr or result.stdout)
