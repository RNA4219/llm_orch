#!/usr/bin/env bash
set -euo pipefail
uvicorn src.orch.server:app --port 31001 &
PID=$!
sleep 1
curl -f http://localhost:31001/healthz
curl -s -H 'Content-Type: application/json'   -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}]}'   http://localhost:31001/v1/chat/completions >/dev/null
kill $PID
