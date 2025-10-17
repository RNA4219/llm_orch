#!/usr/bin/env bash
set -euo pipefail
export ORCH_USE_DUMMY=${ORCH_USE_DUMMY:-1}
uvicorn src.orch.server:app --port 31001 &
PID=$!
trap 'kill $PID >/dev/null 2>&1' EXIT
ready=false
for _ in {1..10}; do
  if curl -fs http://localhost:31001/healthz >/dev/null; then
    ready=true
    break
  fi
  sleep 1
done

if [[ $ready == false ]]; then
  echo "server did not start"
  exit 1
fi

curl -fs \
  -H 'Content-Type: application/json' \
  -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}]}' \
  http://localhost:31001/v1/chat/completions >/dev/null
