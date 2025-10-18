#!/usr/bin/env bash
set -euo pipefail

ORCH_API_KEY_HEADER=${ORCH_API_KEY_HEADER:-x-api-key}
AUTH_HEADER_ARGS=()
if [[ -n "${ORCH_INBOUND_API_KEYS:-}" ]]; then
  IFS=',' read -r first_key _ <<<"${ORCH_INBOUND_API_KEYS}"
  first_key="$(printf '%s' "${first_key}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [[ -n "${first_key}" ]]; then
    AUTH_HEADER_ARGS=(-H "${ORCH_API_KEY_HEADER}: ${first_key}")
  fi
fi

export ORCH_USE_DUMMY=${ORCH_USE_DUMMY:-1}
uvicorn src.orch.server:app --port 31001 &
PID=$!
trap 'kill $PID >/dev/null 2>&1' EXIT
ready=false
for _ in {1..10}; do
  curl -f -s "${AUTH_HEADER_ARGS[@]}" http://localhost:31001/healthz >/dev/null && {
    ready=true
    break
  }
  sleep 1
done

if [[ $ready == false ]]; then
  echo "server did not start"
  exit 1
fi

curl -fs \
  "${AUTH_HEADER_ARGS[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}]}' \
  http://localhost:31001/v1/chat/completions >/dev/null
