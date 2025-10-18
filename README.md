# llm-orch (MVP)

OpenAI互換 `/v1/chat/completions` を受け付ける**薄いオーケストレーター**。**RPM/並列制御、429/5xx再試行、フォールバック、Ollama吸収、/healthz、JSONLメトリクス**に対応（MVP）。

> ⚠️ 初期版は **非ストリーミング**（`stream: false` のみ）。Anthropic/OpenAI/Groq/Ollamaの最小互換。

<!-- LLM-BOOTSTRAP v1 -->
読む順番:
1. docs/birdseye/index.json  …… ノード一覧・隣接関係（軽量）
2. docs/birdseye/caps/<path>.json …… 必要ノードだけ point read（個別カプセル）

フォーカス手順:
- 直近変更ファイル±2hopのノードIDを index.json から取得
- 対応する caps/*.json のみ読み込み
<!-- /LLM-BOOTSTRAP -->

## Quick Start

```bash
# 1) 依存
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) コンフィグ（サンプルのままでOK / ダミーは providers.dummy.toml）
export ORCH_CONFIG_DIR=./config
# export OPENAI_API_KEY=sk-...
# export ANTHROPIC_API_KEY=...

# 3) 起動
uvicorn src.orch.server:app --port 31001

# 4) 疎通
curl -s http://localhost:31001/healthz
curl -s -H "Content-Type: application/json" \
  -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}]}' \
  http://localhost:31001/v1/chat/completions | jq .
```

## 設定

- `config/providers.toml` : プロバイダ定義（type/base_url/model/auth_env/rpm/concurrency）
- `config/router.yaml` : タスク種別ごとのプライマリ/フォールバック定義（ヘッダ `x-orch-task-kind`）

> ローカル動作のみなら `providers.dummy.toml` を `providers.toml` に置き換えてください。

## メトリクス

- 既定は `metrics/requests-YYYYMMDD.jsonl` に 1リクエスト=1行追記。
- フィールド: `ts, req_id, task, provider, model, latency_ms, ok, status, error, retries, usage_prompt, usage_completion`

## 既知の制限（MVP）

- SSEストリーミング非対応（今後 `/v1/chat/completions` の `stream=true` に対応予定）
- トークン制限（TPM）は未対応（RPMのみ）。今後 `usage` を元にTPMバケット化予定。
