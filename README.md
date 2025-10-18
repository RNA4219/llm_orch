# llm-orch (MVP)

OpenAI互換 `/v1/chat/completions` を受け付ける**薄いオーケストレーター**。**RPM/並列制御、429/5xx再試行、フォールバック、Ollama吸収、/healthz、JSONLメトリクス**に対応（MVP）。

<!-- LLM-BOOTSTRAP v1 -->
読む順番:
1. docs/birdseye/index.json  …… ノード一覧・隣接関係（軽量）
2. docs/birdseye/caps/<path>.json …… 必要ノードだけ point read（個別カプセル）

フォーカス手順:
- 直近変更ファイル±2hopのノードIDを index.json から取得
- 対応する caps/*.json のみ読み込み
<!-- /LLM-BOOTSTRAP -->

> ⚠️ 初期版は **非ストリーミング**（`stream: false` のみ）。Anthropic/OpenAI/Groq/Ollamaの最小互換。

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

### 環境変数

- `ORCH_INBOUND_API_KEYS` : カンマ区切りのAPIキー一覧。空なら認証なし。
- `ORCH_API_KEY_HEADER` : APIキーを読むヘッダ名（既定 `x-api-key`）。
- `ORCH_CORS_ALLOW_ORIGINS` : カンマ区切りの許可Origin。
- `ORCH_RETRY_AFTER_SECONDS` : `Retry-After` ヘッダ欠如時のフォールバック秒数（既定30秒）。

## メトリクス

- 既定は `metrics/requests-YYYYMMDD.jsonl` に 1リクエスト=1行追記。
- フィールド: `ts, req_id, task, provider, model, latency_ms, ok, status, error, retries, usage_prompt, usage_completion`
- Prometheusエンドポイント: `GET /metrics` （`orch_requests_total` カウンタ / `orch_request_latency_seconds` ヒストグラム）。`ORCH_INBOUND_API_KEYS` を設定した場合は同じキーで保護されます。

## ストリーミングフォーマット

- リクエストで `{"stream": true}` を指定すると `text/event-stream` を返却します。
- `data: {...}` 形式のOpenAI互換チャンクが連続し、終端は `data: [DONE]`。
- チャンクのJSONには `choices[].delta` など通常のOpenAI互換フィールドが含まれます。

## 既知の制限（MVP）

- トークン制限（TPM）は未対応（RPMのみ）。今後 `usage` を元にTPMバケット化予定。
