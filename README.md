# llm-orch

OpenAI互換 `/v1/chat/completions` を受け付ける**薄いオーケストレーター**。**SSEストリーミング、weighted/priority/sticky ルーティング、RPM/TPM/並列制御、429/5xx再試行、フォールバック、Prometheus/OpenTelemetry メトリクス、APIキー/CORS、ホットリロード**に対応しています。

<!-- LLM-BOOTSTRAP v1 -->
読む順番:
1. docs/birdseye/index.json  …… ノード一覧・隣接関係（軽量）
2. docs/birdseye/caps/<path>.json …… 必要ノードだけ point read（個別カプセル）

フォーカス手順:
- 直近変更ファイル±2hopのノードIDを index.json から取得
- 対応する caps/*.json のみ読み込み
<!-- /LLM-BOOTSTRAP -->

> ℹ️ SSE（Server-Sent Events）に対応済み。Anthropic/OpenAI/Groq/Ollamaの最小互換ストリーム/非ストリームを同一エンドポイントで扱えます。

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

### Stickyヘッダ付きクライアント例

#### 共通リクエストペイロード

<!-- schema: ChatRequest -->
```json
{
  "model": "dummy",
  "messages": [
    {"role": "system", "content": "You are llm-orch demo."},
    {"role": "user", "content": "Hello from README"}
  ],
  "stream": false
}
```

#### curl（Stickyヘッダ）

```bash
cat <<'JSON' > request.json
{
  "model": "dummy",
  "messages": [
    {"role": "system", "content": "You are llm-orch demo."},
    {"role": "user", "content": "Hello from README"}
  ],
  "stream": false
}
JSON

curl -sSf -H "Content-Type: application/json" \
  -H "X-Orch-Sticky-Key: demo-session" \
  -d @request.json \
  http://localhost:31001/v1/chat/completions | jq .
```

#### Python（httpx SDK）

```python
from __future__ import annotations

import asyncio
from typing import Any

import httpx

REQUEST_PAYLOAD: dict[str, Any] = {
    "model": "dummy",
    "messages": [
        {"role": "system", "content": "You are llm-orch demo."},
        {"role": "user", "content": "Hello from README"},
    ],
    "stream": False,
}


async def main() -> None:
    async with httpx.AsyncClient(base_url="http://localhost:31001") as client:
        response = await client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD,
            headers={"X-Orch-Sticky-Key": "demo-session"},
            timeout=30.0,
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    asyncio.run(main())
```

#### JavaScript（Fetch API）

```javascript
const payload = {
  model: "dummy",
  messages: [
    { role: "system", content: "You are llm-orch demo." },
    { role: "user", content: "Hello from README" }
  ],
  stream: false
};

const response = await fetch("http://localhost:31001/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Orch-Sticky-Key": "demo-session"
  },
  body: JSON.stringify(payload)
});

if (!response.ok) {
  throw new Error(`Request failed: ${response.status}`);
}

const body = await response.json();
console.log(body);
```

#### ストリーミング受信例

<!-- schema: ChatRequest -->
```json
{
  "model": "dummy",
  "messages": [
    {"role": "system", "content": "You are llm-orch demo."},
    {"role": "user", "content": "Stream please"}
  ],
  "stream": true
}
```

```bash
cat <<'JSON' > stream.json
{
  "model": "dummy",
  "messages": [
    {"role": "system", "content": "You are llm-orch demo."},
    {"role": "user", "content": "Stream please"}
  ],
  "stream": true
}
JSON

curl -sN -H "Content-Type: application/json" \
  -H "X-Orch-Sticky-Key: demo-session" \
  -d @stream.json \
  http://localhost:31001/v1/chat/completions | while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == data:\ \[DONE\] ]]; then
      break
    fi
  done
```

## 設定

- `config/providers.toml` : プロバイダ定義（`type` / `base_url` / `model` / `auth_env` / `rpm` / `tpm` / `concurrency`）
- `config/router.yaml` : タスク種別ごとの weighted / priority / sticky ルートを宣言（ヘッダ `x-orch-task-kind`）。sticky ルートは `RoutePlanner.plan(..., sticky_key=...)` または HTTP リクエストヘッダ `x-orch-sticky-key` / `X-Orch-Session` から渡されたキーにより TTL 内固定が行われます。

> ローカル動作のみなら `providers.dummy.toml` を `providers.toml` に置き換えてください。

### 環境変数

- `ORCH_INBOUND_API_KEYS` : カンマ区切りのAPIキー一覧。空なら認証なし。設定時は `tools/ci/smoke.sh` も同じキーで疎通します。
- `ORCH_API_KEY_HEADER` : APIキーを読むヘッダ名（既定 `x-api-key`）。
- `ORCH_CORS_ALLOW_ORIGINS` : カンマ区切りの許可Origin。
- `ORCH_RETRY_AFTER_SECONDS` : `Retry-After` ヘッダ欠如時のフォールバック秒数（既定30秒）。
- `ORCH_CONFIG_REFRESH_INTERVAL` : 設定ファイル変更を監視するポーリング間隔（秒）。既定30秒。`0` でポーリング毎ループ。
- `ORCH_METRICS_EXPORT_MODE` : メトリクス出力モード。`prom`（Prometheusのみ）/`otel`（OTelのみ）/`both`（両方）。既定は `prom`。後方互換として `ORCH_OTEL_METRICS_EXPORT` を真値にすると `both` 相当になります。
- `ORCH_OTEL_METRICS_EXPORT` : OpenTelemetryメトリクスを旧来通り有効化する互換フラグ。`ORCH_METRICS_EXPORT_MODE` 未設定時のみ参照されます。

## メトリクス

- 既定は `metrics/requests-YYYYMMDD.jsonl` に 1リクエスト=1行追記。
- Prometheusモード（`prom`/`both`）では同ディレクトリに `metrics/prometheus.prom` をエクスポート。
- フィールド: `ts, req_id, task, provider, model, latency_ms, ok, status, error, retries, usage_prompt, usage_completion`
- Prometheusエンドポイント: `GET /metrics` （`orch_requests_total` カウンタ / `orch_request_latency_seconds` ヒストグラム）。`ORCH_INBOUND_API_KEYS` を設定した場合は同じキーで保護されます。
- OpenTelemetryエクスポートは `ORCH_METRICS_EXPORT_MODE` を `otel`/`both` にするか、互換フラグ `ORCH_OTEL_METRICS_EXPORT=1` を設定すると有効化され、`requests_total` カウンタと `latency_ms` ヒストグラムをプロバイダ/HTTPステータス/成功可否属性付きで送出します。`MetricsLogger` はJSONL書き込み時に同じレコードをOpenTelemetryにも流し、`flush()` 呼び出しで強制エクスポートします。

## ストリーミングフォーマット

- リクエストで `{"stream": true}` を指定すると `text/event-stream` を返却します。
- `data: {...}` 形式のOpenAI互換チャンクが連続し、終端は `data: [DONE]`。
- 各イベントには `event: chat.completion.chunk` / `event: telemetry.usage` / `event: done` が付与され、`choices[].delta` などOpenAI互換フィールドを含みます。
- プライマリプロバイダが初回チャンク生成前に 5xx や接続例外を返した場合はルート定義のフォールバック先へ自動で切り替わり、すべて失敗した場合のみJSONエラーを返します。再試行不可な 4xx/429 は即座にJSONエラーとなり、429/5xx時は `retry_after` を付与します。
- `ProviderGuards` によりRPM/並列/TPM制御がストリームでも適用され、フォールバックごとにスロットが解放・再取得されるため、TPMバケット残量が不足すると待機またはエラーになります。

## レスポンスヘッダ

- すべての `/v1/chat/completions` 応答（成功・エラー・SSE）で `x-orch-request-id` / `x-orch-provider` / `x-orch-fallback-attempts` を返却します。フォールバックが発生した場合は試行回数に応じて `x-orch-fallback-attempts` が加算されます。

## ホットリロード

- `_config_refresh_loop` が `ORCH_CONFIG_DIR` 配下の `providers.toml` / `router.yaml` を監視し、更新検知時に `reload_configuration()` を経由して `RoutePlanner` / `ProviderRegistry` / `ProviderGuards` を再構築します。

## Sticky ルーティング

- `x-orch-sticky-key: <任意キー>` を付与した `/v1/chat/completions` リクエストは、該当キーに対し TTL 期間中同じプロバイダへ固定されます。
- 既存クライアントは `X-Orch-Session` ヘッダでも同等の動作を利用できます。

## 既知の制限

- TPMガードは `usage` が欠落するプロバイダでは推定トークンを用いるため、保守的なスロットリングが発生します。
