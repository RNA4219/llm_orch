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

<!-- schema: ChatRequest -->
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "最新のオーダー状況を教えて。"}
  ],
  "temperature": 0.2,
  "max_tokens": 512,
  "stream": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_order_status",
        "description": "注文IDから現在の出荷状況を取得する",
        "parameters": {
          "type": "object",
          "properties": {
            "order_id": {"type": "string"}
          },
          "required": ["order_id"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "top_p": 0.9
}
```

## Docker

```bash
# ビルド（config/ 以下のサンプルを同梱）
docker build -t llm-orch:dev .

# 起動（ports:8000、config/ を read-only マウント）
docker compose up --build

# バックグラウンド
docker compose up -d
```

`docker compose` の環境変数は `.env` または `docker compose --env-file` で差し替え可能です。例:

```bash
# ダミー → 本番プロバイダ設定に切り替え
cp config/providers.dummy.toml config/providers.toml  # 必要に応じて編集
ORCH_USE_DUMMY=0 ORCH_CONFIG_DIR=/app/config docker compose up --build

# APIキーを追加
echo "ORCH_INBOUND_API_KEYS=sk-local-1" >> .env
docker compose up
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

## セキュリティ

- 本番環境では `ORCH_INBOUND_API_KEYS` によるAPIキー保護を必須運用とし、キー未設定時はアプリケーションログに警告が出力されることを監視してください。
- 既定の `ORCH_CORS_ALLOW_ORIGINS` は空文字列（=許可Originなし）であり、必要なOriginのみを明示的に列挙してください。
- APIキー値など機密情報はレスポンスやHTTPヘッダに含めず、ログにも平文で記録されません。APIキー保護を無効化した場合のみ警告ログが出力されます。

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

## エラーコード

| HTTPステータス | `error.code`                | 意味                             |
|----------------|-----------------------------|----------------------------------|
| 401            | `invalid_api_key`            | APIキー認証に失敗                 |
| 429            | `rate_limit`                 | プロバイダまたはガードのレート制限 |
| 5xx/BadGateway | `provider_server_error`      | プロバイダ側のサーバエラー         |
| 4xxその他      | `provider_error` / `routing_error` | プロバイダ起因の再試行不可エラー |

> 互換性: 既存クライアントは引き続き `error.type` を信頼できますが、`error.code` は上表の列挙値のみを前提に実装してください。

## ホットリロード

- `_config_refresh_loop` が `ORCH_CONFIG_DIR` 配下の `providers.toml` / `router.yaml` を監視し、更新検知時に `reload_configuration()` を経由して `RoutePlanner` / `ProviderRegistry` / `ProviderGuards` を再構築します。

## Sticky ルーティング

- `x-orch-sticky-key: <任意キー>` を付与した `/v1/chat/completions` リクエストは、該当キーに対し TTL 期間中同じプロバイダへ固定されます。
- 既存クライアントは `X-Orch-Session` ヘッダでも同等の動作を利用できます。

## 既知の制限

- TPMガードは `usage` が欠落するプロバイダでは推定トークンを用いるため、保守的なスロットリングが発生します。
