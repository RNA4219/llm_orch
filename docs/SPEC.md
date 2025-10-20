# llm-orch Technical Specification

## 1. 外部API
### 1.1 GET /healthz
- 200 OK:
  ```json
  {
    "status": "ok",
    "providers": ["frontier_primary", "..."],
    "planner": {
      "last_reload_at": "2024-05-10T12:34:56Z",
      "watch": [
        {"name": "providers", "path": "providers.toml", "last_modified_at": "2024-05-10T12:34:00Z"},
        {"name": "router", "path": "router.yaml", "last_modified_at": "2024-05-10T12:33:00Z"}
      ]
    }
  }
  ```
  - `planner.last_reload_at`: 最後にルート設定を読み込んだUTC時刻（ISO 8601）。
  - `planner.watch[].path`: 監視対象ファイルの相対パス（絶対パスは公開しない）。
  - `planner.watch[].last_modified_at`: FastAPI起点で観測した最終更新UTC時刻。ファイルが見つからない場合は `null`。

### 1.2 POST /v1/chat/completions
- 互換: OpenAI Chat Completions（`stream:true` を推奨）。
- 入力（抜粋）:
  ```json
  {
    "model": "ignored-by-orch-or-used-as-hint",
    "messages": [{"role":"user","content":"hi"}],
    "temperature": 0.2,
    "max_tokens": 2048,
    "stream": true
  }
  ```
- 観測用レスポンスヘッダ:
  - `x-orch-request-id`: オーケストレータ内部リクエストID。
  - `x-orch-provider`: 実際に利用されたプロバイダ名。
  - `x-orch-fallback-attempts`: フォールバック実施回数。
- ストリームレスポンス: `Content-Type: text/event-stream`。SSEイベント例:
  ```
  event: chat.completion.chunk
  data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1739560000,"model":"<effective model>","choices":[{"index":0,"delta":{"role":"assistant","content":"..."},"finish_reason":null}]}

  event: telemetry.usage
  data: {"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

  event: done
  data: {}
  ```
- 非stream時（`stream:false`）はOpenAI互換JSONを単発返却。
- エラー時の例:
  ```json
  {
    "error": {
      "message": "rate limited",
      "type": "rate_limit",
      "retry_after": 2.5
    }
  }
  ```
- ヘッダ: `x-orch-task-kind: PLAN|CRITIQUE|CODE|SUMMARY|BULK|DEFAULT`

## 2. コンフィグ
### 2.1 providers.toml
- 例:
  ```toml
  [frontier_primary]
  type = "openai"        # openai|anthropic|ollama|dummy
  base_url = "https://api.openai.com/v1"
  model = "gpt-4.1-mini"
  auth_env = "OPENAI_API_KEY"
  rpm = 120
  concurrency = 8
  ```
- 解説:
  - openai: OpenAI互換エンドポイント（Groq等も可）。`/v1/chat/completions`。
  - anthropic: `/v1/messages` にマッピング。
  - ollama: `POST /api/chat` にマッピング。
  - dummy: ローカルエコーバック。
- 重み付き／優先度付きルーティング: `weight = 2` などの数値で均等割り込み、`priority = 0` で優先度ソート。
- スティッキー割当: `sticky_key = "account_id"` を指定すると同一キーでプロバイダ固定。
- サーキットブレーカ: `circuit_breaker = { failure_rate = 0.5, window = 30, cooldown = 120 }` を追加し、閾値超過で一定時間停止。
- CORS/APIキー: `allow_origins = ["https://example.com"]`, `api_keys = ["...hash..."]` を設定。
- ホットリロード: ファイル保存→`orchctl reload providers` 実行で無停止再読込。

### 2.2 router.yaml
- 例:
  ```yaml
  defaults: { temperature: 0.2, max_tokens: 2048, task_header: "x-orch-task-kind" }
  routes:
    PLAN:     { primary: frontier_primary,  fallback: [frontier_backup] }
    CODE:     { primary: hosted_fast,       fallback: [local_7b, frontier_primary] }
    SUMMARY:  { primary: hosted_fast,       fallback: [local_7b] }
    DEFAULT:  { primary: hosted_fast,       fallback: [frontier_primary, local_7b] }
  ```
- 解決手順: ヘッダ値→該当ルート、無ければ DEFAULT。primary→順に fallback。
- 追加設定例:
  - `weights`: `{ hosted_fast: 3, frontier_primary: 1 }` で加重ラウンドロビン。
  - `priority`: `[hosted_fast, frontier_primary, local_7b]` で優先順を固定。
  - `sticky`: `header: x-user-id, ttl: 3600` で同一ヘッダ値の間はプロバイダ固定。
  - `circuit_breaker_ref`: providers.toml で定義したブレーカを参照。
  - `cors`: `{ allow_origins: ["*"], allow_headers: ["content-type", "authorization"] }`。
  - `api_keys`: `[{ id: "ops", value_env: "ORCH_API_KEY_OPS" }]`。
  - ホットリロード手順: 変更→`orchctl reload router`→ヘルス確認。

## 3. レート制御と並列
- Token Per Minute (TPM) スライディングウィンドウ: 各プロバイダの `tpm` 設定で制御。ウィンドウ長1分、到着時刻ベースで残量算出。
- RPMは補助制限として従来のTokenBucket互換を維持。
- Semaphore (per-provider): concurrency を超える同時実行を阻止。

## 4. リトライ/フォールバック
- 失敗（429/5xx/例外）時: バックオフ `min(0.25 * attempt, 2.0)` 秒（例）で再試行（MVP: 最大3回）。
- プロバイダ切替: primary 失敗→次の fallback へ。全滅で 502 を返す:
  ```json
  { "error": { "message": "<last error>" } }
  ```

## 5. メトリクス
- Prometheusエクスポート: `orch_requests_total`, `orch_request_latency_seconds`, `orch_tokens_total` など。
- OpenTelemetryトレース: span属性に `orch.provider`, `orch.route`, `usage.prompt_tokens`, `usage.completion_tokens`。
- JSONLログ: `metrics/requests-YYYYMMDD.jsonl` は従来通り出力し、構造は互換。
- 両者併存ポリシー: JSONLはバッチ解析、Prometheus/OTelはリアルタイム監視を担当。停止不可の場合はJSONLのみ継続利用。

## 6. 環境変数
- ORCH_CONFIG_DIR … コンフィグディレクトリ（既定: `./config`）。
- ORCH_USE_DUMMY=1 … `providers.dummy.toml` を強制読み込み（通電用）。

## 7. プロバイダ動作（簡易）
- openai: `Authorization: Bearer $AUTH` / `POST {base}/v1/chat/completions`
- anthropic: `x-api-key`, `anthropic-version: 2023-06-01` / `POST {base}/v1/messages`
- ollama: `POST {base}/api/chat`
- dummy: 最後の user メッセージを `dummy:<text>` として返す。

## 8. 互換性・拡張
- SSE配信・TPM制御は現仕様でサポート。旧挙動が必要な場合は `stream:false` や従来設定を維持。
- 互換差異で破壊的変更が必要な場合は、新しいヘッダやルート名を導入し既存を維持。
