# llm-orch Technical Specification

## 1. 外部API
### 1.1 GET /healthz
- 200 OK: `{ "status":"ok", "providers": ["frontier_primary", "..."] }`

### 1.2 POST /v1/chat/completions
- 互換: OpenAI Chat Completions（非stream）。
- 入力（抜粋）:
  ```json
  {
    "model": "ignored-by-orch-or-used-as-hint",
    "messages": [{"role":"user","content":"hi"}],
    "temperature": 0.2,
    "max_tokens": 2048,
    "stream": false
  }
  ```
- ヘッダ: `x-orch-task-kind: PLAN|CRITIQUE|CODE|SUMMARY|BULK|DEFAULT`
- 出力（例）: OpenAI互換
  ```json
  {
    "id":"chatcmpl-...",
    "object":"chat.completion",
    "created": 1739560000,
    "model":"<effective model>",
    "choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
    "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
  }
  ```

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

## 3. レート制御と並列
- TokenBucket (per-provider): rpm で1分窓リフィル。残量0なら窓切替まで待機。
- Semaphore (per-provider): concurrency を超える同時実行を阻止。

## 4. リトライ/フォールバック
- 失敗（429/5xx/例外）時: バックオフ `min(0.25 * attempt, 2.0)` 秒（例）で再試行（MVP: 最大3回）。
- プロバイダ切替: primary 失敗→次の fallback へ。全滅で 502 を返す:
  ```json
  { "error": { "message": "<last error>" } }
  ```

## 5. メトリクス（JSONL）
- 出力: `metrics/requests-YYYYMMDD.jsonl`
- レコード例:
  ```json
  {
    "ts": 1739560000.123,
    "task": "CODE",
    "provider": "hosted_fast",
    "model": "llama-3.1-8b-instruct",
    "latency_ms": 245,
    "ok": true,
    "status": 200,
    "retries": 0,
    "usage_prompt": 0,
    "usage_completion": 0
  }
  ```

## 6. 環境変数
- ORCH_CONFIG_DIR … コンフィグディレクトリ（既定: `./config`）。
- ORCH_USE_DUMMY=1 … `providers.dummy.toml` を強制読み込み（通電用）。

## 7. プロバイダ動作（簡易）
- openai: `Authorization: Bearer $AUTH` / `POST {base}/v1/chat/completions`
- anthropic: `x-api-key`, `anthropic-version: 2023-06-01` / `POST {base}/v1/messages`
- ollama: `POST {base}/api/chat`
- dummy: 最後の user メッセージを `dummy:<text>` として返す。

## 8. 互換性・拡張
- 将来のSSEやTPM制御は defaults/providers 拡張で導入予定。
- 互換差異で破壊的変更が必要な場合は、新しいヘッダやルート名を導入し既存を維持。
