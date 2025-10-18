# llm_orch 仕様概要 (v0.2 次段階)

## 要約

- OpenAI 互換の `/v1/chat/completions` を受け付ける薄いオーケストレーター。
- ルーター設定に基づき OpenAI / Anthropic / Groq / Ollama などへ転送し、RPM・並列数・TPM を `ProviderGuards` で制御しながらフォールバックを実施する（`src/orch/rate_limiter.py` の `SlidingWindowBucket`・`Guard`）。
- `stream:true` の SSE は `_stream_chat_response` で FastAPI の `StreamingResponse` を用いて提供し、プロバイダ別ストリームを正規化しつつフェイルオーバ通知を送出する（`src/orch/server.py`）。
- 応答は OpenAI 形式で返却し、`/healthz` と JSONL ログに加えて Prometheus `/metrics` で主要指標を公開する（`src/orch/server.py` の `_render_prometheus`）。
- リクエスト処理フロー：`POST /v1/chat/completions` → ルーティング (`RoutePlanner`) → レート / 並列 / TPM 制御 → プロバイダ呼び出しとフォールバック → OpenAI 形式応答 / SSE ストリーム → JSONL メトリクス・Prometheus 反映。
- 現状の制限 (MVP)：OpenTelemetry 連携は未着手。設定ファイルのホットリロードは未対応。TPM ガードは推定トークンに依存し、`usage` が欠落するプロバイダでは保守的なスロットリングが発生する。

## 次段階仕様書 (v0.2 Next Stage)

- **SSE ストリーミング**: `stream:true` 時のプロトコルとイベント (`chunk` / `[DONE]`)、フェイルオーバ時の通知設計。
- **TPM レート制御**: `usage` 情報を基にした 60 秒スライディングウィンドウ。欠落時はトークン推定で補完。
- **ルーティング高度化**: weighted / priority / sticky の戦略、サーキットブレーカ導入。
- **可観測性拡張**: Prometheus `/metrics`、OpenTelemetry 連携、主要指標のダッシュボード化。
- **セキュリティ / 運用**: API キー管理、CORS、設定ホットリロード、スキーマ検証。
- **受け入れ基準とマイルストーン**: M1 (Streaming)、M2 (TPM) などの段階的ゴール設定。

## 差分 TODO (棚卸し)

### 完了済み

- [x] SSE 実装の土台とフェイルオーバ通知（`src/orch/server.py` の `_stream_chat_response`）。
- [x] プロバイダ別ストリーム正規化（`_encode_event` と `provider.chat_stream` のイベントハンドリング）。
- [x] TPM バケット制御（`src/orch/rate_limiter.py` の `SlidingWindowBucket` と `Guard.record_usage`）。
- [x] Prometheus エクスポート（`src/orch/server.py` の `_render_prometheus` / `/metrics`）。
- [x] `router.yaml` の weighted / priority / sticky / circuit breaker 対応（`src/orch/router.py`）。
- [x] エラー正規化と `retry_after` 伝搬（`src/orch/server.py` の `_error_type_from_status` 周辺ロジック）。

### 未完了

- [ ] OpenTelemetry など外部トレースシステムへの出力。
- [ ] 設定ファイル (`providers.toml` / `router.yaml`) のホットリロードおよび再読み込み通知。
- [ ] ダッシュボード整備（Grafana 等）と TPM の実測値表示。
