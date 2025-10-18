# llm_orch 仕様概要 (v0.2 次段階)

## 要約

- OpenAI 互換の `/v1/chat/completions` を受け付ける薄いオーケストレーター。
- ルーター設定に基づき OpenAI / Anthropic / Groq / Ollama などへ転送し、RPM・並列数を制御しながら、429 / 5xx 時はリトライ後にフォールバック。
- 応答は OpenAI 形式で返却し、`/healthz` の疎通確認と JSONL 形式でのメトリクス出力を提供。
- 現状の制限 (MVP)：SSE ストリーミング未対応 (`stream:false` のみ)、TPM (Tokens/min) 未実装。将来は `usage` を用いた TPM バケット化を計画。
- リクエスト処理フロー (非ストリーミング)：`POST /v1/chat/completions` → ルーティング → レート / 並列制御 → プロバイダ呼び出しとフォールバック → OpenAI 形式応答 → JSONL メトリクス記録。

## 次段階仕様書 (v0.2 Next Stage)

- **SSE ストリーミング**: `stream:true` 時のプロトコルとイベント (`chunk` / `[DONE]`)、フェイルオーバ時の通知設計。
- **TPM レート制御**: `usage` 情報を基にした 60 秒スライディングウィンドウ。欠落時はトークン推定で補完。
- **ルーティング高度化**: weighted / priority / sticky の戦略、サーキットブレーカ導入。
- **可観測性拡張**: Prometheus `/metrics`、OpenTelemetry 連携、主要指標のダッシュボード化。
- **セキュリティ / 運用**: API キー管理、CORS、設定ホットリロード、スキーマ検証。
- **受け入れ基準とマイルストーン**: M1 (Streaming)、M2 (TPM) などの段階的ゴール設定。

## 差分 TODO (提案)

- [ ] SSE 実装の土台: FastAPI / Starlette の `EventSourceResponse` もしくは `text/event-stream` での逐次送出。
- [ ] プロバイダ別ストリーム変換アダプタ: OpenAI / Anthropic / Groq / Ollama の差異を "delta" に正規化。
- [ ] TPM バケット: `usage.prompt_tokens` / `usage.completion_tokens` を集計し、欠落時はトークナイザ推定で補完。60 秒スライド窓で遮断。
- [ ] Prometheus エクスポート: RPS / Latency / Errors / Retry / Rate 状態の 4 系列を計測。
- [ ] `router.yaml` 拡張: `strategy: weighted|priority|sticky`、重み・sticky TTL・CB 閾値を指定。
- [ ] エラー正規化: 429 / 5xx の OpenAI 互換エラー形式と `retry_after` の提供。
