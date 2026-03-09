# llm-orch Requirements (SRS)

目的: OpenAI互換 `/v1/chat/completions` を1本に集約し、レート制御・自動再試行・フォールバック・Ollama吸収を行う「薄いオーケストレータ」を提供する。既存のLLMアダプタは `OPENAI_BASE_URL` の切替のみで利用可能にする。

## スコープ
- 対象: Chat Completions（SSEストリーミング含む）、Tokens/min（TPM）制御、レート・フォールバック機構。
- 非対象（MVP外）: 画像/ファイルAPI。

## 利用者ストーリー（要約）
- アプリ開発者として、429/5xxが出ても内側で再試行とフォールバックが働き、平均レイテンシのぶれが小さくなってほしい。
- 運用者として、1分あたりのRPMと同時並列concurrencyをプロバイダ単位で制御したい。
- コスト/BCPとして、ローカルOllamaを保険として組み込めるようにしたい。

## 機能要求（FR）
1. OpenAI互換: `POST /v1/chat/completions`・`GET /healthz`・`GET /metrics`（Prometheus形式）を提供する。
2. ルーティング: ヘッダ `x-orch-task-kind`（例: PLAN/CRITIQUE/CODE/SUMMARY/DEFAULT）で `router.yaml` に基づき primary→fallback で順に試行。
3. レート/TPM制御: `providers.toml` の `rpm`（1分あたり上限）、`tpm`（Tokens/min上限）、`concurrency`（同時実行）をプロバイダ単位で強制。
4. 自動再試行: 429/5xx/ネットワーク例外時に指数バックオフ（最大3回目安）で再試行。
5. フォールバック: primary 失敗時に次の候補へ自動切替。
6. ストリーミング: `stream:true` 指定時に SSE で逐次トークンを配信する。
7. 観測性: `metrics/requests-YYYYMMDD.jsonl` に1リクエスト1行で記録し、OpenTelemetry Exporterを通じてトレース/メトリクスを外部送信できる。
8. プロバイダ（MVP）: openai互換系 / Anthropic / Ollama / Dummy。
9. コンフィグ切替: `ORCH_CONFIG_DIR` と `ORCH_USE_DUMMY` で実行環境に応じて切替。
10. セキュリティ運用: APIキーの管理（環境変数/Secret Manager）、CORS許可ドメイン設定、キーのRedactionログ。

## 非機能要求（NFR）
- 性能: 単一ノードで 100 RPS 程度の低オーバーヘッド（環境依存）。
- 可用性: フォールバック構成時、単一プロバイダ障害でも一定の成功率を維持。
- 観測性: Prometheus `/metrics`・OpenTelemetryエクスポート・JSONLメトリクス・プロセスログ（INFO/WARN/ERROR）。
- セキュリティ: APIキー秘匿（環境変数/Secret Manager）、キー非出力、CORS許可ドメイン明示設定。

## 受け入れ基準（DoD）
- [x] `/healthz` が 200 を返し、構成済みプロバイダ名が列挙される。（test_health.py で検証済み）
- [x] `/v1/chat/completions`（Dummy構成）で単発リクエストが成功し、`stream:true` 指定時は SSE で逐次出力される。（test_health.py::test_chat_dummy, test_server_streaming_events.py で検証済み）
- [x] 429/5xx 擬似時に自動再試行が行われ、フォールバック先で成功するケースを確認。（test_server_routes.py::test_chat_retries_success_after_transient_failures で検証済み）
- [x] TPM上限を超えるリクエストが遮断され、対応するイベントがメトリクス/ログに記録される。（test_rate_limiter.py で検証済み）
- [x] `metrics/*.jsonl` への追記、Prometheus `/metrics` の公開、OpenTelemetryエクスポータ動作が確認できる。（test_metrics_logger.py, test_server_routes.py::test_metrics_endpoint_returns_prometheus_text で検証済み）
- [x] `OPENAI_BASE_URL=http://localhost:31001/v1` に切替した既存クライアントが変更なしで応答を得る。（OpenAI SDK互換性テストで検証済み）

## 最終確認（2026-03-09）
- 全テスト 252 passed, 2 skipped
- 機能凍結リリース v0.2.0 として完了
