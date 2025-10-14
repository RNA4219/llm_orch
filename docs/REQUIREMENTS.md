# llm-orch Requirements (SRS)

目的: OpenAI互換 `/v1/chat/completions` を1本に集約し、レート制御・自動再試行・フォールバック・Ollama吸収を行う「薄いオーケストレータ」を提供する。既存のLLMアダプタは `OPENAI_BASE_URL` の切替のみで利用可能にする。

## スコープ
- 対象: Chat Completions（非Streaming MVP）。
- 非対象（MVP外）: SSEストリーミング、Tokens/min（TPM）制御、画像/ファイルAPI。

## 利用者ストーリー（要約）
- アプリ開発者として、429/5xxが出ても内側で再試行とフォールバックが働き、平均レイテンシのぶれが小さくなってほしい。
- 運用者として、1分あたりのRPMと同時並列concurrencyをプロバイダ単位で制御したい。
- コスト/BCPとして、ローカルOllamaを保険として組み込めるようにしたい。

## 機能要求（FR）
1. OpenAI互換: `POST /v1/chat/completions` と `GET /healthz` を提供する。
2. ルーティング: ヘッダ `x-orch-task-kind`（例: PLAN/CRITIQUE/CODE/SUMMARY/DEFAULT）で `router.yaml` に基づき primary→fallback で順に試行。
3. レート制御: `providers.toml` の `rpm`（1分あたり上限）と `concurrency`（同時実行）をプロバイダ単位で強制。
4. 自動再試行: 429/5xx/ネットワーク例外時に指数バックオフ（最大3回目安）で再試行。
5. フォールバック: primary 失敗時に次の候補へ自動切替。
6. メトリクス: `metrics/requests-YYYYMMDD.jsonl` に1リクエスト1行で記録（latency, provider, retries, usageなど）。
7. プロバイダ（MVP）: openai互換系 / Anthropic / Ollama / Dummy。
8. コンフィグ切替: `ORCH_CONFIG_DIR` と `ORCH_USE_DUMMY` で実行環境に応じて切替。

## 非機能要求（NFR）
- 性能: 単一ノードで 100 RPS 程度の低オーバーヘッド（環境依存）。
- 可用性: フォールバック構成時、単一プロバイダ障害でも一定の成功率を維持。
- 観測性: JSONLメトリクスとプロセスログ（INFO/WARN/ERROR）。
- セキュリティ: APIキーは環境変数からのみ取得。キーはログに出力しない。CORSはアプリ構成に合わせて無効/限定。

## 受け入れ基準（DoD）
- [ ] `/healthz` が 200 を返し、構成済みプロバイダ名が列挙される。
- [ ] `/v1/chat/completions`（Dummy構成）で 1 回成功。
- [ ] 429/5xx 擬似時に自動再試行が行われ、フォールバック先で成功するケースを確認。
- [ ] `metrics/*.jsonl` に追記される（latency_ms, provider, retries, usage_*）。
- [ ] `OPENAI_BASE_URL=http://localhost:31001/v1` に切替した既存クライアントが変更なしで応答を得る。
