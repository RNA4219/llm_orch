# llm-orch Design

## 1. コンポーネント
- server.py: FastAPIエンドポイント `/healthz` `/v1/chat/completions`。ルート解決・再試行・フォールバック・メトリクス記録のオーケストレーション。
- router.py: `providers.toml` / `router.yaml` 読み込み、`RoutePlanner.plan(task)` でルート解決。
- providers.py: 各プロバイダ（OpenAI互換/Anthropic/Ollama/Dummy）。`ProviderRegistry.get(name)`。
- rate_limiter.py: Providerごとの TokenBucket + Semaphore に加え、TPMスライディングウィンドウでトークン消費を追跡。
- stream_adapter.py: プロバイダ固有のレスポンスをSSE互換チャンクへ再構成し、フォールバック通知と完了イベントを仲介。
- metrics.py: JSONLロガーとPrometheus/OTelエクスポータ。リクエスト/プロバイダ単位のメトリクスを発行。
- types.py: Pydanticモデル（ChatRequestなど）とOpenAI互換応答ビルダ。

## 2. シーケンス
1. `POST /v1/chat/completions` を受信。
2. `x-orch-task-kind` → `RoutePlanner.plan(task)`。
3. 候補ルートごとに `Guard` でRPM/TPM/並列のリソース確保。
4. `stream` フラグでフローが分岐:
   - `stream: true`
     1. `provider.stream_chat(...)` からのイベントを `StreamAdapter` がチャンク化しSSEで送出。
     2. プロバイダ失敗時は即座にフェイルオーバ通知チャンクを送信し、次候補へスイッチ。
     3. 成功完了後にメトリクス（Prometheus/OTel + JSONL）を送出し、`[DONE]` を返却。
   - 非ストリーム
     1. `provider.chat(...)` でレスポンスを獲得後にチャンク化不要のため即ボディを構築。
     2. エラー時はリトライ/フェイルオーバを内部で完遂してから応答を返却。
     3. レスポンス組み立て後にメトリクスを送出し、OpenAI互換JSONを返却。
5. 全滅で 502。

## 3. エラー方針
- `httpx` 例外・タイムアウト・非2xxは capture。`last_err` として保持。
- リトライは軽いジッター付き（例: 0.25s, 0.5s, 0.75s, ... 上限2s）。

## 4. 制約と設計根拠
- OpenAI互換APIとSSE: 既存クライアントの差し替えを最小化しつつ、ストリーム/非ストリーム双方を同一エンドポイントで提供。
- TPMスライディングウィンドウ: プロバイダ仕様に合わせた分単位のトークン消費制御で、突発的な負荷集中を緩和。
- OpenTelemetry/Prometheus: オペレーション統合のためにOTelメトリクス/トレースを第一級とし、Prometheusエクスポータを併設して従来運用と接続。
- JSONLメトリクス: 履歴監査とローカル検証を低コストで実現する補助チャネル。
- Anthropic等プロバイダ差異はStreamAdapterで統一フォーマットへ吸収。完全互換を保証しない。

## 6. 推奨ディレクトリ構成
```
config/{providers.toml, router.yaml, providers.dummy.toml}
metrics/.gitkeep
src/orch/{server.py, router.py, providers.py, rate_limiter.py, metrics.py, types.py}
docs/{REQUIREMENTS.md, SPEC.md, DESIGN.md}
```

## 7. 受け入れテスト（例）
- `/healthz` が 200。
- `ORCH_USE_DUMMY=1` で dummy 経由の `/v1/chat/completions` が 200。
- 429/5xx をモックして、再試行とフォールバックが機能すること。
- 高並列で叩き、しきい値超過で待機が発生すること（レート制御動作）。
