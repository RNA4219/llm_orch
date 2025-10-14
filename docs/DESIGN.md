# llm-orch Design

## 1. コンポーネント
- server.py: FastAPIエンドポイント `/healthz` `/v1/chat/completions`。ルート解決・再試行・フォールバック・メトリクス記録のオーケストレーション。
- router.py: `providers.toml` / `router.yaml` 読み込み、`RoutePlanner.plan(task)` でルート解決。
- providers.py: 各プロバイダ（OpenAI互換/Anthropic/Ollama/Dummy）。`ProviderRegistry.get(name)`。
- rate_limiter.py: Providerごとの TokenBucket + Semaphore。`Guard` で `async with` 制御。
- metrics.py: JSONLロガー。日付別ファイルへappend。
- types.py: Pydanticモデル（ChatRequestなど）とOpenAI互換応答ビルダ。

## 2. シーケンス
1. `POST /v1/chat/completions` を受信。
2. `x-orch-task-kind` → `RoutePlanner.plan(task)`。
3. primary → fallback 順で:
   - `Guard` でRPM/並列を確保 → `provider.chat(...)` 実行。
   - 例外/429/5xx → バックオフして再試行。失敗なら次へ。
4. 成功でOpenAI互換レスポンスを返却。メトリクスをJSONLへ出力。
5. 全滅で 502。

## 3. エラー方針
- `httpx` 例外・タイムアウト・非2xxは capture。`last_err` として保持。
- リトライは軽いジッター付き（例: 0.25s, 0.5s, 0.75s, ... 上限2s）。

## 4. 設計の根拠
- OpenAI互換: 既存クライアントの差し替えを最小化。
- TokenBucket+Semaphore: RPMと並列の直感的運用。TPMは後続拡張。
- JSONLメトリクス: 軽量・追記のみでツールに取り込みやすい。

## 5. 制約
- SSE非対応（MVP）。Anthropic特有のメッセージ構造差異は簡易吸収のみ。

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
