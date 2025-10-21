# TASKS

- [x] README: OpenTelemetryメトリクス説明を更新（`ORCH_OTEL_METRICS_EXPORT` 追記、既知の制限を整理済み）。他タスクは同節の編集時にOTel設定例を保持すること。
- [x] src/orch/server.py: `_config_refresh_loop` のリロード処理を更新済み。後続タスクは同ロジック変更との重複に注意。最新plannerへの差し替え後は即座に再評価する設計。
- [x] tests/test_server_config_reload.py: plannerのリロード判定テストを追加済み。重複する検証ケースの再追加を避けること。
- [x] README: 既知の制限節を `_config_refresh_loop` / `reload_configuration()` の自動監視・反映説明に更新済み。今後の追記でもホットリロード対応済みである点を保持すること。
- [x] docs/birdseye: rate_limiterノード追加とcaps整備（担当: gpt-5-codex、完了）。
- [x] docs/birdseye: rate_limiterカプセル要約を刷新し index の generated_at を更新（担当: gpt-5-codex、2025-10-19 完了）。
- [x] tests/test_server_streaming_events.py / src/orch/server.py: SSEイベントの仕様名マッピングと `[DONE]` センチネル互換を担保する実装・テストを追加済み。後続改修時は `chat.completion.chunk` / `telemetry.usage` / `done` のエイリアス維持を徹底すること。
- [x] tests/test_server_streaming_events.py / src/orch/server.py: dict由来 `event_type` も SSE 仕様イベント名に正規化し、`data` を常に JSON 解釈可能にする改修を実施済み。重複対応を避け、`event`/`event_type` の両経路で仕様名を崩さないこと。
- [x] tests/test_server_streaming_events.py / src/orch/server.py: dataclass 由来イベントの `event` 属性を SSE 仕様名へ正規化し、内部フィールドを除外する挙動を追加済み。後続改修でも `done` 終端と `[DONE]` センチネルの整合性を維持すること。
- [x] tests/test_server_streaming_events.py / src/orch/server.py: `_encode_event` を共通化し Pydantic/dataclass/辞書イベントの JSON 化・イベント名マッピングを一本化済み（担当: gpt-5-codex, 2025-02-15）。後続改修では `_extract_event` / `_coerce_payload` を経由させること。
- [x] tests/test_providers_openai.py / src/orch/providers/openai.py: OpenAI SSE の `data:` プレフィックス剥離と `[DONE]` センチネル終端処理を共通化済み（担当: gpt-5-codex, 2025-02-19）。追加入力時も `finalize_event` ロジックと usage/delta 正規化を保持すること。
- [ ] リリースノート整備: `VERSION` / `CHANGELOG.md` / `docs/release-notes.md` の同期と PR チェックリスト更新を継続管理すること。
