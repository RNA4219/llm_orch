# TASKS

- [x] README: OpenTelemetryメトリクス説明を更新（`ORCH_OTEL_METRICS_EXPORT` 追記、既知の制限を整理済み）。他タスクは同節の編集時にOTel設定例を保持すること。
- [x] src/orch/server.py: `_config_refresh_loop` のリロード処理を更新済み。後続タスクは同ロジック変更との重複に注意。最新plannerへの差し替え後は即座に再評価する設計。
- [x] tests/test_server_config_reload.py: plannerのリロード判定テストを追加済み。重複する検証ケースの再追加を避けること。
