# Changelog

すべての notable な変更はこのファイルに記録されます。

## v0.2.0 - 2026-03-09

- OpenTelemetry only / both モードで `MetricsLogger` が安定してメトリクスを記録できるよう、専用 `MeterProvider` の管理方法を修正。
- `workflow-cookbook-main/tools/ci/check_governance_gate.py` のポリシー読込を改善し、Windows 環境で生成された非 UTF-8 ファイルでも forbidden path 判定が落ちないよう修正。
- README 冒頭の状態説明を現状実装に合わせて更新し、機能凍結リリースとしての位置づけを明確化。
- `docs/release-notes.md` と `VERSION` を `v0.2.0` に同期し、最終仕上げのリリース手順を整理。

## v0.1.1 - 2024-10-05

- `requirements.txt` の `ruff` / `mypy` の重複行を整理し、CI 用ツールとして単一バージョン（`ruff==0.6.9`, `mypy==1.13.0`）に固定。

## v0.1.0 - 2024-01-01

- 初期リリースのプレースホルダ。

