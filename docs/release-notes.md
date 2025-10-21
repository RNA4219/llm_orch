# リリースノート運用ガイド

Conventional Commits に基づき、リリース前に以下の手順で CHANGELOG と VERSION を更新します。

1. `VERSION` の SemVer をインクリメントし、`MAJOR.MINOR.PATCH` 形式を維持する。
2. `CHANGELOG.md` に `## vMAJOR.MINOR.PATCH - YYYY-MM-DD` 見出しを追加し、主な変更点を箇条書きで記載する。
3. コミットメッセージは Conventional Commits (`feat:`, `fix:`, `chore:` など) を遵守する。
4. リリースノートを PR 説明と同期し、必要に応じて docs 配下の関連資料も更新する。

テストとリリース工程は `tools/ci/smoke.sh` を含む既存の CI スクリプトで検証し、失敗時は速やかに修正すること。
