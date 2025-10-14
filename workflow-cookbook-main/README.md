---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Workflow Cookbook / Codex Task Kit

This repo defines QA/Governance-first workflows (not application code).
AI agents implement changes under these policies with acceptance tests and
canary rules.

任意のリポジトリに貼るだけで、**仕様→実装→検収**まで一貫して回せるMD群。

- 人間にもエージェント（Codex等）にも読ませやすい最小フォーマット
- 言語・技術スタック非依存（存在するコマンドだけ使う）

## 使い方（最短）

1. これらのMDをリポジトリ直下に配置
2. `BLUEPRINT.md` で要件と制約を1ページに集約
3. 実行手順は `RUNBOOK.md`、評価基準は `EVALUATION.md` に記述
4. タスクごとに `TASK.codex.md` を複製して内容を埋め、エージェントに渡す
5. リリースは `CHECKLISTS.md` をなぞり、差分は `CHANGELOG.md` に追記

<!-- markdownlint-disable MD013 -->
![lint](https://github.com/RNA4219/workflow-cookbook/actions/workflows/markdown.yml/badge.svg)
![links](https://github.com/RNA4219/workflow-cookbook/actions/workflows/links.yml/badge.svg)
![lead_time_p95_hours](https://img.shields.io/badge/lead__time__p95__hours-72h-blue)
![mttr_p95_minutes](https://img.shields.io/badge/mttr__p95__minutes-60m-blue)
![change_failure_rate_max](https://img.shields.io/badge/change__failure__rate__max-0.10-blue)
<!-- markdownlint-enable MD013 -->

> バッジ値は `governance/policy.yaml` の `slo` と同期。更新時は同ファイルの値を修正し、上記3つのバッジ表示を揃える。

### Commit message guide

- feat: 〜 を追加
- fix: 〜 を修正
- chore/docs: 〜 を整備
- semver:major/minor/patch ラベルでリリース自動分類
