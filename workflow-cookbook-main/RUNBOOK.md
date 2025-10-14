---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Runbook

## Environments

- Local / CI / Prod の差分（キー名だけ列挙）

## Execute

- 準備 → 実行 → 確認（最短手順）
- 例）
  - 準備: データ投入 / キャッシュ初期化
  - 実行: コマンド/ジョブ名
  - 確認: 出力の存在・件数・整合

## Observability

- ログ/メトリクスの確認点、失敗時の兆候
- インシデント発生時は docs/IN-YYYYMMDD-XXX.md に記録し、最新サンプル（[IN-20250115-001](docs/IN-20250115-001.md)）を参照して検知ログ・メトリクスの抜粋を添付

## Rollback / Retry

- どこまで戻すか、再実行条件
- インシデントサマリを更新後、該当PRの説明欄と本RUNBOOKの該当セクションにリンクを追加する
