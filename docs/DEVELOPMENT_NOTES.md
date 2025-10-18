# 開発ノート

## Birdseye `generated_at` 更新手順
1. `docs/birdseye/index.json` と必要な `docs/birdseye/caps/*.json` を再生成または手動編集する。
2. 生成が完了したら `generated_at` を UTC ISO8601 (`YYYY-MM-DDTHH:MM:SSZ`) で更新する。
3. 差分を確認し、対応する変更をコミットする。
