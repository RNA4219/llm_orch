---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Blueprint

## 1. Problem Statement

- 誰の何の課題を、なぜ今解くか

## 2. Scope

- In:
- Out:

## 3. Constraints / Assumptions

- 時間/コスト/依存/互換性の前提

## 4. I/O Contract

- Input: 形式・例
- Output: 形式・例

## 5. Minimal Flow

```mermaid
flowchart LR
  A[Input] --> B[Process]
  B --> C[Validate]
  C -->|OK| D[Publish]
  C -->|NG| B
```

## 6. Interfaces

- CLI/API/Files（エンドポイント/パス命名）
