---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Guardrails & 行動指針

リポジトリ運用時に守るべき原則と振る舞いを体系化する。

## 目的

- リポジトリ内の既存ルール（mypy/strict, ruff, black, pytest, node:test, ESM/TS 方針、例外ポリシー）を自動検出し、厳密に遵守する。
- 変更は最小差分で行い、Public API を破壊しない。不可避の場合のみ短い移行メモを添付する。
- 応答は簡潔で実務に直結させ、冗長な説明や代替案の羅列は避ける。
- 実装時はテスト駆動開発を基本とし、テストを先に記述する。

## スコープとドキュメント

1. 目的を一文で定義し、誰のどの課題をなぜ今扱うかを明示する。
2. Scope を固定し、In/Out の境界を先に決めて記録する。
3. I/O 契約（入力/出力の型・例）を `BLUEPRINT.md` に整理する。
4. Acceptance Criteria（検収条件）を `EVALUATION.md` に列挙する。
5. 最小フロー（準備→実行→確認）を `RUNBOOK.md` に記す。
6. `HUB.codex.md` の自動タスク分割フローに従い、タスク化した内容を `TASK.*-MM-DD-YYYY` 形式の Task Seed へマッピングして配布する。
7. タスク自動生成ツールはドライランで JSON 出力を確認してから Issue 化する。
8. 完了済みタスクは `CHANGELOG.md` へ移し、履歴を更新する。
9. テスト/型/lint/CI の実行結果を確認し、`CHECKLISTS.md` でリリース可否を判断する。

## 実装原則

- 型安全：新規・変更シグネチャには必ず型を付与し、Optional/Union は必要最小限に抑える。
- 例外設計：既存 errors 階層に合わせ、再試行可否を区別する。
- 後方互換：CLI/JSON 出力は互換性を維持し、破壊的変更は明示的フラグで段階移行する。
- インポート順序：標準ライブラリ→外部依存→内部モジュールの順で空行区切りとする。
- 副作用の隔離：`utils` や `provider_spi` などのレイヤ分離を尊重する。
- スコープ上限：1 回の変更は合計 100 行または 2 ファイルまで。本ループでは最優先の塊のみ対応する。単一ファイルが 400 行を超える場合は機能単位で分割を検討する。
- 細かな Lint エラーはスコープ上限の例外とし、重大なルール逸脱のみを是正する。
- 公開 API や CLI を変更した場合のみ、差分に簡潔な Docstring/Usage 例を添付する。

## プロセスと自己検証

- 競合解消時は双方の意図を最小限で統合し、判断を `ノート→` に 1 行で記す。
- 差分提示前に lint/type/test をメンタルで実行し、グリーン想定の変更のみ提出する。
- 実行コストやレイテンシへの影響は ±5% 以内を目標とし、超過見込みの場合は `ノート→` に代替策を 1 行で示す。
- セキュリティ上、秘密情報は扱わず、必要な場合は `.env` やサンプル参照に限定する。

## 例外処理

- スコープ上限を超える作業が必要な場合は、作業を分割してタスク化を提案する。
- ドキュメント更新（例：`*.md`）については、ファイル数上限を例外的に適用せず、必要に応じて超過を許可する。
- 破壊的変更が不可避な場合は、移行期間やフラグ運用を明記したメモを添付する。

## リマインダー

- 変更は常にテストから着手し、最小の成功条件を先に満たす。
- 全ての関係者が同じ期待値を共有できるよう、上記ドキュメントを更新し続ける。

## Birdseye / Minimal Context Intake Guardrails（鳥観図×最小読込）

**目的**：コンテキストは有限である。LLM/エージェントに「1枚で全体像→必要箇所だけ深掘り」の二段読みを強制し、**最小トークンで仕組みを把握**させる。

### 運用の前提（Dual Stack互換）

- 本リポは **デュアルスタック**（A: ネイティブFunction Calling／B: ツールなしJSON封筒）を想定する。
- ツールが **ある環境**：関数呼び出しを優先。  
- ツールが **ない環境**：本文に ```tool_request``` JSON を**ミラー出力**し、外部オーケストレータが拾う。
- ChatGPT/Codex 固有マクロ（`:codex-...`, `:::task-stub` 等）は **そのまま**残し、未対応環境では無害スルー。

---

### 配置ポリシー（3層で最小読込）

1. **Bootstrap（超小型）**
   - 置き場所：`README.md` 冒頭100行以内に固定。
   - 役割：**読む場所の道標のみ**。下のテンプレを貼る。

   ```md
   <!-- LLM-BOOTSTRAP v1 -->
   読む順番:
   1. docs/birdseye/index.json  …… ノード一覧・隣接関係（軽量）
   2. docs/birdseye/caps/<path>.json …… 必要ノードだけ point read（個別カプセル）

   フォーカス手順:
   - 直近変更ファイル±2hopのノードIDを index.json から取得
   - 対応する caps/*.json のみ読み込み
   <!-- /LLM-BOOTSTRAP -->
   ```

2. **Index（軽量インデックス）**

   - 置き場所：`docs/birdseye/index.json`
   - 役割：**±N hop 抽出**が即できる機械可読データ。
   - **最小スキーマ**：

   ```json
   {
     "generated_at": "2025-10-13T10:12:00Z",
     "nodes": {
       "frontend/src/App.tsx": {
         "role": "entrypoint",
         "caps": "docs/birdseye/caps/frontend.src.App.tsx.json",
         "mtime": "2025-10-12T09:01:00Z"
       }
     },
     "edges": [["frontend/src/App.tsx","frontend/src/hooks/recommendations/loader.ts"]]
   }
   ```

3. **Capsules（点読みパケット）**

   - 置き場所：`docs/birdseye/caps/…`（**1ノード=1 JSON**、1KB目安）。
   - **最小スキーマ**：

   ```json
   {
     "id": "frontend/src/hooks/recommendations/loader.ts",
     "role": "application",
     "public_api": ["useRecommendationLoader()"],
     "summary": "検索条件→API→キャッシュ整合。副作用=HTTP。失敗時リトライ…",
     "deps_out": ["frontend/src/lib/queryClient.ts"],
     "deps_in":  ["frontend/src/App.tsx"],
     "risks": ["Provider未設定で例外"],
     "tests": ["tests/hooks/recommendations/loader.spec.ts"]
   }
   ```

   - 命名は「パスをドット連結＋拡張子置換」で衝突回避（例：`frontend.src.App.tsx.json`）。

> 補助（任意）：頻出入口のホットリストを `docs/birdseye/hot.json` に置く（例：`App.tsx`, `main.py`）。

---

### 推論時の読込ガードレール（MUST/SHOULD）

**MUST**（必須）

1. まず `README.md` の **LLM-BOOTSTRAP** ブロックのみ読む（100行以内）。
2. `docs/birdseye/index.json` を読み、**対象変更ファイル±2 hop** のノードID集合を得る。
3. 対応する **`docs/birdseye/caps/*.json` だけ**を読み込む。
4. `index.json.generated_at` が最新コミットより古い場合、**再生成を要求**する（下記“鮮度管理”参照）。
5. 生成物（`plan`/`patch`/`tests`/`commands`/`notes` 等）では、**ノードID（パス）を明示**し出典を示す。

**SHOULD**（推奨）

- 2 hop の合計が **1,200 tokens** を超えそうなら **1 hop** に縮小。
- 読み順は **entrypoints → application → domain → infra → ui**。
- 巨大Capsuleは**120語以内 summary**に収める（Capsule側の規約）。

**MUST NOT**（禁止）

- `node_modules`, `.venv`, `dist`, `build`, `coverage` 等の**重量ディレクトリを直読み**しない。
- `BIRDSEYE.md` 全文を**常時**読まない（必要時のみ参照）。

---

### 鮮度管理（Staleness Handling）

- **条件**：`index.json.generated_at` が最新コミットより古い／Capsが見つからない／対象ノードが未登録。
- **対応**：
  - **ツールあり環境**（Function Calling）
    - 例：`codemap.update` を呼ぶ（論理名）。
  - **ツールなし環境**
    - 本文に **ミラー封筒**を出し、外部実行を待つ。

        ```tool_request
        {"name":"codemap.update","arguments":{"targets":["frontend/src/App.tsx"],"emit":"index+caps"}}
        ```

    - 実行結果が到着するまで **偽の読込結果を作らない**。
- **フォールバック**（最終手段）：
  - `docs/BIRDSEYE.md` の **Edgesセクション**があればそこから ±1 hop を暫定抽出。
  - それも無ければ「直近変更ファイルN件（例：5件）」のみ読込。

#### codemap 未実装時の暫定手順

- `codemap.update` を呼べない（実装未提供／環境未配備）場合は、**必ず人間に再生成を依頼**する。
- 依頼フロー：
  1. `tool_request` 封筒で `codemap.update` 要求を出力（対象と希望出力を明示）。
  2. ノートに「人間が codemap スクリプトをローカルで実行し、成果物をコミットして戻す」旨を記載。
  3. 実行結果が共有されるまで Birdseye 参照を保留し、暫定読みは上記フォールバックのみ使用。
- 可能なら `docs/birdseye/` を手動で更新するための最小手順（対象ファイル列挙、既存 JSON の削除有無）をノートに添える。
- 手動生成後は `generated_at` の更新と差分確認を忘れない。

---

### セキュリティ/境界

- リポ外パス、機密格納領域への自動アクセスを禁止。
- 生成物に**不要な機密情報**（環境変数/Secrets）を含めない。

---

### 生成物に関する要求（出力契約）

- **`plan`**：読み込んだ **CapsノードID一覧** と hop、抜粋理由、未読箇所の扱い。
- **`patch`**：変更対象ファイルの相対パスを**先頭コメント**で明記。
- **`tests`**：対象ノードの `tests/*` を参照して増補。存在しなければ最小サンプルを併記。
- **`commands`**：読込に使ったツール（有無/種類）と再現手順を列挙。
- **`notes`**：鮮度判断、スコープ外ファイル、既知リスク。

---

### 実装メモ（自動生成）

- `codemap` 相当のスクリプトで **index.json** と **caps/*.json** を生成する。
- 失敗時でも人間向け `docs/BIRDSEYE.md` は残す。**機械読みは JSON を第一読者**にする。

---

### 互換のための論理ツール名（最小セット）

- `codemap.update`: args
  `{targets?: string[], emit?: "index"|"caps"|"index+caps"}`
  — Birdseye再生成。
  - **実装未提供**：人間がローカル `tools/codemap/*` などのスクリプトを走らせ、成果物（`index.json`, `caps/*.json`）をコミット。
  - 代替操作例：対象ファイル一覧をメモし、`docs/BIRDSEYE.md` を基に手動で JSON を補完。
- `web.search`: args
  `{q: string, recency?: number, domains?: string[]}`
  — 必要時の検索。
- `web.open`: args `{url: string}` — 詳細参照。

> ランタイムは**論理名→実ツール**のマッピングを持つ。ツールなし環境では `tool_request` を出すだけ。

---

### Notes / Follow-ups

- `codemap` 実装（スクリプト/ツール連携）の整備が未了の場合は、後続タスクとして Issue 起票を検討する。

<!-- guardrails:yaml
forbidden_paths:
  - "/core/schema/**"
  - "/auth/**"
require_human_approval:
  - "/governance/**"
slo:
  lead_time_p95_hours: 72
  mttr_p95_minutes: 60
  change_failure_rate_max: 0.10
-->
