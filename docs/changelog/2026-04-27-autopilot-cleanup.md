# 2026-04-27 — Autopilot 清理 + E2E 整修

Branch：`chore/autopilot-cleanup-and-e2e-fixes` → 推送至 origin
模式：`/autopilot`（清理 + E2E 自動化）

## 改了什麼

### 移除（清理過期內容）
- `system-feature-inventory.md`（top-level，14 KB，停留在 Mar 24，與後續所有重構不一致）
- `backend/GNN_System_Architecture_Report.docx.md`（23 KB，2 月底的早期架構報告，無人引用）
- `frontend/refactor_report.yaml`（Dec 8，去年某次重構的草稿，零引用）
- `backend/demo_data/demo_multigraph_hetero.xlsx`（被 v2 取代且僅在 changelog 歷史記錄出現）
- `scripts/quality-gate.sh`（root，非追蹤檔，腳本內 reference `frontend/src/` 已不存在）
- `backend/.coverage`、`__pycache__`、`.pytest_cache`、各 `.ruff_cache`（local cache，非追蹤）
- `frontend/out/`（jest 報告輸出，會自動再生）

### `.gitignore` 擴充
新增 ruff / swc / OMC / design-ref 條目，避免之後又把工作狀態誤 commit 進去。

### 修正（E2E 走查發現的 5 個 bug）
1. **Upload 頁面文字過時** — 文案描述 `Node_*` / `Edge_*` / `Graph_*` per-type sheet，但 2026-04-24 schema 簡化後是單一 `Node` / `Edge` / `Graph` sheet + 可選 `Type` 欄位。改成正確描述。
2. **Train 頁面異質圖仍顯示 GCN / GIN** — 後端對 hetero 會自動跳過這兩個 backbone（`to_hetero` 不支援），但前端把使用者的選擇照單全收，造成沒有訊息的「悄悄略過」。改成根據 `dataset_summary.is_heterogeneous` 動態切換 model 列表（hetero 只剩 GAT / SAGE / MLP）。
3. **Explore 頁面 React duplicate key 警告** — 異質圖支援共用 feature 後，同名 feature 會在多個 node_type 各出現一次（例：`area_um2` on cell + pin），但 React `key` 只用 `col.name`。改用 `${name}:${type}` 並在顯示名稱附上 type 標註。
4. **`val_metrics` 是 `test_metrics` 的拷貝** — `pipeline.py:395` 直接 `dict(test_metrics)`，導致 Evaluate 頁面的 VALIDATION 與 TEST 欄位完全相同。改成對 val split 真實做一次 prediction 並計算指標。
5. **Predict 頁面對 regression 仍顯示 confidence UI** — Avg confidence、Needs review、Confidence histogram、表格 Confidence 欄位、Node→Graph 列名都是 classification 概念。改成依 `taskType` 條件渲染、graph-task 改用 "Graph" 列名。

### 文件
- 重寫 `README.md`（先前停留在 Mar 24，許多事實已過期）
- 新增 `docs/architecture/overview.md`（給新工程師的入門地圖）
- 新增本檔（changelog）+ 對應的 `docs/usage/` 與 `docs/technical/` 文件

## 為什麼

- 過去三週 (2026-04-08 ~ 04-26) 連續做了 schema 簡化、Excel 統一格式、HPO 統一、hetero 加回、train UI 簡化等大量重構；累積的 stale 檔案、過時 UI 文案、未隨之更新的型別需要一次性整理。
- E2E 走查證明所有重構之後仍有沒被 unit test 涵蓋到的 UX 問題（duplicate key warning、模型選單錯誤、metrics 假象），這些都會直接傷害 demo 信任度。
- 文件停留在 Mar 24，新加入的工程師（或自己過幾個月）會被誤導；需要重設 source of truth。

## 怎麼用 / 怎麼驗證

- 拉 branch：`git checkout chore/autopilot-cleanup-and-e2e-fixes`
- 跑 `cd backend && .venv/Scripts/python -m pytest -q` → **50 passed**
- 跑 `cd frontend && npm test` → **46 passed**
- 開 Next.js dev → 建 project → 載入 `Multi-Graph Heterogeneous` demo → 進 Explore（看 console 沒有 duplicate key warning）→ Confirm → Train（model 列表只有 GAT / SAGE / MLP）→ Evaluate（VALIDATION ≠ TEST）→ Predict（沒有 Confidence 卡片、表頭是 Graph）

## 注意事項

- `val_metrics` 在「單圖 node-task」場景下仍會與 `test_metrics` 相等 — 因為該路徑 `val_items = test_items`（沒有獨立 val 切分）。屬於資料切分本身的限制，未來若要修要在 `_prepare_node()` 加 val split。
- 未處理的 deprecation warnings：`[antd: Progress] strokeWidth`、`[antd: Modal] destroyOnClose` — antd v5 → v6 的 migration warning，等批次升級時統一處理。
- README 的 Excel schema 範例僅涵蓋同質與異質的單純情況；複雜混合情境（例：edge 也有共用 feature）請查 `docs/technical/hetero-unified-sheet-and-hpo.md`。

## 相關 commit

- `chore(repo): remove stale top-level docs and obsolete demo asset`
- `fix(backend): compute val_metrics from val split, not test copy`
- `fix(frontend): address 4 UX issues found via end-to-end walkthrough`
