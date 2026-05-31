# 技術文件 — 2026-04-27 Autopilot 清理 + E2E 整修

## 這次改了什麼（What Changed）

### 新增
- `README.md`（重寫）— 反映 6-step pipeline + 新 Excel schema + hetero 限制
- `docs/architecture/overview.md` — 給新工程師的全域導覽
- `docs/changelog/2026-04-27-autopilot-cleanup.md`
- `docs/usage/2026-04-27-autopilot-cleanup.md`
- `docs/technical/2026-04-27-autopilot-cleanup.md`（本檔）
- `.gitignore` 加入 `.ruff_cache/` `.swc/` `.omc/` `.design-ref/`

### 修改
- `backend/app/training/pipeline.py`（+5 / -1）— val_metrics 改從 val split 真實計算
- `frontend/lib/api.ts`（+3）— `ColumnInfo` 加上 `node_type` / `edge_type`
- `frontend/app/projects/[id]/upload/page.tsx`（+3 / -2）— schema 描述更正
- `frontend/app/projects/[id]/train/page.tsx`（+19 / -10）— hetero 過濾 backbone
- `frontend/app/projects/[id]/explore/page.tsx`（+22 / -16）— React key 加上 type suffix + 顯示 type
- `frontend/app/projects/[id]/predict/page.tsx`（+33 / -31）— regression 隱藏 confidence UI、列名改 Graph

### 刪除
- `system-feature-inventory.md`（root，14 KB）
- `backend/GNN_System_Architecture_Report.docx.md`（23 KB）
- `backend/demo_data/demo_multigraph_hetero.xlsx`（被 v2 取代）
- `frontend/refactor_report.yaml`（去年遺留）
- `scripts/quality-gate.sh`（broken：reference 已不存在的 src/）

## 為什麼這樣做（Why）

### 為什麼選擇「真實計算 val_metrics」而非「前端隱藏 VALIDATION 欄位」

兩個方案都能解決「val 跟 test 看起來一樣」的問題：

| 方案 | 優點 | 缺點 |
|------|------|------|
| 前端隱藏 VALIDATION | 改動最小（只動 evaluate/page.tsx） | 把 backend 行為的 bug 視覺化掩蓋；其他 client（API 直呼、未來的 CLI）仍會拿到誤導資料 |
| **後端真實計算（採用）** | 修到根本；report API 變誠實；前端不用條件渲染 | 多一次 forward pass，但 val 通常 ≤ 15% 資料量，耗時 < 1 秒 |

選後端方案。額外好處：之後 dashboard 想加「val 與 test 差距」當 generalization 指標時直接可用。

### 為什麼把 ALL_MODELS 拆成 HOMO / HETERO 而非由後端 reject

可以在 backend 收到 `models=['gcn']` + hetero dataset 時 raise 400，但：

- 拉長使用者迴路：要等到按下 Start Training 才知道不行
- 教育價值低：使用者看不到「為什麼這個模型不能用」
- 對齊一致：後端 HPO 內部已經會 skip GCN/GIN for hetero，前端也應該看得到

所以採取「前端 UI 直接不給選」的方案，背後仍保留 backend skip 作為防呆。

### 為什麼 React key 用 `${name}:${type}` 而非 `${type}.${name}`

純粹可讀性。`area_um2:cell` 看起來像「area_um2 屬於 cell」、`cell.area_um2` 像點操作（C-style）。
key 只在 React reconciliation 用，不會直接渲染給使用者。

### 為什麼 Predict 頁面選擇「動態渲染」而非分兩個檔案

把 Predict 拆成 `predict-classification.tsx` + `predict-regression.tsx` 也可以，但：

- 兩個任務 95% 邏輯相同（檔案上傳、模型選擇、Run inference、表格、graph viz）
- 拆開反而要維護兩份重複的 code
- 將來加 reranker / explainability 等 feature 時會需要在兩處重複改

所以選擇單一檔案 + 條件渲染（4 行 conditional + spread expression）。

## 怎麼運作（How It Works）

### val_metrics 計算的呼叫路徑

```
training.pipeline.run_pipeline(task_id, dataset_id, ...)
  ├── _prepare_hetero / _prepare_graph_homo / _prepare_node
  │     return train_items, val_items, test_items, ...
  │
  ├── run_hpo(train_items, val_items, ...)            ← 用 val_items 做早停
  │     return best_config
  │
  ├── trainer.fit(model, train_loader, val_loader)    ← 用 best_config 重新訓練
  │
  └── _predict_list / _predict_single
        ├── train_preds, train_y = predict(train_items)
        ├── val_preds, val_y = predict(val_items)     ← 新增這行
        └── test_preds, test_y = predict(test_items)

      _regression_metrics 或 _classification_metrics 各算三組

      report = {
          "train_metrics": train_metrics,
          "val_metrics":  val_metrics,                ← 從 dict(test_metrics) 改成真實
          "test_metrics": test_metrics,
          ...
      }
```

對 graph-level task，`val_items` / `test_items` 是不同的 list，所以三組指標會明顯不同。
對 single-graph node-task，`_prepare_node()` 內 `val_items = test_items`，所以結果仍會相同 — 屬於資料切分本身的限制，已記在 caveats。

### React duplicate key 的根因

`exploreData.columns` 對異質圖會回傳像這樣的 list：

```json
[
  { "name": "area_um2", "node_type": "cell", "dtype": "numeric", ... },
  { "name": "area_um2", "node_type": "pin",  "dtype": "numeric", ... },
  { "name": "cap_ff",   "node_type": "pin",  "dtype": "categorical", ... },
  { "name": "cap_ff",   "node_type": "net",  "dtype": "categorical", ... },
  ...
]
```

`columns.map(col => <Row key={col.name}/>)` 就會出現 `area_um2` 與 `cap_ff` 各兩次重複的 key。
React 的處理是「無聲合併」—— 第二筆寫入覆蓋第一筆，所以實際上**有筆 row 沒被渲染**。

修正後 key 變成 `node-area_um2:cell`、`node-area_um2:pin`，每筆都唯一，schema 表會看到 8 行而不是 6 行。

correlation column 那邊我採用「dedupe by name」（用 `Map`）—— 因為 correlation 是按 column name 做的，如果同名 column 出現多次只會選任一個，沒必要勾兩次 checkbox。

### Predict 條件渲染

```tsx
const isRegression = summary?.taskType?.endsWith('regression') ?? false;
const isGraphTask  = summary?.taskType?.startsWith('graph')   ?? false;
const itemLabel    = isGraphTask ? 'Graph' : 'Node';

// 中央 KPI 卡
<Row>
  {!isRegression && <>
    <Col span={8}><AvgConfidenceCard /></Col>
    <Col span={8}><NeedsReviewCard /></Col>
  </>}
  <Col span={isRegression ? 24 : 8}><PredictionsCard /></Col>
</Row>

// 表格欄位
columns={[
  { title: itemLabel, ... },
  { title: 'True', ... },
  { title: 'Predicted', ... },
  ...(isRegression ? [] : [{ title: 'Confidence', ... }]),
]}

// 右側欄
<Col xs={24} md={6} style={{ display: isRegression ? 'none' : undefined }}>
  ... Confidence histogram + Needs review queue ...
</Col>
```

值得一提：右欄用 `display: none` 而非 conditional return，這樣 React 樹結構維持一致，
切換 task type 不會觸發 unmount/remount 副作用（雖然目前沒有，但保險）。

## 怎麼使用（Usage）

### 同質圖場景

```bash
# 啟動兩個 server
cd backend && .venv/Scripts/python -m uvicorn app.main:app --reload --port 8000 &
cd frontend && npm run dev &

# 透過 API 跑（也可以用 UI）
PROJECT=$(curl -s -X POST http://localhost:8000/api/v1/projects/ \
    -H 'Content-Type: application/json' \
    -d '{"name":"test","tags":[]}' | jq -r .project_id)

curl -X POST "http://localhost:8000/api/v1/projects/$PROJECT/load-demo-excel?demo_id=multigraph_homo"

curl -X POST "http://localhost:8000/api/v1/projects/$PROJECT/confirm" \
    -H 'Content-Type: application/json' \
    -d '{"task_type":"graph_regression","label_column":"target_delay"}'

curl -X POST "http://localhost:8000/api/v1/projects/$PROJECT/train" \
    -H 'Content-Type: application/json' \
    -d '{"models":["gcn","gat","sage","gin","mlp"],"n_trials":50}'   # 全部 5 個 backbone OK

# 等訓練完成後
curl "http://localhost:8000/api/v1/projects/$PROJECT/report" | jq '.train_metrics, .val_metrics, .test_metrics'
# 會看到三組不同的 metrics
```

### 異質圖場景

跟上面一樣，但 demo_id 改成 `multigraph_hetero`、models 改成 `["gat","sage","mlp"]`（送 gcn/gin 也不會壞，後端會自動 skip 但浪費 trial 額度）。

### 確認 Predict 頁面 regression 行為

訓練完之後直接訪問：
`http://localhost:3000/projects/<PROJECT>/predict`

確認：
- 沒有 "Avg confidence" / "Needs review" / "Confidence histogram"
- 表格表頭：Graph / True / Predicted（**沒有** Confidence 欄）
- 右側欄整個摺疊（display: none）

## 注意事項（Caveats）

1. **單圖 node-task 的 val 仍 == test**：`_prepare_node()` 沒有切 val。要修需要在那層補 split 邏輯。
2. **antd v6 deprecation warnings**：`strokeWidth`、`destroyOnClose` 都會繼續出現直到 antd 官方 codemod 一次處理。
3. **Predict 頁面是 stand-in**：目前複用最後一次 training 的 `node_predictions`。新加的 `taskType` 條件邏輯已經就位，等 `POST /predict` 真正實作之後直接受惠。
4. **共用 feature 的 missing 計算**：對 hetero，每個 (feature, type) 各自計算 missing。要看「整體 missing」需要在前端再 reduce 一次。
5. **`val_metrics` 是 Optional 欄位**：`api_models.py` 的 `Report.val_metrics` 仍是 `Optional[SplitMetrics]`，意思是未來想關掉 val 計算（例如 user 顯式 skip）也還能 round-trip。

## 相關檔案路徑速查

| 議題 | 主要檔案 |
|------|----------|
| Backend metrics 計算 | `backend/app/training/pipeline.py:359-410` |
| Hetero backbone filter | `frontend/app/projects/[id]/train/page.tsx:23-27,46,63-73` |
| Explore 共用 feature key | `frontend/app/projects/[id]/explore/page.tsx:191-228` |
| ColumnInfo 型別 | `frontend/lib/api.ts:121-130` |
| Predict 條件渲染 | `frontend/app/projects/[id]/predict/page.tsx:133-144,309-340,360-395` |
| Upload 頁面文字 | `frontend/app/projects/[id]/upload/page.tsx:177-180` |
| Excel schema 規範 | `backend/app/data/excel_spec.py` + `docs/technical/hetero-unified-sheet-and-hpo.md` |
