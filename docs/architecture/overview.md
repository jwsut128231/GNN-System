# GraphX.AI 架構總覽 — 給新加入的開發者

最後更新：2026-04-27

這份文件是給「第一次接觸這個專案的工程師」的白話入門。讀完之後你應該能：

- 說出這個系統在做什麼、要解決什麼問題
- 知道 frontend / backend / ML pipeline 怎麼分工、為什麼這樣分
- 看著程式碼資料夾大概知道哪個檔案做什麼
- 理解 GNN AutoML 背後在跑什麼（不需要深度學習背景）
- 自己跑一次完整的 E2E 流程
- 知道接下來該往哪裡看才能繼續開發

> 想看 API 細節去查 `http://localhost:8000/docs`；想看每個功能的歷史去翻 `docs/changelog/`；想看深度技術解析去翻 `docs/technical/`。本文只負責「先建立心智模型」。

---

## 1. 這個專案在做什麼？

GraphX.AI 是一個**圖神經網路（GNN）的 AutoML 平台**，專門服務 IC layout / EDA 工程師。
使用者把電路或佈局資料整理成一份 Excel，丟進系統，系統會：

1. 解析 Excel 把它變成一張或多張 PyTorch Geometric 的圖
2. 自動跑 hyper-parameter 搜尋（Optuna），在多個 GNN 模型間挑出表現最好的那組
3. 用網頁呈現訓練過程、訓練/驗證/測試指標、重要超參數、預測結果

對使用者來說最大的價值：**不用寫一行 ML 程式碼**就能拿到一個訓練好的 GNN，並且可以在介面上做資料探索、找問題。

兩種典型情境：

- **同質圖（homogeneous）**：所有節點是同一種東西（例如 cell），所有邊也是同一種關係。簡單、直觀。
- **異質圖（heterogeneous）**：節點分成多種類型（cell / pin / net），邊也分多種關係（cell→pin、pin→net）。比較貼近真實 IC layout。

---

## 2. 為什麼選這個技術組合？

| 層 | 技術 | 為什麼 |
|---|------|--------|
| 前端 | Next.js 16 + React 19 | App Router 對巢狀路由、串流 UI 很友善；ML 結果頁面動態複雜，SSR 不是重點。 |
| UI 元件 | Ant Design 5 | 表單、資料表、Modal、Drawer 都現成；中文 / 工程介面常見的元件齊全。 |
| 圖表 | Recharts | 訓練曲線、殘差圖、長條圖夠用就好。 |
| 後端 | FastAPI + Pydantic v2 | Python 跟 ML 生態相容；Pydantic 強型別 + 自動 OpenAPI doc。 |
| ML | PyTorch + PyG + Lightning | PyG 提供 GNN 模型與 hetero 工具；Lightning 接管 trainer / callbacks 樣板。 |
| AutoML | Optuna | 跨模型 + 跨超參數的搜尋只要一個 study 就能做；early-pruning 也內建。 |
| 儲存 | 純記憶體 | 為了「下載即跑」的 demo 體驗，故意不接 DB。重啟就清空，這是已知 trade-off。 |
| 認證 | NextAuth + Keycloak | 走標準 OIDC，企業內部 SSO 容易接；本機 dev 可關掉。 |

刻意 **沒** 用的東西：

- **Redis / Celery / MongoDB**：早期 branch 試過，但和「零依賴」目標衝突，最終沒合進主線。
- **後端 ORM**：所有 state 都在 `app/core/store.py` 的 in-memory dict；reload uvicorn 就會清空。
- **WebSocket**：訓練進度用前端 polling（每 2 秒打一次 `/status`），夠用且更穩定。

---

## 3. 端到端資料流：一次上傳到預測

```
使用者 ─▶ Next.js dev server (3000)
              │
              │  fetch（純 REST，無 GraphQL）
              ▼
         FastAPI app (8000)
              │
              ├── routers/projects.py     ← 入口路由
              ├── data/excel_ingestion.py ← 解析 .xlsx 變成 dict
              ├── data/pyg_conversion.py  ← dict → PyG Data / HeteroData
              ├── training/pipeline.py    ← Optuna HPO + Lightning fit
              └── core/store.py           ← 把所有結果塞進記憶體
```

**Step by step：**

1. **Create Project**：前端開 modal，POST `/projects/`，後端在記憶體建一個空 project。
2. **Upload Excel** 或 **Load Demo**：
   - Upload → POST `/projects/{id}/upload-excel`（multipart）
   - Demo  → POST `/projects/{id}/load-demo-excel?demo_id=...`（複製 `backend/demo_data/*.xlsx`）
   - 後端呼叫 `excel_ingestion.parse_excel_file()`：
     - 讀 `Parameter` sheet → 知道哪些欄位是 X / Y、屬於哪個 Level、屬於哪個 Type
     - 讀 `Node` / `Edge` / `Graph` sheet → 把資料切到對應的 graph_id 裡
     - 偵測同質 vs 異質（看有沒有 Type 欄位）
   - 結果是一個 `dataset` dict，存在 `store`，附上 `dataset_id`
3. **Explore**：GET `/projects/{id}/explore`
   - 後端統計每個 column 的 dtype、缺失率、唯一值數
   - 算 numeric column 的相關係數矩陣
   - 對異質圖，每個 (feature, node_type) 組合都會產生一個 ColumnInfo（所以同名 feature 會出現多次）
   - 回傳 sample graph（給前端 GraphPreview 畫）
4. **Confirm**：POST `/projects/{id}/confirm` 含 `task_type` 與 `label_column`
   - 從 `dataset` 把 X / Y 切出來，跑 imputation（有缺值就填）
   - 標準化 numeric 特徵、one-hot encoding categorical
   - 切 train / val / test（預設 70 / 15 / 15）
   - 把處理結果再寫回 `store`
5. **Train**：POST `/projects/{id}/train` 含 `models` 與 `n_trials`
   - 啟動背景 task：`training/pipeline.py:run_pipeline()`
     - 建立 Optuna study
     - 每個 trial：選一個 backbone (GAT/SAGE/...) + 隨機超參數 → Lightning `Trainer.fit()`
     - 觀察 val_loss → early-stop / pruning
     - 全部 trial 跑完之後挑出 best_config
   - 用 best_config 重新訓練一個「正式版」模型，存 `.pt` 到 `backend/storage/models/`
   - 計算 train / val / test 的 metrics（regression: MSE/MAE/R²/MAPE；classification: Accuracy/F1/Precision/Recall）
   - 一邊跑一邊 update task status，前端用 polling 抓進度
6. **Evaluate**：GET `/projects/{id}/report`
   - 拿到完整的 report：三組 metrics、history（每個 epoch 的 loss）、residual_data（每筆預測的誤差）、leaderboard（top-N trial）、best_config
   - 前端畫表格、殘差圖（y=0 參考線）、訓練曲線
7. **Predict**：GET `/projects/{id}/models` → 選一個 → 拖一個檔案進去（目前重用 report 的 node_predictions 做 stand-in）
   - graph-level 任務的 predictions 其實是「整張圖一個值」；前端會把列名改成 `Graph` 而不是 `Node`
   - classification 才會顯示 confidence 相關 UI

> 想看更深的 hetero pipeline 細節：[`docs/technical/hetero-unified-sheet-and-hpo.md`](../technical/hetero-unified-sheet-and-hpo.md)

---

## 4. 重要檔案速查表

挑「會被修改頻率高」或「動了就會壞」的檔案：

### Backend

| 檔案 | 你會什麼時候動它 |
|------|------------------|
| `backend/app/main.py` | 加 router、加 middleware、改 CORS。 |
| `backend/app/core/config.py` | 加環境變數、改 split ratio、改 max epochs。 |
| `backend/app/core/store.py` | 加新的記憶體欄位、debug state。 |
| `backend/app/data/excel_spec.py` | 改 Excel schema（欄位定義 / 解析規則）。 |
| `backend/app/data/excel_ingestion.py` | 改 Excel → dict 的 parser。 |
| `backend/app/data/pyg_conversion.py` | 改 dict → PyG Data 的轉換（標準化、缺值處理） |
| `backend/app/models/*.py` | 改個別 GNN backbone 或加新的。 |
| `backend/app/models/hetero_wrapper.py` | 改 to_hetero 包裝邏輯（共享層 vs 不共享、metadata 取得）。 |
| `backend/app/training/pipeline.py` | **核心訓練邏輯**：HPO loop、metrics 計算、model save。動之前先看 callbacks 怎麼回呼。 |
| `backend/app/training/hpo.py` | Optuna study 設定、search space。 |
| `backend/app/routers/projects.py` | 新增 / 修改 API endpoint。 |
| `backend/app/schemas/api_models.py` | 改 request / response 型別 → 前端 ts 也要跟著改。 |
| `backend/scripts/generate_excel_demos.py` | 改 demo 資料；改完跑一次 `python -m scripts.generate_excel_demos` 重新產出。 |
| `backend/tests/test_*.py` | 寫 / 改 unit test。改完用 `.venv/Scripts/python -m pytest -q` 跑全部。 |

### Frontend

| 檔案 | 你會什麼時候動它 |
|------|------------------|
| `frontend/lib/api.ts` | **API 介面 + 全部 TS 型別**。後端動了 schema 就要改這裡。 |
| `frontend/lib/sanitize.ts` | URL params 清洗（防 path injection）。 |
| `frontend/contexts/ProjectContext.tsx` | 全域 project list / current project state。 |
| `frontend/contexts/ColorModeContext.tsx` | 暗色模式、主題 token。 |
| `frontend/theme/` | Ant Design token 客製化。 |
| `frontend/app/dashboard/page.tsx` | Project 列表 / 統計 KPI / 建立 project modal。 |
| `frontend/app/projects/[id]/upload/page.tsx` | Step 1 — 上傳 / 載入 demo。 |
| `frontend/app/projects/[id]/explore/page.tsx` | Step 2 — 資料分析 / 缺值 / label 確認。 |
| `frontend/app/projects/[id]/train/page.tsx` | Step 3 — 模型選擇、Optuna 試驗數、訓練監控。 |
| `frontend/app/projects/[id]/evaluate/page.tsx` | Step 4 — Metrics / 殘差 / leaderboard。 |
| `frontend/app/projects/[id]/predict/page.tsx` | Step 5 — 推論（目前 stand-in）。 |
| `frontend/app/projects/[id]/models/page.tsx` | Model Registry。 |
| `frontend/components/GraphPreview.tsx` | force-directed 圖視覺化（>200 nodes 自動 fallback list）。 |
| `frontend/components/AppHeader.tsx` | 上方 nav + pipeline stepper。 |
| `frontend/components/PredictionTable.tsx` | 預測結果表（跟 Predict 頁面共用）。 |
| `frontend/jest.config.ts` | 測試設定（門檻 60% coverage）。 |

---

## 5. GNN AutoML 的核心原理（白話版）

如果你不熟 GNN，這節可以快速建立直覺。

### GNN 是什麼

傳統神經網路吃 vector / image / sequence。GNN 吃**圖**：點和邊。
最常見的玩法是「**鄰居聚合**」（neighbor aggregation）：每個節點看自己的鄰居，
把鄰居的特徵收集起來、做一次線性轉換，更新自己的 embedding。重複 K 次（叫 K 層），
每個節點就能「聞到」距離自己 K 步以內所有鄰居的味道。

不同 backbone 差在「怎麼聚合」：

- **GCN**：對所有鄰居一視同仁，用度數做 normalization。
- **GAT**：用 attention 算每個鄰居的權重，重要鄰居拿到更多注意力。
- **GraphSAGE**：取樣固定數量的鄰居（不要全部），對大圖比較友善。
- **GIN**：用 sum + MLP，理論上區分能力最強（Weisfeiler-Lehman 等價）。
- **MLP**：完全不看邊，只看自己的特徵。當作「沒用 GNN」的 baseline。

### 為什麼異質圖只能用 GAT / SAGE / MLP

PyG 提供 `to_hetero()` 把單一 backbone 自動轉成多關係版本。
但 GCN 假設「只有一種邊」，GIN 內部有個 inner MLP 在 bipartite 圖會崩。
所以前端的模型選單、後端的 HPO 都會在 hetero 情境下排除這兩個。

### Optuna AutoML 在做什麼

把「選哪個模型 + 用哪組超參數」變成一個搜尋問題。
每跑一個 trial 就：

1. Optuna 抽樣：backbone, hidden_dim (16/32/64/128), num_layers (1~3), dropout (0~0.5), lr (1e-4 ~ 1e-2)
2. 用這組超參數實際訓練（受 epoch 上限與 early-stop 限制）
3. 把 val_loss 回報給 Optuna
4. Optuna 用 TPE / median pruner 等策略決定下一組怎麼抽

跑完 N 個 trial（預設 150）之後，挑 val_loss 最低的那組當 best_config，
再重新跑一次完整訓練、存模型、算指標。

### Train / Val / Test 三組 metrics 為什麼要分開

- **Train**：模型有看過的資料，metrics 越好越正常。
- **Val**：HPO / early-stopping 用來「偷看」泛化能力的資料。
- **Test**：完全沒用過的資料，唯一誠實的泛化指標。

> 過去有個 bug 是 backend 把 val_metrics 寫成 test 的拷貝，2026-04-27 已修。
> 細節：[`docs/changelog/2026-04-27-autopilot-cleanup.md`](../changelog/2026-04-27-autopilot-cleanup.md)。

### Regression 為什麼沒有 confidence

confidence = 「模型對這個分類結果的把握」，是 softmax 機率推導出來的概念。
Regression 輸出的是連續值，本來就沒有 softmax，因此 Predict 頁面在
graph_regression / node_regression 任務下會把 confidence 相關 UI 都隱藏。

---

## 6. 開發環境與常用指令

### 第一次設定

```bash
git clone <repo> && cd LayoutXpert

# Backend
cd backend
uv sync                    # 第一次比較久，會裝 PyTorch + PyG
.venv/Scripts/python -m pytest -q    # 確認測試全綠（50 個）

# Frontend
cd ../frontend
npm install                # 約 1 分鐘
npm test                   # 確認 jest 全綠
```

### 日常開發

```bash
# 兩個 terminal 分別跑
cd backend && .venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev
```

或用 Claude Code 的 preview tool：`launch.json` 已經設好 `backend` 與 `frontend` 兩個 server，
直接呼叫 `preview_start` 即可。

### 改了 API schema 之後

1. 修 `backend/app/schemas/api_models.py`
2. 修 `frontend/lib/api.ts`（**手動**對齊型別 — 沒有自動 codegen）
3. `npx tsc --noEmit` 確認 TS 沒錯
4. 跑 `pytest` + `jest` 雙測

### 加新模型

1. 在 `backend/app/models/<name>.py` 實作 `nn.Module`
2. 在 `backend/app/models/__init__.py` 的 registry 加進去
3. 如果 hetero 友善 → 加進 `ALL_MODELS_HETERO`（前端 `train/page.tsx`）
4. 加 unit test

### 改 Excel schema

最容易踩坑的地方。動之前先讀：

- `backend/app/data/excel_spec.py`（schema 定義）
- `backend/tests/test_excel_ingestion.py`（測試覆蓋很完整）
- `docs/technical/hetero-unified-sheet-and-hpo.md`（最新一次 schema 重構的設計理由）

---

## 7. 接下來可能要處理的方向

按優先序：

1. **Predict 頁面 stand-in 換成真正的 inference endpoint**：目前 Predict 是把上一次 training 的 report 當預覽，沒有真的 forward。需要新增 `POST /projects/{id}/predict` 接受新檔案。
2. **`val_metrics` 在單圖 node-task 路徑仍然 == `test_metrics`**：因為單圖場景沒有獨立 val 切分；要嘛改成有 val split，要嘛在前端對這種 case 隱藏 VALIDATION 欄位。
3. **In-memory store → 持久化**：目前重啟就清空。可以接 SQLite / Redis 做 minimal persistence。
4. **antd v6 升級準備**：console 已經出現 `strokeWidth` / `destroyOnHidden` deprecation warning，未來需要批次替換。
5. **任務排隊 / 多人並行訓練**：目前訓練是同步觸發背景 thread；多使用者場景下要排隊。
6. **真實 IC 資料對接**：demo 資料是合成的；真實 LEF/DEF / GDS 還沒有 importer。

---

## 8. 我還是不懂某某地方？

- 想看歷次大改：翻 `docs/changelog/`，每個檔案有「改了什麼 / 為什麼 / 怎麼用」三段。
- 想看 API 行為：`http://localhost:8000/docs`（FastAPI 自動產生）。
- 想看前端 component 範例：跑 dev server 之後在 `http://localhost:3000` 點點看，所有 page 都是 self-contained。
- 想看訓練 log：`backend/storage/checkpoints/<task_id>/` 下會有 Lightning 的 ckpt 與 events。
- 卡關了：先 `git log --since="3 weeks ago" --oneline` 看最近改了什麼，常常你的問題剛被別人遇過。

歡迎開 issue / 直接聯絡 @KamiMaki。
