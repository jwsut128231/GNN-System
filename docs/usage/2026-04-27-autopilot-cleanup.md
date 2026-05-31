# 使用指南 — 2026-04-27 修正後的 E2E 流程

這份是給「實際操作 GraphX.AI」的使用者看的：跟隨步驟跑一次完整流程，了解這次修正後使用者體驗有什麼差異。

> 想知道為什麼要做這些修正：[changelog](../changelog/2026-04-27-autopilot-cleanup.md)
> 想知道實作細節：[technical](../technical/2026-04-27-autopilot-cleanup.md)

## 前置作業

```bash
cd backend && .venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev
```

打開 `http://localhost:3000`。

## 步驟 1：建立 Project

點「**+ New Project**」，輸入專案名稱（tags 可空）→ **Create**。

> 改變：對話框沒變動，只是清理確認過。

## 步驟 2：上傳資料 / 載入 Demo

### 載入 Demo（推薦）

點任一 demo 卡片上的「**Load**」。
這次走查推薦 `Multi-Graph Heterogeneous`：30 張圖、3 種 node type（cell / pin / net）、共用 feature（`area_um2` 在 cell+pin、`cap_ff` 在 pin+net），會把這次所有修正都演到。

### 自己上傳

點「**Select .xlsx File**」。

> 改變：頁面文字現在說
> > Fill the **Parameter** sheet to declare features and labels, then fill the **Node**, **Edge**, and **Graph** sheets with data. For heterogeneous graphs, add a **Type** column to **Node / Edge** rows to distinguish node and edge types.
>
> 比舊版的 `Node_* / Edge_* / Graph_*`（誤導成有多張 sheet）正確。

## 步驟 3：Explore（資料分析）

頁面會顯示：

- **Dataset Summary**：圖數、平均 node/edge 數、node types、edge types、feature correlation 矩陣
- **Interactive Graph Preview**：Canvas 畫一張 sample graph，可以點 / hover 看 attributes
- **Node Feature Schema** 表格：列出每個 (feature, type) 的 dtype / role / 缺值
- **Edge Feature Schema** 表格：同上但對邊

> 改變：
> - 共用 feature 不再警告 React duplicate key（瀏覽器 console 應該乾淨）
> - 共用 feature 在 schema 表會顯示成 `area_um2 (cell)` / `area_um2 (pin)`，能看出來它分別屬於哪些 type

確認 label 看起來正常後，按右下「**Confirm & Proceed to Training**」。

## 步驟 4：Train（模型訓練）

### Model Families

> 改變：對 hetero 資料集，列表只剩 **GAT / SAGE / MLP** 三個 checkbox（過去會錯誤地顯示 GCN / GIN，使用者勾了之後 backend 會悄悄跳過）。
>
> 對 homo 資料集，列表還是完整 5 個 backbone（GCN / GAT / SAGE / GIN / MLP）。

預設全選（= AutoML 模式，在所有可用 backbone 之間搜尋）。

### Optuna Trials

滑桿從 10 到 300。第一次測試建議拉到最低（10），整個流程約 30 秒就完成。
正式訓練再拉到 100~300。

### Start Training

按「**Start Training**」。下方 Training Log 會即時更新，狀態包含：

- `QUEUED → TRAINING → COMPLETED`
- 進度條 0~100%
- 試驗計數 `(Trial X/Y)`
- 完成後跳到 `Training completed!`

完成後右上角顯示「**N Experiments**」tag。

## 步驟 5：Evaluate（評估報告）

頁面三大區塊：

### Performance Metrics（指標）

橫向三欄：**TRAINING**、**VALIDATION**、**TEST**

> 改變：VALIDATION 不再是 TEST 的拷貝。三個欄位現在會看到不同的數值（VALIDATION 通常落在 train 與 test 中間，反映 early-stopping 用的那段資料的真實表現）。

對 **graph_regression** / **node_regression** 任務：MSE、MAE、R² Score、MAPE
對 **classification** 任務：Accuracy、F1、Precision、Recall + 混淆矩陣

### Residual Plot（殘差圖）

X 軸 = Predicted、Y 軸 = Error（= actual − predicted）、灰色 y=0 線。
理想狀況：點散落在 y=0 兩側、無系統性偏差。如果你看到斜向趨勢，模型有 bias。

### Training History（訓練曲線）

每個 epoch 的 train loss / val loss。
如果 val loss 比 train loss 大很多 → 過擬合。
如果兩條都很高 → 欠擬合 / 資料量不夠 / lr 太大。

### Best Model Configuration + Leaderboard

最佳那組超參數（model / hidden_dim / num_layers / dropout / lr）+ HPO 期間 top-N trial 的對照表。

### Model Registry

點右上的「**Model Registry →**」進去管理已訓練模型。

## 步驟 6：Predict（推論）

點 sidebar 的「**Predict**」進入。

### 對 classification 任務

頁面有：
- 左：檔案 picker、模型選單、Run Inference
- 中上：**Avg confidence** / **Needs review** / **Predictions** 三個 KPI 卡
- 中下：圖視覺化（節點數 ≤ 200 才顯示） + 預測表（Node / True / Predicted / Confidence）
- 右：Confidence histogram（10 bin）+ Needs review queue（confidence < 0.7 的紀錄）

### 對 regression 任務

> 改變：confidence 相關的 UI 全部隱藏（KPI 卡只剩 **Predictions** 一張，Confidence histogram / Needs review queue 整欄消失），表格列名從 **Node** 改為 **Graph**（graph-level 任務）。
>
> 這些 confidence 概念在 regression 上沒有意義（連續值沒有 softmax 機率），舊版顯示是 bug。

操作流程：
1. 在 Model 區選一個模型（preset 為最近訓練的那個）
2. 拖 Excel 進 Drop area，或之後實作的 sample 模式
3. 按 Run Inference（目前是 stand-in，會把現有 report 的 predictions 顯示出來）
4. 按「Export predictions」可以下載 csv（之後實作）

## 常見問題

**Q：為什麼 R² 是負的？**
A：負 R² 代表模型比「永遠預測平均值」還差。常見原因：trial 數太少、特徵與 label 關聯不足、資料量太小。提高 Optuna trials、增加 demo 圖數試試看。

**Q：載入 demo 之後 Explore 頁面說 0 nodes？**
A：可能 backend uvicorn 重啟過、in-memory state 被清空了。重新 Create + Load demo 即可。

**Q：訓練很慢？**
A：CUDA 沒被偵測到。打開 backend log 確認顯示 `GPU available: True (cuda)`；若否，去查 `nvidia-smi` 與 PyTorch CUDA 版本是否對齊。
