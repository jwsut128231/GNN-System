# 使用指南 · GraphX Frontend v2

## 新的 6 步驟流程

舊版是 5 步驟 (Upload → Analysis → Training → Evaluation → Models)。v2 改成 6 步驟:

```
Create → Upload → Analyze → Train → Evaluate → Predict
```

畫面頂部的步驟列,每個圓點代表一個階段:

- ✓ (青色) = 已完成,可以點回去檢視或重做
- 實心高亮 = 你現在所在的頁面
- 🔒 (灰色) = 還沒解鎖;必須先完成前一步才能進去

**Predict 步驟**要等訓練完成 (status = completed) 才會解鎖,因為需要一個註冊好的模型才能做推論。
模型註冊 (Models) 不在主流程步驟內;訓練完成後會自動註冊,到 Evaluate 頁面可以看到模型並跳到 Models 頁管理。

## Dashboard — Grid / List 切換

右上角 toolbar 多了一組 Grid / List 切換鈕。

- **Grid**:維持原本的卡片列表,適合視覺掃視。每張卡片下方的進度條現在對應 6 步驟並顯示 `N/6`。
- **List**:緊湊的表格,一行一個專案,顯示名稱 / 標籤 / 狀態 / 進度 / 更新時間。適合想一次看很多專案時使用。

你的選擇會記在瀏覽器 localStorage 裡,下次開啟會維持上次的模式。

## Explore — 多圖切換與全螢幕檢視

### 多圖切換 (multi-graph dataset only)

如果你的資料集包含多張圖 (例如 demo 裡的 multigraph 範例),Graph Preview 卡片的右上角會出現下拉選單。可以輸入關鍵字搜尋圖名,選擇後整個 Data Quality 和預覽都會跟著切換。

### 全螢幕 Graph Inspector

Graph Preview 卡片右上角有一顆 ⛶ **Fullscreen** 鈕。點下去會打開一個大視窗 (佔畫面 95%),用更大的畫布呈現同一張圖,方便檢視較複雜的拓撲。按 ESC 或點右上角 X 關閉。

### 資料品質檢查 (Data Quality)

新的「Data Quality · Graph-level checks」卡片會針對圖資料做 12 項檢查,每項都以 ok/warn/err 呈現:

| 檢查項目 | 意義 |
|---|---|
| Connected components | 圖的連通分量數;同質圖通常應為 1 |
| Isolated nodes | 完全沒有邊的節點 |
| Self-loops | 自環數量 |
| Duplicate edges | 重複邊 (在樣本中) |
| Graph schema | 同質 / 異質以及類型數量 |
| Feature NaN / missing | 特徵欄位的缺失值總數 |
| Label leakage | 特徵和標籤的相關係數是否逼近 1 (資料洩漏風險) |
| Degree distribution | 節點度數分布 (min / avg / max) |
| Edge attr coverage | 邊屬性填滿比例 |
| Graph count | 單圖 / 多圖 |
| Class balance / target range | 分類任務的類別平衡度,或回歸任務的目標值範圍 |
| Train/val/test split | 訓練時自動切分 |

卡片標題欄的徽章會快速顯示 ok / warn / err 三類檢查各有幾項。

## Train — 精簡的模型選擇

舊版有 Auto / Manual 的切換。現在統一成一個 checkbox 清單:

1. 最上方的 **Select all (AutoML · try all)**:一鍵勾選所有模型家族,等同 AutoML 模式,backend 會自己跑所有組合。
2. 下方各家族單選:GCN / GAT / SAGE / GIN / MLP,旁邊有一句話提示這個家族的特性。
3. 勾選狀態即時反映在 Start Training 按鈕的可用性:全部取消勾選會出現警告且無法開始。

另外新增了 **Objective** 下拉選單,會依 `task_type` 自動顯示合適的指標:

- 分類:val_acc / val_f1 / val_auroc
- 回歸:val_rmse / val_mae / val_r2

> 目前 Objective 僅作為前端偏好保存;backend 尚未支援自訂最佳化目標,實際搜索仍以內建預設為準。

## Predict (新頁面)

訓練完成後,步驟列的最後一步 **Predict** 會解鎖。進入後會看到三欄版面:

- **左欄**:上傳要推論的 Excel/CSV (可選) + 選擇要使用的已註冊模型 + Run Inference 按鈕。
- **中欄**:推論結果摘要 (平均信心度、需複查數量、總預測數) + 圖視覺 (節點數 ≤ 200) + 每節點預測表。
- **右欄**:信心度直方圖 + 低於 0.70 信心度的待複查清單。

### 大圖 fallback

當預測節點數超過 **200** 時,圖視覺會自動收起,改以橘色 Alert 提示「Graph too large」,並只顯示清單。瀏覽器不會因為上萬節點的 force-directed 圖而卡住。需要視覺檢視的話,可以:

- 回到 Analyze 步驟看取樣的圖
- 用 Export 按鈕下載整份預測 (即將提供)

> 目前 Predict 頁面使用最近一次訓練報告作為預覽,實際即時推論 API 待 backend 提供。
