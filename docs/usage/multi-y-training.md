# 多 Y 訓練（Multi-Y Training）使用指南

## 是什麼？

如果你的設計參數同時對應多個指標（例如：延遲、面積、功耗），你可以**一次訓練一個模型同時預測它們**，不必為每個指標訓練一個模型。

平台同時支援：

- **節點層多 Y**（Node-level）：每個節點要預測多個數值
- **圖層多 Y**（Graph-level）：每張圖要預測多個數值
- **同質與異質圖**：兩種拓樸都能跑多 Y
- **每個 Y 可獨立加權**：在 Excel 內指定權重，loss 訓練時自動套用

只支援 **regression**（回歸）—— 分類任務目前一次只能有一個 Y。

## 一、Excel Parameter 表怎麼填？

加 Y row 就好，每個 Y 一列。`Weight` 欄寫該 Y 在訓練時的 loss 權重。

### 範例：圖層多 Y 回歸（兩個目標）

```
XY   | Level | Type    | Parameter        | Weight
-----+-------+---------+------------------+-------
X    | Node  | default | delay_ps         |
X    | Node  | default | area_um2         |
X    | Edge  | default | wire_length_um   |
X    | Graph | default | num_cells        |
Y    | Graph | default | target_delay     | 2.0
Y    | Graph | default | target_power_mw  | 0.5
```

**解讀**：
- 兩個 Y 都掛在 `Graph` 層，所以平台會自動推導出 `graph_regression` 任務。
- `target_delay` 比 `target_power_mw` 在 loss 中重要 4 倍（2.0 vs 0.5），因為延遲是這個設計的關鍵指標。

### 範例：節點層多 Y 回歸（兩個目標）

```
XY   | Level | Type    | Parameter | Weight
-----+-------+---------+-----------+-------
X    | Node  | default | feat_1    |
X    | Node  | default | feat_2    |
Y    | Node  | default | delay     | 1.0
Y    | Node  | default | slack     | 3.0
```

對每個節點同時預測 `delay` 與 `slack`。

## 二、Weight 規則

- 寫進 Parameter 表 `Weight` 欄的數值就是該 Y 在 MSE loss 內的權重。
- 留空 → 預設 `1.0`。
- 多 Y 訓練時的 loss 公式：

  ```
  loss = mean_over_samples( sum_t ( weight_t * (pred_t - actual_t)^2 ) )
  ```

  例如權重 `[2.0, 0.5]`，第一個 Y 對 loss 的貢獻是第二個的 4 倍。

## 三、Y 的 regression / classification 自動判斷

平台依每個 Y 欄資料推斷類型：
- 連續值（非整數或唯一值 > 20）→ regression
- 少量整數 → classification

多 Y 的所有目標必須**屬於同一種**：
- 都是 regression → 多 Y 訓練 ✅
- 都是 classification → 暫不支援，會在上傳階段直接擋下 ❌
- 混合（一些 regression、一些 classification）→ 直接擋下 ❌

## 四、訓練流程不變

照舊：上傳 Excel → Explore → Confirm → Train → Evaluate。平台會自動偵測多 Y 並啟用對應的訓練/評估路徑。

## 五、Evaluation 看什麼？

訓練完成後到 Evaluate 頁面：

1. **Performance Metrics — Aggregate**：把所有 Y 的指標平均（MSE / MAE / R²）。給你一個整體印象。
2. **Per-Target Test Metrics**：每個 Y 個別的 MSE / MAE / R²。可以一眼看出哪個 Y 學得好、哪個還差。
3. **Residual Plot — {Y 名稱}**：每個 Y 各一張殘差圖（實際 vs 預測）。用來判斷是否有系統性偏誤。

## 六、典型問題

> Q：我的某一個 Y 數量級很大（例如 1000），另一個很小（例如 0.1），會不會被大的吃掉？

不會。Backend 內部使用 `TargetScaler`，對每個 Y 個別做標準化（mean=0、std=1）之後才丟進 loss。預測時再 inverse transform 回原本的數值範圍。

> Q：可以混合 Node-level 和 Graph-level 的 Y 嗎？

目前不行。所有 Y 必須在同一個 Level。如果真的需要，請拆成兩個專案分別訓練。

> Q：權重要怎麼選？

如果各 Y 數量級差不多，留空（=1.0）就好。如果某個 Y 對你更重要，把它的權重調高；如果某個 Y 雜訊很大不想拖累其他 Y，把權重調低。
