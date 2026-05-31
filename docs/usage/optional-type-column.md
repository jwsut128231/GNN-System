# Type 欄位變成可選 + Y Weight 預設值

## 簡單版

如果你的圖是 **同質的** —— 也就是 Parameter 表裡每個 Level（Node / Edge / Graph）只有一個 Type —— 那麼你的資料表 **不用再寫 Type 欄位**了。系統會自動把所有 row 當成那唯一一個 Type。

如果你的 Y 欄位的 Weight 是空的，系統會自動套 **1.0** 當預設值。

## 一個範例：最精簡的 Excel

**Parameter 表**
```
XY | Level | Type    | Parameter      | Weight
---+-------+---------+----------------+-------
X  | Node  | default | delay_ps       |
X  | Node  | default | area_um2       |
X  | Edge  | default | wire_length_um |
Y  | Graph | default | target_delay   |   ← Weight 空 → 自動 1.0
```

**Node 表**（注意 → 沒有 Type 欄位！）
```
Graph_ID | Node | delay_ps | area_um2
---------+------+----------+----------
1        | 0    | 12       | 0.5
1        | 1    | 18       | 0.7
...
```

**Edge 表**（也沒有 Type 欄位）
```
Graph_ID | Source_Node_ID | Target_Node_ID | wire_length_um
---------+----------------+----------------+----------------
1        | 0              | 1              | 23.5
...
```

**Graph 表**（也沒有 Type 欄位）
```
Graph_ID | target_delay
---------+-------------
1        | 15.7
...
```

上傳這個 workbook → 自動辨識為 `graph_regression` + 同質圖。

## 什麼時候還需要 Type 欄位？

當你的 Parameter 表針對同一個 Level 宣告 **超過一種 Type** 時，資料表就必須要有 Type 欄位，否則系統不知道怎麼分。

例如以下 Parameter 表會強制 Node 表必須有 Type 欄位：
```
XY | Level | Type | Parameter
---+-------+------+----------
X  | Node  | cell | cell_area
X  | Node  | pin  | pin_cap
Y  | Graph | default | score
```
因為這宣告了兩種 Node Type（cell + pin），系統需要 Type 欄位才能判斷哪一 row 屬於哪一種。

## Y Weight 預設行為

- Parameter 表完全沒有 Weight 欄位 → 所有 Y 自動 1.0
- 有 Weight 欄位但 Y row 的 Weight 是空的 → 1.0
- 有 Weight 欄位且 Y row 寫了數字 → 用那個數字

多 Y 的情況下這個機制特別好用：你只需要寫你想加權的 Y，其它留空就好。

## Demo 檔案

- `demo_multigraph_homo.v2.xlsx` —— 同質圖，有 Type 欄位（既有）
- `demo_multigraph_homo_no_type.xlsx` —— 同質圖，**沒有** Type 欄位（新）
- `demo_multigraph_multi_y.xlsx` —— 多 Y 同質圖，沒有 Type 欄位，第二個 Y 的 Weight 留空（新）
- `demo_multigraph_hetero.v2.xlsx` —— 異質圖，需要 Type 欄位（既有）

從 UI 的 Demo Excel 下拉選單都可以一鍵載入。
