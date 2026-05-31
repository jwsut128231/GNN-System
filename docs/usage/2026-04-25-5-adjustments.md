# 2026-04-25 Adjustments — Usage Guide

Quick reference for the five refinements landed on 2026-04-25.

## 1. Residual plot (Error vs Predicted)

**Where**: Evaluate page (`/projects/{id}/evaluate`), under "Residual Plot (Error vs Predicted)".

**What changed**: Y-axis now shows the residual error (`actual − predicted`), with a red dashed reference line at `y = 0` and a symmetric Y-domain. A cloud of points centred on the zero line indicates an unbiased model; systematic drift above/below the line flags bias; a funnel shape flags heteroscedasticity.

No user action required — the chart automatically uses the new format for any regression report.

## 2. MAPE metric

**Where**: Evaluate page, regression metrics grid (alongside MSE, MAE, R²).

**What it means**: Mean Absolute Percentage Error, shown as a percentage (e.g. `12.34%`).

**`N/A` case**: MAPE is undefined when the ground truth contains zero. If any `y_true` value is `0`, the backend stores `null` and the UI renders `N/A`. The other metrics (MSE / MAE / R²) are still computed and displayed.

## 3. Learning-rate scheduler — basic exponential decay

Previously the trainer used `ReduceLROnPlateau` (adaptive — shrinks LR only when validation loss stalls). From 2026-04-25 onward the trainer uses `ExponentialLR(gamma = 0.95)` — the LR is multiplied by `0.95` after every epoch.

Effect on loss curves: smooth, monotonic decay — no step-down artefacts from adaptive scheduling.

To tune the decay rate, edit `backend/app/models/_lr.py`:

```python
DEFAULT_LR_GAMMA = 0.95
```

## 4. Explore page — no more Y-label selector

**Where**: Explore page (`/projects/{id}/explore`), Section III "Label & Target Analysis".

**What changed**: The Task Type and Label Column `Select` controls have been removed. The page now reads `declared_task_type` and `declared_label_column` from the project metadata (which the backend populates during Excel ingestion based on the Parameter sheet). The detected label name is shown in the card title:

> **III. Label & Target Analysis — \<label_name\>**

**If detection is wrong**: edit your Parameter sheet (`XY` column — exactly one `Y` row per target level) and re-upload the Excel file. There is no inline override.

## 5. Excel template — one sheet per level, no Type column

**New schema** (homogeneous graphs only):

| Sheet name | Required columns |
|------------|-------------------|
| `Parameter` | `XY`, `Level`, `Type`, `Parameter` (each `Level` may have only one distinct `Type` value) |
| `Node` | `Node` (id), `Graph_ID`, plus X/Y feature columns declared in Parameter |
| `Edge` | `Graph_ID`, `Source_Node_ID`, `Target_Node_ID`, plus declared edge features |
| `Graph` | `Graph_ID`, plus declared graph-level features/labels |

**Removed**:
- `Type` column inside `Node` / `Edge` / `Graph` data sheets.
- Per-type sheet suffixes (`Node_cell`, `Node_pin`, `Edge_pin2net`, ...).
- Unified-split ingestion path (`_split_unified_by_type`).

**If you need heterogeneous support**: not available in this release. File an issue if your workflow needs it back.

**Regenerating the template**:
- Easiest: `GET /api/v1/projects/sample-excel` on a running backend.
- Scripted: `python backend/scripts/generate_excel_demos.py` writes `demo_multigraph_homo.v2.xlsx` into `backend/demo_data/`.

## Manual E2E verification checklist

Run through these after pulling the changes to confirm the UI behaves as expected. Each item is 30–60 seconds.

1. **Explore page, no selectors**
   - Upload a v2-schema Excel file → navigate to `/projects/{id}/explore`.
   - Section III card title should read `III. Label & Target Analysis — <label>`.
   - **Fail if**: you still see `Task Type` / `Label Column` `Select` dropdowns.

2. **Excel ingest — new schema accepted**
   - Upload a workbook with plain `Node` / `Edge` / `Graph` sheets (no `Type` column).
   - Expect HTTP 200 and project advances to step 3 (`data_confirmed`).

3. **Excel ingest — multi-type Parameter rejected**
   - Upload a Parameter sheet with two different `Type` values for `Level=Node`.
   - Expect `422` with message starting `Heterogeneous graphs are no longer supported (2026-04-25)`.

4. **Evaluate page — MAPE + error plot**
   - Train any regression model (e.g. MLP or GCN on a Graph_regression task).
   - Navigate to `/projects/{id}/evaluate`.
   - Expect: MAPE cell present in the metrics grid (or `N/A` if your labels contain zero).
   - Expect: residual chart titled *"Residual Plot (Error vs Predicted)"* with red dashed `y = 0` line, Y-axis symmetric about zero.

5. **Training logs — ExponentialLR decay**
   - During training, `lr` in the training history should decrease by a factor of `0.95` each epoch (e.g. `1e-3 → 9.5e-4 → 9.025e-4 → ...`). No "plateau" or "patience" log lines.

## Troubleshooting

- **"Label column not detected"** — check the Parameter sheet has exactly one `XY=Y` row per target level; re-upload.
- **MAPE shows `N/A`** — your test labels include `0`; MAPE is mathematically undefined; use MAE or RMSE instead.
- **Training loss looks flat** — after ~100 epochs the LR has shrunk to `0.95^100 ≈ 0.006×` the base. If underfitting, raise `DEFAULT_LR_GAMMA` toward `0.98` or `0.99`.
