# 2026-04-25 Five Adjustments — Technical Reference

## What Changed (files)

| # | File | Delta |
|---|------|-------|
| T1 | `backend/app/training/pipeline.py` (370–373) | Residual entries now include `error` field |
| T1 | `frontend/app/projects/[id]/evaluate/page.tsx` (355–376) | Residual chart → error plot with `ReferenceLine y=0` |
| T1 | `frontend/lib/api.ts` (218) | `Report.residual_data` entry adds `error: number` |
| T2 | `backend/app/training/pipeline.py` (52–57) | `_regression_metrics` adds `mape`; returns `None` when `y_true` contains 0 |
| T2 | `backend/app/schemas/api_models.py` (159–162) | `SplitMetrics` adds `mape: Optional[float]` |
| T2 | `frontend/app/projects/[id]/evaluate/page.tsx` (46–59) | Metrics grid gains MAPE `<Statistic>` card |
| T2 | `frontend/lib/api.ts` (55–58) | `SplitMetrics` adds `mape?: number \| null` |
| T2 | `backend/tests/test_pipeline_metrics.py` (new) | Unit tests for MAPE + residual shape |
| T3 | `backend/app/models/_lr.py` (new) | Shared `build_scheduler(opt)` helper + `DEFAULT_LR_GAMMA = 0.95` |
| T3 | `backend/app/models/{mlp,gcn,gat,gin,sage,hetero_wrapper}.py` | Each `configure_optimizers` now calls `build_scheduler`; drops `monitor`/`interval` dict keys |
| T3 | `backend/tests/test_lr_scheduler.py` (new) | Smoke test — `lr = base × gamma^N` within 1e-6 |
| T4 | `frontend/app/projects/[id]/explore/page.tsx` | Removed `TASK_TYPES`, the two `Select` components, `setTaskType`/`setLabelColumn`, and "Please select a task type" alert. Added `getProject()` call; `taskType`/`labelColumn` now derived from `projectMeta.dataset_summary.declared_{task_type,label_column}`. Retitled the Label Analysis card. |
| T5 | `backend/app/data/excel_ingestion.py` | Removed `_split_unified_by_type`; removed per-type sheet-suffix handling; internal `_node_type`/`_edge_type` hard-coded to `"default"` |
| T5 | `backend/app/data/excel_spec.py` | New `validate_single_type_per_level(spec)`; called at the top of `parse_excel_file` |
| T5 | `backend/tests/test_excel_ingestion.py` | Drop Type column from fixtures; new `test_single_sheet_homogeneous`, `test_multi_type_parameter_raises`; removed legacy hetero tests |
| T5 | `backend/tests/test_demo_data.py` | Removed `test_demo_hetero_parses_and_has_three_node_types` |
| T5 | `backend/tests/test_excel_router.py` (44) | Fixture sheet name `Node_default` → `Node` |
| T5 | `backend/scripts/generate_excel_demos.py` | Emits only the homogeneous template; hetero generator removed |

## Why

Each change was a direct user directive on 2026-04-25:

1. **T1** — *"residual plot修正，改成以Y軸為0為主，畫出error，而不是實際vs預測的scatter plot"*. The old actual-vs-predicted scatter is useful for correlation but hides bias; error-vs-predicted makes bias and heteroscedasticity jump out immediately.
2. **T2** — *"metric補上MAPE"*. MAPE is the most interpretable error metric for non-technical stakeholders ("we're off by X%").
3. **T3** — *"lr scheduler先用基本的指數就好，不要cosine或其他的，training loss會看起來很怪"*. Cosine annealing and adaptive plateau schedulers can produce visually surprising loss curves (sudden drops, long flats). Exponential decay is the simplest monotonic schedule that still lets the model fine-tune in late epochs.
4. **T4** — *"data analysis頁面移除label的選擇，現在改成上傳excel時自動讀入Y label，在analysis頁面只需要呈現label的分布就好"*. The backend already auto-detects task type and label column from the Parameter sheet's `XY` column. Exposing both as a user-editable `Select` created a path for silent disagreement with the Excel source of truth.
5. **T5** — *"excel data之後不會有type欄位，統一在mock data 移除並更新 ... Excel異質圖也不需要多個sheet，保留V2的每個level一個sheet就好"*. Heterogeneous graphs were adding significant schema complexity with low usage. Dropping the `Type` column and per-type sheets lets new users build a valid template with only four sheets (Parameter + Node + Edge + Graph) and no metadata to track.

## How It Works

### T1 / T2 — metrics pipeline

```
┌──────────────────────────┐       ┌─────────────────────────┐
│ training/pipeline.py     │       │ schemas/api_models.py   │
│ _regression_metrics()    │  ───▶ │ SplitMetrics            │
│   mse / mae / r2_score   │       │   mape: Optional[float] │
│   + mape (None if y₀)    │       └─────────────────────────┘
│                          │                     │
│ residual list comp       │                     ▼
│   {actual, predicted,    │       ┌─────────────────────────┐
│    error = actual−pred}  │  ───▶ │ Report.residual_data    │
└──────────────────────────┘       │   .error: number        │
                                   └───────────┬─────────────┘
                                               ▼
                        ┌──────────────────────────────────────┐
                        │ evaluate/page.tsx                    │
                        │   <Statistic>MAPE</Statistic> (N/A)  │
                        │   <ScatterChart>                     │
                        │     y=error, ReferenceLine y=0       │
                        │   </ScatterChart>                    │
                        └──────────────────────────────────────┘
```

Frontend never computes residuals locally — it receives `error` ready to render. This keeps the two sides in lockstep: if the backend formula changes, the frontend updates automatically.

### T3 — scheduler flow

```
┌──────────────────────────┐
│ models/_lr.py            │
│   DEFAULT_LR_GAMMA=0.95  │
│   build_scheduler(opt)   │──┐
└──────────────────────────┘  │
                              │  (shared helper)
  ┌───────────────────────────┼────────────────────────────┐
  │                           │                            │
  ▼                           ▼                            ▼
mlp.py                      gcn.py  gat.py  gin.py   hetero_wrapper.py
configure_optimizers()      (same pattern)
  return {                                     (Lightning calls
    "optimizer": opt,                           sched.step() once
    "lr_scheduler": sched,  ←── per-epoch       per epoch automatically
  }
```

No adaptive/monitor-based decay any more — every model trains with `lr_epoch = lr_base × 0.95 ^ epoch`.

### T4 — explore page state flow

```
(on mount)
  getProjectExplore(id)     getProject(id)
        │                       │
        └─────── Promise.all ───┘
                     │
                     ▼
            setExploreData(exploreResult)
            setProjectMeta(projectResult)
                     │
                     ▼  (derived, not useState)
   taskType    = projectMeta?.dataset_summary?.declared_task_type    ?? ''
   labelColumn = projectMeta?.dataset_summary?.declared_label_column ?? ''
                     │
                     ▼
   Same downstream consumers unchanged:
     validateLabel(id, taskType, labelColumn)
     canConfirm = Boolean(labelValidation?.valid) && !confirming
     handleConfirm → confirmData(id, taskType, labelColumn)
     DataQualityCard / deriveQualityChecks (same signatures)
```

Only the input source changed (user click → API value). Every consumer of `taskType` / `labelColumn` works identically.

### T5 — ingestion schema

Before (v2a, multi-layout):

```
Excel/                          Excel/
├─ Parameter                    ├─ Parameter
├─ Node      ← Type column  OR  ├─ Node_cell
├─ Edge      ← Type column      ├─ Node_pin
└─ Graph                        ├─ Node_net
                                ├─ Edge_pin2net
                                └─ ...
```

After (v2 homogeneous-only):

```
Excel/
├─ Parameter          (one Type per Level)
├─ Node               (no Type column inside)
├─ Edge               (no Type column inside)
└─ Graph              (no Type column inside)
```

Ingestion flow:

```
parse_excel_file(bytes)
   │
   ▼  load sheets by exact name ('Parameter', 'Node', 'Edge', 'Graph')
   │  (no suffix search, no unified-split attempt)
   │
   ▼  validate_single_type_per_level(spec)   ──── raises ValueError
   │                                              if any Level declares
   │                                              more than one Type
   ▼
   _node_type = 'default'   (hard-coded — no longer read from column)
   _edge_type = 'default'
   │
   ▼
   PyG homogeneous Data object
```

## Usage

### Reading the new residual error

```python
from app.training.pipeline import _regression_metrics

m = _regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
# {"mse": 0.02, "mae": 0.1333, "r2_score": 0.97, "mape": 0.0444}

m_with_zero = _regression_metrics([0.0, 1.0], [0.1, 0.9])
# {"mse": 0.01, "mae": 0.1, "r2_score": 0.96, "mape": None}
```

### Tuning the LR decay

```python
# backend/app/models/_lr.py
DEFAULT_LR_GAMMA = 0.95   # change here; affects all 6 model families
```

Or override per model by replacing `build_scheduler(opt)` with `torch.optim.lr_scheduler.ExponentialLR(opt, gamma=...)`.

### Generating a v2 template programmatically

```python
import io, pandas as pd

parameter = pd.DataFrame([
    {"XY": "X", "Level": "Node",  "Type": "default", "Parameter": "feat_a"},
    {"XY": "X", "Level": "Edge",  "Type": "default", "Parameter": "weight"},
    {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "score"},
])
nodes = pd.DataFrame({"Graph_ID": [1]*3, "Node": [0, 1, 2], "feat_a": [0.1, 0.2, 0.3]})
edges = pd.DataFrame({"Graph_ID": [1]*2, "Source_Node_ID": [0, 1], "Target_Node_ID": [1, 2], "weight": [0.5, 0.8]})
graph = pd.DataFrame({"Graph_ID": [1], "score": [42.5]})

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    parameter.to_excel(w, sheet_name="Parameter", index=False)
    nodes.to_excel(w,     sheet_name="Node",      index=False)
    edges.to_excel(w,     sheet_name="Edge",      index=False)
    graph.to_excel(w,     sheet_name="Graph",     index=False)
```

## Caveats

1. **MAPE for `y_true = 0`** — MAPE is mathematically undefined when any ground-truth value is zero (`|y − ŷ| / |y|`). Implementation stores `None` (not `inf`, not `NaN`) so the UI can render `N/A` cleanly. Users who care about relative error on zero-heavy targets should use MAE or a weighted metric instead.

2. **`gamma = 0.95` decay math** — after 100 epochs the LR has shrunk to `0.95^100 ≈ 0.00592×` the base. If you train beyond ~50 epochs and see the loss flatten, raise `DEFAULT_LR_GAMMA` toward `0.98` or `0.99`.

3. **`hetero_wrapper.py` is now dead code** — the ingestion pipeline no longer produces heterogeneous `HeteroData` objects after T5. `hetero_wrapper` is preserved but unreachable. Follow-up: delete or repurpose once T5 is bedded in.

4. **Frontend derives from `projectMeta.dataset_summary`** — this path is fragile if the backend response shape changes. If `declared_task_type` / `declared_label_column` move to a different nesting level in `ProjectDetail`, the explore page silently renders empty (no selector to fall back on). Regression test recommended.

5. **Schema hard-break on `Type` column** — no accept-but-ignore fallback was added. Legacy Excel files will fail with a clear error message pointing at the old column/sheet names. The choice prioritises cleanliness over backwards compatibility; users re-upload with the new template.

6. **E2E in this session was not fully automated** — Next.js dev server on Windows + Chinese path did not start within the session timeout. Verification was done via component/unit tests (42 Jest + 38 pytest, all green). Manual E2E checklist is documented in [../usage/2026-04-25-5-adjustments.md](../usage/2026-04-25-5-adjustments.md#manual-e2e-verification-checklist).

## Module Interaction Diagram

```
┌────────────────┐     upload .xlsx     ┌────────────────────────┐
│ Browser (user) │─────────────────────▶│ /upload-excel endpoint │
└────────────────┘                      └────────┬───────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────────┐
                                    │ excel_ingestion            │
                                    │   validate_single_type…    │◀─── T5 new
                                    │   _node_type='default'     │
                                    └────────────┬───────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────────┐
                                    │ Project.dataset_summary    │
                                    │   declared_task_type       │
                                    │   declared_label_column    │
                                    └────────────┬───────────────┘
                                                 │
                                                 ▼ (explore)
                                    ┌────────────────────────────┐
                                    │ explore/page.tsx           │
                                    │   reads dataset_summary    │◀─── T4
                                    │   renders distribution     │
                                    └────────────┬───────────────┘
                                                 │
                                                 ▼ (train)
                                    ┌────────────────────────────┐
                                    │ models/{mlp,gcn,gat,...}   │
                                    │   build_scheduler(opt)     │◀─── T3
                                    └────────────┬───────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────────┐
                                    │ pipeline.py                │
                                    │   _regression_metrics      │◀─── T2 (mape)
                                    │   residual with error      │◀─── T1
                                    └────────────┬───────────────┘
                                                 │
                                                 ▼ (evaluate)
                                    ┌────────────────────────────┐
                                    │ evaluate/page.tsx          │
                                    │   MAPE stat                │◀─── T2
                                    │   ScatterChart error+y=0   │◀─── T1
                                    └────────────────────────────┘
```
