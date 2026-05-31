# Multi-Y Training — Technical Reference

## What Changed

### Added
- `backend/app/models/loss.py` — `weighted_regression_loss(out, y, loss_weights, num_targets)` helper used by every regression model.
- `backend/tests/test_pyg_converter_multi_y.py` — converter shape tests.
- `backend/tests/test_multi_y_models.py` — model output + weighted loss tests.
- `backend/tests/test_pipeline_multi_y.py` — pipeline prepare-stage tests.
- `docs/changelog/2026-05-12-multi-y-training.md`
- `docs/usage/multi-y-training.md`
- `docs/technical/multi-y-training.md` (this file)

### Modified
- `backend/app/data/excel_ingestion.py` — Multi-Y on a single Level now allowed. Multi-Y classification and mixed kinds rejected with clear messages.
- `backend/app/data/pyg_converter.py` — Both converters accept `label_column: str | Sequence[str]`. Emit `y` shape `(N, T)` (node-level) or `(1, T)` (graph-level) when `T > 1`.
- `backend/app/data/pyg_converter_hetero.py` — `_build_single_hetero` accepts `label_columns: list[str]` and emits `(1, T)` `data.y` for multi-Y.
- `backend/app/training/target_scaler.py` — `mean` / `std` are now `float | np.ndarray`. `fit` accepts 1-D (scalar mean/std) or 2-D (per-target arrays). Transform / inverse broadcast over the last dim.
- `backend/app/models/gcn.py|gat.py|sage.py|gin.py|mlp.py` — Each `__init__` now accepts `num_targets` and `loss_weights`. Final classifier output dim becomes `num_classes * num_targets`. Regression squeeze keeps `[N]` shape only when `num_targets == 1`.
- `backend/app/models/hetero_wrapper.py` — Same kwargs; head emits `num_classes * num_targets`.
- `backend/app/models/factory.py` — Forwards `num_targets` and `loss_weights` to `HeteroGraphRegressor`.
- `backend/app/training/optuna_search.py` — Adds `num_targets` and `loss_weights` parameters that are passed into every trial's model construction.
- `backend/app/training/pipeline.py` — Reads `label_columns` / `label_weights` from the dataset, builds the loss-weights tensor, fits a vector scaler on stacked Y, computes per-target metrics + residuals, populates the new report fields. Model checkpoint payload extended.
- `backend/app/schemas/api_models.py` — `Report` gains `label_columns`, `per_target_metrics`, `per_target_residuals`. `DatasetSummary` gains `label_columns` and `label_weights`.
- `backend/app/routers/projects.py` — `_store_excel_dataset` records the new lists. `_dataset_to_summary` exposes them in the API.
- `backend/tests/test_excel_ingestion.py` — 4 new multi-Y tests; the multi-level-Y deferred test now matches the new error message.
- `backend/tests/test_target_scaler.py` — 6 new vector-target tests.
- `frontend/lib/api.ts` — `Report` interface extended.
- `frontend/app/projects/[id]/evaluate/page.tsx` — Renders Per-Target Test Metrics section + per-target residual plots.

## Why

Designs in EDA workflows often need to predict **multiple correlated targets** at once (delay + area + power; setup slack + hold slack; etc.). Training one model per target is expensive (3× HPO + 3× checkpoints) and discards the inductive bias that comes from sharing a representation. The user request was: "我的 Excel 可能會包含多個 Y，要能一次 predict，weight 欄位表示不同 Y 的權重，evaluation 也要能分別呈現多個 Y 的結果."

We deliberately **stopped short of multi-Y classification** for v1. Cross-entropy per Y with potentially different `num_classes` per Y means either a list of heads or a padded "multi-classifier" — both are more code surface than the regression case warrants. We rejected it at the Excel ingestion layer with a clear error so users get fast feedback rather than a silent miscompile.

## How It Works

### Data flow

```
Excel workbook (Parameter sheet with N Y rows, all same Level)
  │
  ▼  parse_excel_file()
{ "label_columns": ["y1", "y2", ...],         ← new
  "label_weights": [w1, w2, ...],             ← new
  "label_column":  "y1",   "label_weight": w1 ← legacy single-Y aliases
  "task_type":     "graph_regression",
  ... node_dfs / edge_dfs / graph_df ...
}
  │
  ▼  _store_excel_dataset()      (routers/projects.py)
Dataset record (in `store`) carries label_columns + label_weights.
  │
  ▼  run_training_task()         (training/pipeline.py)
  │
  ├─ _prepare_{node,graph_homo,hetero}(dataset, gen)
  │     dataframes_to_*(label_columns=...)
  │     ► Per-graph data.y shape: (1, T) for graph-level, (N, T) for node-level
  │     ► TargetScaler.fit(stacked_train_y)  — vector mean/std when T>1
  │     ► transform_tensor applied to train + val items
  │
  ├─ run_hpo(num_targets=T, loss_weights=..., ...)
  │     Each Optuna trial builds a model with num_targets/loss_weights
  │     via get_model(..., num_targets=T, loss_weights=w).
  │
  ├─ get_model(best, num_targets=T, loss_weights=w)
  │     classifier = Linear(hidden, num_classes * T)
  │     Regression squeeze only when T==1.
  │
  ├─ Trainer.fit(...)
  │     Per-step loss = weighted_regression_loss(out, batch.y, lw, T)
  │
  └─ Evaluation
        _predict_list / _predict_single → predictions/y of shape (N_total, T)
        scaler.inverse_np broadcasts over T
        For each Y column → _regression_metrics(y_true[:, i], y_pred[:, i])
        train/test_metrics ← mean of per-target metrics (legacy fields)
        per_target_metrics, per_target_residuals ← keyed by column name
```

### Tensor shapes — the contract

| Path | T=1 (legacy) | T>1 |
| --- | --- | --- |
| `dataframes_to_pyg_dynamic` → `data.y` | `(N,)` | `(N, T)` |
| `dataframes_to_graph_list` → per-graph `data.y` | `(1,)` | `(1, T)` |
| `parsed_excel_to_hetero_list` → per-graph `data.y` | `(1,)` | `(1, T)` |
| PyG batching collates B graphs of `(1, T)` into | `(B,)` | `(B, T)` |
| Model forward (regression) | `(B,)` after `squeeze(-1)` | `(B, T)` |
| `TargetScaler.mean` / `.std` | `float` | `np.ndarray[T]` |
| `loss_weights` model buffer | `None` (single-Y) | `torch.Tensor[T]` |

When `T == 1` everything keeps its historical shape so all single-Y tests and existing demos continue to work bit-for-bit.

### Weighted regression loss

Defined in `backend/app/models/loss.py`:

```python
def weighted_regression_loss(out, y, loss_weights, num_targets):
    if num_targets <= 1 or out.dim() == 1 or out.shape[-1] == 1:
        return F.mse_loss(out.reshape_as(y), y)
    se = (out - y) ** 2          # shape (B, T)
    if loss_weights is not None:
        se = se * loss_weights   # broadcasts over batch dim
    return se.sum(dim=-1).mean() # per-sample sum across targets, mean across batch
```

We deliberately use `sum across targets → mean across batch` rather than `mean over (B*T)`. The latter would divide by `T` and shrink the loss signal as you add Y columns; the former keeps each sample's loss interpretable as "total weighted error for this graph/node".

### TargetScaler — scalar vs vector

`TargetScaler.fit(values)`:

- 1-D input → `mean: float`, `std: float`. Behaviour unchanged.
- 2-D input `(N, T)` → `mean: ndarray[T]`, `std: ndarray[T]`. Per-target standardisation.

`transform_tensor(t)` / `inverse_tensor(t)`:

- If `mean` is `np.ndarray`, broadcast over the last dim of `t`.
- If `mean` is `float`, behave as before.

`to_dict()` / `from_dict()` round-trip lists when vector, floats when scalar. The model checkpoint stores either form transparently.

## Usage

### Constructing a multi-Y model in code

```python
from app.models.factory import get_model
import torch

model = get_model(
    "sage",
    num_features=8, num_classes=1, task_type="graph_regression",
    hidden_dim=64, num_layers=3, dropout=0.3, lr=1e-3,
    num_targets=2,                                  # <-- T
    loss_weights=torch.tensor([2.0, 0.5]),          # <-- per-target weights
)
```

For hetero graphs, pass `metadata=(node_types, canonical_edges)` and the factory will build a `HeteroGraphRegressor` with the same `num_targets` / `loss_weights` semantics.

### Reading per-target metrics from the report

```python
report = task["report"]              # dict, also returned via the /report endpoint
report["label_columns"]              # → ["target_delay", "target_power_mw"]
report["per_target_metrics"]         # → {"target_delay": {"mse": ..., "mae": ..., "r2_score": ...}, ...}
report["per_target_residuals"]       # → {"target_delay": [{"actual": ..., "predicted": ...}, ...], ...}
report["test_metrics"]               # → mean of per-target metrics (backwards-compatible single-Y semantics)
```

## Caveats

- **Multi-Y classification is not supported.** Detected at parse time; you'll see "Multi-Y classification is not yet supported; only multi-Y regression is supported in this release." If you really need this, train one classifier per Y for now.
- **All Y columns must be on the same Level.** Mixing Node-Y with Graph-Y is rejected at parse time.
- **All Y columns must be the same task kind.** Mixing regression and classification Ys is rejected.
- **The scaler operates per-target.** This is usually the right call (each Y normalised to ~N(0,1)), but it means the absolute scale of `loss_weights` matters less than their **relative** scale. Doubling all weights uniformly does not change the optimal solution; doubling one of them does.
- **Aggregate `test_metrics` is a plain mean across targets.** That's fine as a one-glance summary but it can hide a single bad target. Always look at `per_target_metrics` to spot Y columns that didn't converge.
- **Frontend per-Y residual plots scale with T.** With many Y columns the page gets long; we'd consider a tabbed layout if multi-Y use cases routinely exceed ~4 targets.
- **HPO objective is single-valued (val_loss).** Optuna optimises the aggregate weighted MSE, not per-target. If you want to optimise one Y at the expense of others, set its weight much higher than the rest.
