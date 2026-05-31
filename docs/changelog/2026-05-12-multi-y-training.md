# 2026-05-12 — Multi-Y target training

## What

A single Excel workbook can now declare **multiple Y columns** (on the same Level — Node or Graph), and the platform trains one model that predicts all of them simultaneously.

- Parameter sheet may list 2+ rows with `XY = Y` on the same Level/Type.
- The `Weight` column of each Y row becomes its **per-target loss weight**.
- The evaluation report now shows **per-Y test metrics** (MSE / MAE / R²) and a **residual plot for every Y**, in addition to the existing aggregate metrics.

## Scope

- ✅ Multi-Y **regression** (node-level and graph-level, homogeneous and heterogeneous)
- ❌ Multi-Y **classification** — rejected at ingest time with a clear error. Different num_classes per Y is structurally complex; deferred to a future release.
- ❌ Multi-Y across **different Levels** (e.g. one Y on Node and one Y on Graph) — also rejected; the platform currently expects all Y columns on the same Level.

## Backwards compatibility

Single-Y workbooks behave exactly as before. The new fields (`label_columns`, `label_weights`, `per_target_metrics`, `per_target_residuals`) stay empty or length-1, and the original `test_metrics` / `residual_data` carry the data as they always have.

## Files touched

Backend:
- `backend/app/data/excel_ingestion.py` — `parse_excel_file` returns `label_columns: list[str]` and `label_weights: list[float]`. Multi-Y classification + mixed kinds rejected. Multi-level Y error message clarified.
- `backend/app/data/pyg_converter.py` — converters accept `label_column: str | Sequence[str]` and emit `y` of shape `(N, T)` (node level) or `(1, T)` (graph level when T>1). Backward-compatible 1-D shape for T==1.
- `backend/app/data/pyg_converter_hetero.py` — `_build_single_hetero` builds `HeteroData.y` of shape `(1, T)` when multi-Y.
- `backend/app/training/target_scaler.py` — `TargetScaler.fit` accepts 1-D (legacy) or 2-D inputs. `mean` / `std` become numpy arrays for multi-Y; transform/inverse broadcast over the last dim.
- `backend/app/models/loss.py` — new `weighted_regression_loss` helper.
- `backend/app/models/{gcn,gat,sage,gin,mlp}.py` — accept `num_targets` + `loss_weights`. Final classifier emits `num_classes * num_targets` outputs. Regression squeeze only when T==1.
- `backend/app/models/hetero_wrapper.py` — same num_targets + loss_weights plumbing.
- `backend/app/models/factory.py` — threads new kwargs into model construction.
- `backend/app/training/optuna_search.py` — HPO forwards num_targets + loss_weights to every trial.
- `backend/app/training/pipeline.py` — fits vector scaler on stacked train Y, computes per-target metrics + residuals, populates new report fields. Model checkpoint payload includes `label_columns` and `num_targets`.
- `backend/app/schemas/api_models.py` — `Report` adds `label_columns`, `per_target_metrics`, `per_target_residuals`. `DatasetSummary` adds `label_columns` and `label_weights`.
- `backend/app/routers/projects.py` — `_store_excel_dataset` records `label_columns`/`label_weights`; `_dataset_to_summary` exposes them.

Frontend:
- `frontend/lib/api.ts` — `Report` interface extended with the new multi-Y fields.
- `frontend/app/projects/[id]/evaluate/page.tsx` — renders a Per-Target Test Metrics section and one residual plot per Y when the report carries multi-Y data.

Tests:
- `backend/tests/test_excel_ingestion.py` — 4 new tests (multi-Y graph regression, multi-Y node regression, mixed kinds rejected, multi-Y classification rejected). Existing multi-level Y test updated to match the new error message.
- `backend/tests/test_pyg_converter_multi_y.py` — new file, 8 tests covering converter shapes for T=1 / T>1.
- `backend/tests/test_target_scaler.py` — 6 new tests covering 2-D fit, broadcast transform/inverse, vector dict round-trip, ndim>2 rejection.
- `backend/tests/test_multi_y_models.py` — new file, 7 tests covering weighted_regression_loss and model output shapes.
- `backend/tests/test_pipeline_multi_y.py` — new file, 5 tests covering prepare functions + end-to-end forward pass with multi-Y model.

## Verification

All 66 backend tests pass. Frontend builds with no TypeScript errors.
