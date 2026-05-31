# 2026-04-25 — Five Focused Adjustments

Five scoped refinements to the training / analysis / ingestion pipeline, delivered as six atomic commits on `feat/graphx-frontend-v2`.

## Summary

| # | Area | Change |
|---|------|--------|
| T1 | Evaluate page | Residual plot now shows **error vs predicted** with a red dashed `y = 0` reference line, symmetric Y-domain |
| T2 | Regression metrics | **MAPE** added to backend + frontend; `N/A` fallback when `y_true` contains zero |
| T3 | Training | `ReduceLROnPlateau` → **`ExponentialLR(gamma = 0.95)`** across all 6 model families; shared helper in `backend/app/models/_lr.py` |
| T4 | Explore page | Task Type / Label Column `Select` components removed; values now derived from backend-detected project metadata |
| T5 | Excel schema | **Homogeneous only**: one `Node` / `Edge` / `Graph` sheet per file, no `Type` column in data sheets, no per-type sheet-suffix |

## Commits

1. `05a65d4 feat(backend): add MAPE to regression metrics and error field to residual_data`
2. `f403444 refactor(backend): ExponentialLR replaces ReduceLROnPlateau across models`
3. `4df435c refactor(backend): simplify Excel schema to one sheet per level, no Type column`
4. `baad93d refactor(frontend): explore page derives task_type and label from project API`
5. `c01bc9c feat(frontend): display MAPE and render residual error plot with y=0 reference`
6. `docs: 2026-04-25 changelog + usage + technical for 5 adjustments` (this PR)

## Breaking Changes

**Excel schema (T5)** — legacy workbooks that contain either
- a `Type` column inside `Node` / `Edge` / `Graph` sheets, or
- per-type sheets like `Node_cell`, `Node_pin`, `Edge_pin2net`

will now raise a `ValueError`. Users must regenerate their Excel using the simplified v2 template. A helper is available at `backend/scripts/generate_excel_demos.py`, or download the template via `GET /api/v1/projects/sample-excel`.

If the Parameter sheet declares more than one `Type` value for the same `Level`, ingestion raises:

```
Heterogeneous graphs are no longer supported (2026-04-25).
Parameter sheet declares multiple types for Level=Node: ['cell', 'pin', 'net'].
Use a single Type value per level.
```

## Verification

- Backend: `.venv/Scripts/python.exe -m pytest tests/ -v` → **38 passed**.
- Frontend: `npm test` → **10 suites, 42 tests passed**.
- Manual browser E2E checklist (see `docs/usage/graphx-frontend-v2.md` section *"2026-04-25 verification"*): confirm explore page has no selectors, evaluate page shows MAPE + error plot, training logs show `ExponentialLR` per-epoch decay.

## Migration Notes

- `hetero_wrapper.py` model is still present but no longer exercised by ingestion. Flagged for future cleanup.
- Demo workbook `backend/demo_data/demo_multigraph_hetero.v2.xlsx` is obsolete and no longer parsed by any test. Retained on disk for reference; safe to delete.
- `DEFAULT_LR_GAMMA = 0.95` is exposed at `backend/app/models/_lr.py` for future tuning.

## Author / Context

- Author: penguin (with Claude Opus 4.6 assist)
- Plan: [.omc/plans/ralplan-5-adjustments.md](../../.omc/plans/ralplan-5-adjustments.md)
- Review loop: ralplan (Planner → Architect → Critic) with v2 → v3 → v4 iterations
