# 2026-05-13 — Optional Type column + Y weight default + mock data refresh

## What

- **Data sheets (Node / Edge / Graph) no longer require a `Type` column** when the Parameter sheet declares only a single Type for that Level. Users with homogeneous graphs can skip the Type column entirely.
- **Heterogeneity is still driven by the Parameter sheet** (number of declared Types per Level), but the data-sheet layout is now more forgiving — auto-assign all rows to the single declared Type when no Type column is present.
- **Y weight defaults to `1.0`** explicitly in the parsed `ParameterEntry`. Blank Weight cells and Parameter sheets without a Weight column now both produce `weight = 1.0` instead of `None`.
- **Two new demo workbooks** ship in `backend/demo_data/`:
  - `demo_multigraph_homo_no_type.xlsx` — 30-graph homogeneous regression with no Type columns.
  - `demo_multigraph_multi_y.xlsx` — 30-graph multi-Y regression (target_delay weight=2.0, target_power_mw weight blank → 1.0) with no Type columns.
- The UI `/demo-excels` endpoint now lists all four variants.

## Scope

- ✅ Single-Type Node / Edge / Graph sheets may omit the Type column entirely.
- ✅ Multi-Type sheets still require the Type column to disambiguate (clear error otherwise).
- ✅ Y weight defaults to `1.0` in every code path (single-Y `label_weight`, multi-Y `label_weights`).
- ✅ Smoke test exercises full training pipeline on every demo workbook.
- ❌ Existing workbooks with a `Type` column unchanged in behaviour.

## Files touched

Backend:
- `app/data/excel_spec.py` — Y rows default to `weight=1.0` (instead of `None`); X rows still `None`.
- `app/data/excel_ingestion.py` — `_split_unified_by_type` accepts missing Type column when only one Type is declared.
- `app/routers/projects.py` — DEMO_EXCELS list extended with the two new variants.
- `scripts/generate_excel_demos.py` — adds `make_homo_no_type` and `make_multi_y_no_type` builders.
- `tests/test_excel_ingestion.py` — 8 new tests covering the new behaviours.
- `tests/test_demo_training_smoke.py` — new file, 4 parametrised E2E tests over the demo workbooks.

Demo data (regenerated):
- `demo_data/demo_multigraph_homo.v2.xlsx`
- `demo_data/demo_multigraph_hetero.v2.xlsx`
- `demo_data/demo_multigraph_homo_no_type.xlsx` *(new)*
- `demo_data/demo_multigraph_multi_y.xlsx` *(new)*

## Verification

- `78/78 backend pytest pass` (74 regression + 4 new smoke tests).
- Frontend `tsc --noEmit` + ESLint clean (unchanged surface).
