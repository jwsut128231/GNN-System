# Optional Type column + Y weight default — Technical Reference

## What Changed

### Modified
- `backend/app/data/excel_spec.py` — `ParameterEntry.weight` defaults to `1.0` (float) for Y rows when the Weight cell is blank or the Parameter sheet has no Weight column. X rows still default to `None` (weight is meaningless for features).
- `backend/app/data/excel_ingestion.py` — `_split_unified_by_type` accepts data sheets without a `Type` column when the Parameter sheet declares only one Type for that Level. All rows are auto-assigned to that single declared type. When ≥2 Types are declared, a missing Type column still raises a clear `ValueError` (with a hint pointing to single-Type sheets).
- `backend/app/routers/projects.py` — `DEMO_EXCELS` list extended with the two new demo variants (`multigraph_homo_no_type`, `multigraph_multi_y`) so the UI can surface them.
- `backend/scripts/generate_excel_demos.py` — adds `make_homo_no_type` (strips Type columns from the existing homo demo) and `make_multi_y_no_type` (30-graph multi-Y regression with no Type columns).
- `frontend/components/GraphPreview.tsx` — fix pre-existing ESLint error (`react-hooks/exhaustive-deps` + React Compiler memoization preservation) by adding `typeColors` to the `graphData` useMemo dependency array. Behaviour unchanged (`typeColors` is already used inside the memo).

### Added
- `backend/tests/test_demo_training_smoke.py` — 4 parametrised end-to-end tests, one per demo workbook (homo, homo-no-type, multi-Y-no-type, hetero). Each test parses → prepares → fits 5 epochs → predicts → asserts shapes & finite metrics.
- `backend/demo_data/demo_multigraph_homo_no_type.xlsx` — 30-graph homo demo with no Type columns.
- `backend/demo_data/demo_multigraph_multi_y.xlsx` — 30-graph multi-Y demo (target_delay weight=2.0, target_power_mw weight blank → 1.0) with no Type columns.

### Backward compatibility
- Single-Y workbooks unchanged in behaviour.
- Multi-Type sheets still require Type columns (only single-Type sheets get the new shortcut).
- Existing Y rows that had non-default Weight values still produce the same weight in `ParameterEntry`.
- `to_payload()` and `_dataset_to_summary` now emit Y entries with `weight=1.0` instead of `None` for blank cells — clients that previously coerced `None → 1.0` continue to work; clients that distinguished `None` from `1.0` will see the explicit `1.0`. None of the in-tree tests or callers depended on that distinction (verified via grep).

## Why

Two pain points pre-2026-05-13:

1. **Homogeneous-graph users had to write a Type column with every row reading "default".** This is pure boilerplate when there's only one declared Type — the column carries no information. Removing the requirement makes the simplest case the easiest to author.

2. **Blank Y weight cells could be confusing.** `label_weight` was stored as `None` internally and coerced to `1.0` at the last moment. That meant `ParameterEntry.weight` had a tristate (None / 1.0 / explicit number) instead of binary (default / explicit). Treating `None` and `1.0` identically inside the entry simplifies downstream code and serialisation.

## How It Works

### `_split_unified_by_type` — single-Type auto-detection

```python
type_col = next(
    (c for c in unified.columns if str(c).strip().lower() == "type"),
    None,
)
if type_col is None:
    if len(declared_types) == 1:
        # Synthesise a Type column with the single declared type.
        unified = unified.copy()
        unified["Type"] = declared_types[0]
        type_col = "Type"
    else:
        raise ValueError("Sheet ... missing a 'Type' column ...")
```

The rest of the function operates on the synthesised Type column, so the per-type split + feature-column dropping logic stays unchanged.

### `ParameterEntry` weight defaulting

```python
weight: Optional[float] = None
if xy == "Y":
    weight = 1.0   # default for Y rows
    if has_weight:
        w_raw = _get(row, "Weight")
        if not pd.isna(w_raw) and str(w_raw).strip() != "":
            weight = float(w_raw)
```

Two paths:
- No Weight column → `weight = 1.0` (early return without checking the cell).
- Weight column present + cell blank → `weight = 1.0` (the `if not pd.isna` branch doesn't fire).
- Weight column present + cell numeric → `weight = float(w_raw)`.

## Usage

### Minimal homo workbook (no Type columns anywhere)

```
Parameter sheet:
  XY | Level | Type    | Parameter      | Weight
  X  | Node  | default | delay_ps       |
  X  | Edge  | default | wire_length_um |
  Y  | Graph | default | target         |   ← blank → 1.0

Node sheet (NO Type column):
  Graph_ID | Node | delay_ps
  1        | 0    | 12
  ...

Edge sheet (NO Type column):
  Graph_ID | Source_Node_ID | Target_Node_ID | wire_length_um
  ...

Graph sheet (NO Type column):
  Graph_ID | target
  1        | 15.7
  ...
```

This parses to:
- `task_type = "graph_regression"`
- `is_heterogeneous = False`
- `label_columns = ["target"]`
- `label_weights = [1.0]`

### Multi-Type Node sheet still requires a Type column

```
Parameter sheet:
  XY | Level | Type | Parameter
  X  | Node  | cell | cell_area
  X  | Node  | pin  | pin_cap

Node sheet:
  Graph_ID | Node | Type  | cell_area | pin_cap    ← Type column REQUIRED
  1        | 0    | cell  | 1.2       |
  1        | 1    | pin   |           | 0.4
```

If the Type column is missing in this case, `parse_excel_file` raises with the message: `"Sheet 'Node' is missing a 'Type' column. The Parameter sheet declares multiple Types (['cell', 'pin']); a Type column is required to split rows. (For a single-Type / homogeneous sheet you may omit the Type column.)"`

## Caveats

- **Heterogeneity is still declared in the Parameter sheet**, not inferred from the data sheet's Type column presence. The Type column drives row-splitting, not classification of the graph as homo/hetero.
- **`to_payload()` now emits `weight=1.0` for Y rows that previously emitted `weight=None`.** Frontend code that relies on `weight == null` to mean "default" needs to switch to `weight === 1.0` (or treat both as default — both produce the same training result).
- **The demo workbooks are regenerated each time `scripts/generate_excel_demos.py` runs.** They may show as modified in `git status` after a regenerate even when the script logic is unchanged (due to embedded timestamps in `.xlsx`).
