# Heterogeneous Excel: Shared Features & Isolated-Node Fix

**Date:** 2026-04-26

---

## Declaring a Shared Feature Across Multiple Node Types

Sometimes a feature column (e.g. `area`) is meaningful for more than one node type. The correct way to declare this is **two Parameter rows with the same Parameter name but different Type values**, combined with a **single shared column** in the Node data sheet.

### Parameter sheet

| XY | Level | Type | Parameter    |
|----|-------|------|--------------|
| X  | Node  | cell | area         |
| X  | Node  | pin  | area         |
| X  | Node  | cell | cell_only    |
| Y  | Graph | default | score     |

### Node data sheet

| Graph_ID | Node | Type | area | cell_only |
|----------|------|------|------|-----------|
| 1        | 0    | cell | 1.2  | 0.4       |
| 1        | 1    | cell | 0.9  | 0.7       |
| 1        | 2    | pin  | 0.7  | NaN       |
| 1        | 3    | pin  | 0.5  | NaN       |

The parser splits the Node sheet by the `Type` column. Because `area` is a column in the unified sheet, it is **retained in both the `cell` and `pin` per-type DataFrames** after the split — no duplication, no extra work required.

### Type-scoped NaN behaviour

`cell_only` is declared only for `cell` in the Parameter sheet. After the split:

- The `cell` DataFrame contains real values for `cell_only`.
- The `pin` DataFrame retains the column but all values are `NaN` (the original data had no `cell_only` values for pin rows).

Missing-count statistics are always computed **per-type DataFrame**, so `cell_only` will correctly show 0 missing for `cell` rows and will appear as fully missing (or simply be ignored) for `pin` rows, depending on the downstream feature-engineering layer.

### Rule of thumb

> One Parameter row per (Type, feature) combination you want to use. A single column in the data sheet serves all types that declare it.

---

## Mock Data Generator: No More Isolated Nodes

`backend/scripts/generate_excel_demos.py` previously could generate graphs where some nodes had no edges. Both `make_homo` and `make_hetero` now run a **coverage pass** after the initial edge generation:

**Homogeneous (`make_homo`):**
- After random edge generation, any node not present in any edge (as source or target) gets one additional edge connecting it to a randomly chosen different node.

**Heterogeneous (`make_hetero`):**
- After the structured cell_pin / pin_net edge loops, isolated nodes are detected per type:
  - Isolated `cell` → one new `cell_pin` edge to a random pin.
  - Isolated `pin` → one new `pin_net` edge to a random net.
  - Isolated `net` → one new `pin_net` edge from a random pin.

This matches the declared canonical edge triples `(cell, cell_pin, pin)` and `(pin, pin_net, net)`.

### Regression tests

`backend/tests/test_demo_data.py` now asserts that the first graph in both demo files has zero isolated nodes:

```python
def test_demo_homo_no_isolated_nodes_first_graph(): ...
def test_demo_hetero_no_isolated_nodes_first_graph(): ...
```

To regenerate the demo files after any change to the generator:

```bash
cd backend
.venv/Scripts/python.exe scripts/generate_excel_demos.py
```

The hetero demo writes to `demo_multigraph_hetero.v2.xlsx`; if that file is locked (open in Excel), it falls back to `demo_multigraph_hetero.v2.new.xlsx`.
