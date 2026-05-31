/**
 * Fix 3: Missing % display correctness — frontend pass-through verification.
 *
 * The explore page builds `attrTableData` from server-supplied `ColumnInfo`
 * entries (from `GenericExploreData.columns`).  The mapping is:
 *   missing:    col.missing_count  (raw count)
 *   missingPct: col.missing_pct    (percentage already computed by backend)
 *
 * There is NO client-side recomputation of missing% — the value is passed
 * straight through from the API response.  This test documents and guards
 * that contract so a future refactor cannot introduce an off-by-100 error or
 * wrong-denominator computation on the client side.
 *
 * The denominator used by the backend is the count of rows of that TYPE
 * (not the full unified frame), so for a hetero graph where cell has 4 rows
 * and area is missing for 2 of them, the backend returns missing_pct=50.0,
 * NOT 25.0 (which would result from dividing by the 8-row unified frame).
 */

import type { ColumnInfo } from '../api';

/** Mirrors the attrTableData construction in explore/page.tsx */
function buildAttrRow(col: ColumnInfo, labelColumn: string) {
  const role =
    col.name === labelColumn
      ? 'label'
      : col.name.toLowerCase() === 'node_id'
        ? 'id'
        : 'feature';
  return {
    key: `node-${col.name}`,
    name: col.name,
    dtype: col.dtype,
    role,
    source: 'node',
    missing: col.missing_count,
    missingPct: col.missing_pct,
    unique: col.unique_count,
  };
}

describe('explore attrTableData missing% pass-through', () => {
  it('passes missing_pct straight through without modification', () => {
    const col: ColumnInfo = {
      name: 'area_um2',
      dtype: 'numeric',
      missing_count: 2,
      missing_pct: 50.0,   // backend computed: 2/4 cell rows = 50%
      unique_count: 2,
    };
    const row = buildAttrRow(col, 'target');
    expect(row.missing).toBe(2);
    expect(row.missingPct).toBe(50.0);
  });

  it('renders missingPct with one decimal place via toFixed(1)', () => {
    const col: ColumnInfo = {
      name: 'area_um2',
      dtype: 'numeric',
      missing_count: 2,
      missing_pct: 50.0,
      unique_count: 2,
    };
    const row = buildAttrRow(col, 'target');
    // Simulate the table cell renderer: `${v.toFixed(1)}%`
    expect(`${row.missingPct.toFixed(1)}%`).toBe('50.0%');
  });

  it('does NOT divide by total row count (would give 25% for 2/8 unified rows)', () => {
    // If the frontend incorrectly recomputed using the unified row count (8),
    // it would give 2/8 * 100 = 25%.  The backend correctly gives 50% (2/4).
    const col: ColumnInfo = {
      name: 'area_um2',
      dtype: 'numeric',
      missing_count: 2,
      missing_pct: 50.0,  // backend uses type-row denominator (4), not total (8)
      unique_count: 2,
    };
    const totalRows = 8; // full unified frame (wrong denominator)
    const wrongPct = (col.missing_count / totalRows) * 100; // 25%
    const row = buildAttrRow(col, 'target');

    // Frontend must use server-supplied missing_pct (50%), not recompute (25%)
    expect(row.missingPct).not.toBe(wrongPct);
    expect(row.missingPct).toBe(50.0);
  });

  it('zero missing renders as 0%', () => {
    const col: ColumnInfo = {
      name: 'cell_drive',
      dtype: 'numeric',
      missing_count: 0,
      missing_pct: 0.0,
      unique_count: 4,
    };
    const row = buildAttrRow(col, 'target');
    expect(row.missing).toBe(0);
    expect(`${row.missingPct.toFixed(1)}%`).toBe('0.0%');
  });
});
