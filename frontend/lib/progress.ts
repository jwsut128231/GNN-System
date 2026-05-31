/**
 * Shared pipeline-progress helpers for the v2 6-step workflow
 * (Create → Upload → Analyze → Train → Evaluate → Predict).
 *
 * Backend ProjectSummary still uses a 1..5 `current_step` from the legacy 5-step pipeline.
 * These helpers translate legacy step numbers to the v2 "reached" index (0..6) used by
 * the dashboard mini-rail and anywhere else the UI wants to reason about completion.
 */

export const V2_STEP_LABELS = ['Create', 'Upload', 'Analyze', 'Train', 'Eval', 'Predict'] as const;
export const V2_STEP_COUNT = V2_STEP_LABELS.length;

/**
 * Translate a legacy `current_step` (backend 1..5) plus project `status` into the
 * v2 reached count (1..6). A project record always implies Create is done (reached ≥ 1).
 * Predict (reached = 6) unlocks only when `status === 'completed'`.
 */
export function reachedFromLegacy(currentStep: number, status: string): number {
    // legacy 1 (upload)     → reached 2  (create + upload done)
    // legacy 2 (analysis)   → reached 3
    // legacy 3 (training)   → reached 4
    // legacy 4 (evaluation) → reached 5
    // legacy 5 (models)     → reached 5 (Models is out of the v2 pipeline — cap here)
    const base = Math.min(Math.max(currentStep, 1), 4) + 1; // 2..5
    if (status === 'completed' && base >= 5) return 6;
    return base;
}
