/**
 * Tests for the v2-pipeline progress helpers. Covers the legacy→v2 mapping
 * (including the Predict unlock on `status === 'completed'`) and label invariants.
 */

import { V2_STEP_LABELS, V2_STEP_COUNT, reachedFromLegacy } from '../progress';

describe('progress · v2 step labels', () => {
    it('has 6 labels in Create→Upload→Analyze→Train→Eval→Predict order', () => {
        expect(V2_STEP_COUNT).toBe(6);
        expect(Array.from(V2_STEP_LABELS)).toEqual(['Create', 'Upload', 'Analyze', 'Train', 'Eval', 'Predict']);
    });
});

describe('progress · reachedFromLegacy', () => {
    it.each([
        [1, 'data_uploaded',   2],  // legacy upload           → v2 Upload done
        [2, 'data_confirmed',  3],  // legacy analysis         → v2 Analyze done
        [3, 'training',        4],  // legacy training         → v2 Train current
        [4, 'failed',          5],  // legacy evaluation       → v2 Eval done
        [5, 'training',        5],  // legacy models (capped)  → v2 Eval
    ])('maps legacy %i / %s → reached %i', (legacyStep, status, expected) => {
        expect(reachedFromLegacy(legacyStep, status)).toBe(expected);
    });

    it('unlocks Predict (reached = 6) only when status === completed AND legacy ≥ 4', () => {
        expect(reachedFromLegacy(4, 'completed')).toBe(6);
        expect(reachedFromLegacy(5, 'completed')).toBe(6);
        expect(reachedFromLegacy(3, 'completed')).toBe(4); // not yet at Evaluate
    });

    it('clamps out-of-range legacy values', () => {
        expect(reachedFromLegacy(0, 'created')).toBe(2); // min 1 → reached 2
        expect(reachedFromLegacy(99, 'created')).toBe(5); // cap at Evaluate
    });
});
