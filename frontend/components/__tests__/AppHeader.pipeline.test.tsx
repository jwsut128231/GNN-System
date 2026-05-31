/**
 * Verifies the v2 pipeline configuration exported from AppHeader.
 * Covers: 6-step labels in order, STEP_PATHS targeting the predict route, and the
 * index-0 (Create) link going back to /dashboard rather than a per-project page.
 */

import { STEPS, STEP_PATHS } from '../AppHeader';

describe('AppHeader · v2 pipeline config', () => {
    it('exposes 6 steps in v2 order: Create → Upload → Analyze → Train → Evaluate → Predict', () => {
        expect(STEPS.map(s => s.label)).toEqual([
            'Create',
            'Upload',
            'Analyze',
            'Train',
            'Evaluate',
            'Predict',
        ]);
    });

    it('maps step index 0 (Create) to the dashboard rather than a project-scoped path', () => {
        const paths = STEP_PATHS('abc');
        expect(paths[0]).toBe('/dashboard');
    });

    it('routes Predict (index 5) to /projects/:id/predict', () => {
        const paths = STEP_PATHS('abc');
        expect(paths[5]).toBe('/projects/abc/predict');
    });

    it('keeps the existing project-scoped paths for upload/explore/train/evaluate', () => {
        const paths = STEP_PATHS('xyz');
        expect(paths.slice(1, 5)).toEqual([
            '/projects/xyz/upload',
            '/projects/xyz/explore',
            '/projects/xyz/train',
            '/projects/xyz/evaluate',
        ]);
    });
});
