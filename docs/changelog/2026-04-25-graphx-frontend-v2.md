# 2026-04-25 — GraphX Frontend v2

Applied the design handoff bundle at `.design-ref/project/GraphX Frontend Improvements v2.html`.
Rebuilt the project workflow around a new 6-step pipeline and added a scaffolded Predict page.

## Pipeline

- Pipeline is now **Create → Upload → Analyze → Train → Evaluate → Predict** (6 steps, was 5).
- The Models/Register step was removed from the stepper. Models is still project-scoped
  (`/projects/[id]/models`) and linkable from Evaluate / project landing.
- Stepper nodes now show:
  - ✓ check for completed steps (clickable to revisit)
  - highlighted index for the current step
  - 🔒 lock icon + tooltip for steps the user hasn't unlocked yet
- Predict (index 5) unlocks only when `status === 'completed'` AND Evaluate (index 4) has been reached.

## Dashboard

- Added a **Grid / List** toggle (persisted to `localStorage['dashboard.view']`).
  - Grid = original card view, now using the 6-step mini-rail.
  - List = AntD Table with Name / Tags / Status / Progress / Updated / Actions.
- Mini-rail shows `reached / 6` and short labels (Create · Upload · Analyze · Train · Eval · Predict).

## Explore

- Graph Preview card now has a **Fullscreen** button that opens a 95vw modal hosting
  the same `GraphPreview` at a larger height.
- Multi-graph switcher (the existing `Select`) became a searchable combobox.
- Added a new **Data Quality · Graph-level checks** card below the preview with 12 graph-specific
  checks (connected components, isolated nodes, self-loops, duplicate edges, schema type, feature
  NaN, label leakage, degree distribution, edge attr coverage, graph count, class balance /
  target range, train/val/test split). Each check reports `ok / warn / err / na` with badge counts
  in the card header.

## Train

- Removed the Auto/Manual Switch (Darren's feedback: IC designers don't want a mode toggle).
- Single checkbox list: top row "Select all (AutoML · try all)" toggles all families; below are
  GCN / GAT / SAGE / GIN / MLP with one-line role hints.
- New **Objective** dropdown, populated from `project.task_type`
  (classification → val_acc/f1/auroc, regression → val_rmse/mae/r²).
- Warning alert surfaces when no family is selected.

## Predict (new)

- New route `/projects/[id]/predict`. Three-column layout:
  - **Left**: file drop zone + registered-model selector + Run Inference
  - **Centre**: summary strip (avg confidence · needs-review count · prediction count)
    + graph view (hidden when `node_count > 200`, per Darren's feedback)
    + per-node predictions table
  - **Right**: confidence histogram + "Needs review" queue (confidence < 0.70)
- Uses the latest training report as a preview stand-in until the backend gains
  `POST /api/v1/projects/{id}/predict`.

## Shared helpers

- `frontend/lib/progress.ts` — exports `V2_STEP_LABELS`, `V2_STEP_COUNT`, `reachedFromLegacy`.
  Used by the dashboard today; any future page that needs to render workflow progress
  should import from here instead of re-deriving the mapping.

## Tests

- `AppHeader.pipeline.test.tsx` — verifies 6-step order and STEP_PATHS shape.
- `lib/__tests__/progress.test.ts` — verifies legacy→v2 step mapping + Predict unlock rule.
- Full suite: 42/42 passing.

## Files touched

```
M  frontend/components/AppHeader.tsx
M  frontend/components/GraphPreview.tsx
M  frontend/app/dashboard/page.tsx
M  frontend/app/projects/[id]/explore/page.tsx
M  frontend/app/projects/[id]/train/page.tsx
A  frontend/app/projects/[id]/predict/page.tsx
A  frontend/lib/progress.ts
A  frontend/lib/__tests__/progress.test.ts
A  frontend/components/__tests__/AppHeader.pipeline.test.tsx
```
