# GraphX Frontend v2 — 技術文件

## What Changed

### Modified files

| 檔案 | 變更重點 |
|---|---|
| `frontend/components/AppHeader.tsx` | `STEPS` / `STEP_PATHS` 改為 6 步驟 (Create/Upload/Analyze/Train/Evaluate/Predict)、加入 lock/check icon、`maxReachableIndex` 支援 Predict 解鎖 |
| `frontend/components/GraphPreview.tsx` | 加上 optional `height` prop (default 420),讓 Explore 的全螢幕 Modal 可以給更大的 canvas |
| `frontend/app/dashboard/page.tsx` | 新增 Grid/List Segmented toggle (`localStorage` 持久化)、新增 `ProjectListTable`、用 `lib/progress.ts` 取代本地 step labels |
| `frontend/app/projects/[id]/explore/page.tsx` | Graph Preview 加 Fullscreen Modal、Select 改成 searchable、新增 `DataQualityCard` + `deriveQualityChecks` (12 項圖專屬檢查) |
| `frontend/app/projects/[id]/train/page.tsx` | 移除 Auto/Manual Switch → 單一 checkbox list (含 Select all)、新增 Objective Select 並依 task_type 動態填 options |

### Added files

| 檔案 | 用途 |
|---|---|
| `frontend/app/projects/[id]/predict/page.tsx` | 全新的 Predict 頁面 (UI scaffold);> 200 節點自動切回 list-only |
| `frontend/lib/progress.ts` | 共用 v2 step labels + `reachedFromLegacy` 映射 helper |
| `frontend/lib/__tests__/progress.test.ts` | helper 的單元測試 (5 cases) |
| `frontend/components/__tests__/AppHeader.pipeline.test.tsx` | 6-step 配置的 invariant 測試 (4 cases) |

## Why

使用者把設計稿 (`.design-ref/project/GraphX Frontend Improvements v2.html`) 透過 Claude Design 交付。裡面和團隊夥伴 Darren 反覆迭代後的落定方向:

1. **流程一致性**:六個明確步驟、Register/Models 不應該在主線,因為它是訓練的自動產出而非使用者主動操作。
2. **IC designer 本位**:Train 頁面不要有 Auto/Manual tab 或 parallel coords — 使用者是 IC 設計師,不是 ML 工程師。
3. **大圖可用性**:Predict 超過 200 節點時前端不該硬畫,要 fallback 成清單。
4. **圖資料才有的品質檢查**:用 tabular ML 的 QA 清單無法涵蓋 connected components / self-loops / label leakage 這類 graph-specific 問題。
5. **Dashboard 要能密集瀏覽**:新手需要卡片,熟手需要 table。

## How It Works

### Pipeline stepper (6 steps)

```
┌ components/AppHeader.tsx ─────────────────────────────────────┐
│ STEPS = [Create, Upload, Analyze, Train, Evaluate, Predict]   │
│ STEP_PATHS(id) = [/dashboard, /upload, /explore, /train,      │
│                   /evaluate, /predict]                         │
│                                                                │
│  legacyStep (backend 1..5)                                    │
│    → activeIndex = min(max(legacyStep, 1), 4)  ←── clamps     │
│    → maxReachableIndex = activeIndex                          │
│        +1 if status === 'completed' && activeIndex >= 4       │
│                                                                │
│  buildStepItems():                                             │
│    i < activeIndex     → CheckOutlined (green)                │
│    i === activeIndex   → AntD default highlight               │
│    i > maxReachableIndex → LockOutlined + disabled + Tooltip  │
└────────────────────────────────────────────────────────────────┘
```

`legacyStep` 是 backend 目前仍傳的 1..5 欄位。`AppHeader` 直接在元件內映射,backend/API 型別不用改。
legacy 5 (舊 Models) 會被 clamp 到 v2 的 Evaluate,因為 Models 不再是步驟之一。

### Dashboard view toggle

```
dashboard/page.tsx
  ├─ Segmented (Grid|List)  ──► handleViewChange ──► setView + localStorage
  │
  ├─ view === 'grid' ────► <Row> Card grid (existing)
  │                          └─ mini-rail: 6 bars + "N/6"
  │
  └─ view === 'list' ────► <ProjectListTable>
                               └─ AntD Table with <MiniRail reached={…}/>
```

`MiniRail` 和卡片版的 mini-rail 共用 `lib/progress.ts` 的 `reachedFromLegacy`,保證兩處呈現一致。

### Explore 頁面: fullscreen graph + data quality

```
explore/page.tsx
  Graph Preview Card
    extra: [<Select searchable>, <Button Fullscreen>]
    body : <GraphPreview graphSample={…} />          ← height default 420

  + <Modal width="95vw" ...>                         ← open on Fullscreen click
       <GraphPreview graphSample={…} height={大} />
    </Modal>

  + <DataQualityCard
       exploreData, graphSample, taskType, labelColumn, labelValidation
    />
    └─ deriveQualityChecks(...) → QualityCheck[] (ok/warn/err/na)
       └─ 在客戶端用 graphSample 的 nodes/edges 算 connected components,
          self-loops, isolated, duplicates, degree stats 等。
          Feature NaN 和 correlation 用 server-side exploreData。
```

`deriveQualityChecks` 是純函式,未來若要遷移到 backend 只要把輸入輸出對齊即可。

### Train 精簡配置

```
state:
  selectedModels: string[]  (default = ALL_MODELS)   ← "全選 = AutoML"
  objective: string          (default = 'val_acc', 隨 task_type 同步)

UI:
  <Checkbox indeterminate={partial} checked={allSelected}
           onChange={toggleSelectAll}>Select all (AutoML · try all)</Checkbox>
  <Divider />
  <Checkbox.Group value={selectedModels} ...>
    {ALL_MODELS.map(m => <Checkbox value={m}>{m.toUpperCase()} — {hint}</Checkbox>)}
  </Checkbox.Group>

  <Select value={objective} options={objectiveOptionsFor(task_type)} />

submit:
  models = (allSelected ? [] : selectedModels)  // [] = backend 預設 "try all"
  startProjectTraining(projectId, models, nTrials)
```

Backend 目前接收 `{ models: [], n_trials }`;傳空陣列代表搜尋全部,這個對應關係沒變。

### Predict 頁面 + 200 節點 threshold

```
Promise.all([
  getProject(projectId),
  listProjectModels(projectId),
  getProjectReport(projectId).catch(()=>null),  // 用最近 report 當 preview
])

summary = summarize(report)
  total, avgConfidence, lowConfidenceCount(conf<0.7), classDistribution,
  predictions, taskType

showGraph = summary.total <= 200
  ? <GraphPreview graphSample={由 predictions 轉出來的 nodes-only sample}/>
  : <Alert warning "Graph too large"/>

reviewQueue = predictions
  .filter(p => (p.confidence ?? 1) < 0.7)
  .sort(by confidence asc)
  .slice(0, 20)
```

`GRAPH_VIZ_THRESHOLD = 200` 是 Darren 指定的門檻。目前推論是以最近的 training report 作 stand-in,等 backend 提供 `POST /api/v1/projects/{id}/predict` 再替換 `handleRunInference` 即可。

## Usage

### 新增 progress helper

如果要在新的頁面呈現 v2 workflow 進度,不要重複實作:

```tsx
import { V2_STEP_LABELS, V2_STEP_COUNT, reachedFromLegacy } from '@/lib/progress';

const reached = reachedFromLegacy(project.current_step, project.status);
// reached ∈ [1..6]
```

### 客製 GraphPreview 高度

```tsx
<GraphPreview graphSample={...} />              // 420px (default)
<GraphPreview graphSample={...} height={720} /> // for fullscreen modal
```

### 擴充 Data Quality 檢查

在 `explore/page.tsx` 的 `deriveQualityChecks()` 底下新增一筆:

```ts
checks.push({
    key: 'your_check',
    label: 'Your human label',
    status: 'ok' | 'warn' | 'err' | 'na',
    detail: 'short one-liner shown after the label',
});
```

## Caveats

1. **Backend `current_step` 仍是 1..5**。AppHeader 在 component 層做映射。如果將來 backend 改成 1..6,直接移除 `reachedFromLegacy` 裡的 `+1` 和 clamp,比較乾淨。
2. **Predict 尚未連接真實推論 API**。`handleRunInference` 只顯示 Alert,主畫面用 `getProjectReport` 當 preview。等 backend 加上 `POST /projects/{id}/predict` 後只要替換 handler。
3. **Data Quality 的圖結構檢查用 sample (500 nodes) 估算**,所以 connected components / duplicates 這類計數是近似值;完整精確值要 backend 計算。
4. **Objective 下拉的值目前只保存在 component state**,沒送到 backend。要讓 Optuna 真的依這個 metric 搜尋,需要 `POST /projects/:id/train` 接受 `objective` 參數。
5. **GraphPreview.tsx 仍有 2 個 pre-existing lint errors**(`react-hooks/preserve-manual-memoization`),不在本次變更範圍。
6. **Dashboard List 模式的 progress column 寬度固定 190px**,非常窄的螢幕會擠;未來如果要支援 <768px 可改用 responsive column 或隱藏某些欄位。

## Verification

- `npm run build` → ✓ 靜態生成成功,`/projects/[id]/predict` route 已登錄
- `npx tsc --noEmit` → 無 source 錯誤 (只有 `.next/types/validator.ts` 的 pre-existing 錯誤,與本次無關)
- `npx jest` → 42 / 42 tests pass
- `npm run lint` → 只剩 3 個 pre-existing 的 GraphPreview 警告 (master 上就存在)
