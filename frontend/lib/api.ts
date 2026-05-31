const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ══════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════

export interface ExcelSchemaEntry {
  xy: 'X' | 'Y';
  level: 'Node' | 'Edge' | 'Graph';
  type: string;
  parameter: string;
  weight: number | null;
}

export interface ExcelSchemaSpec {
  entries: ExcelSchemaEntry[];
  is_heterogeneous?: boolean;
  node_types?: string[];
  edge_types?: string[];
}

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  num_nodes: number;
  num_edges: number;
  num_features: number;
  num_classes: number;
  is_directed: boolean;
  task_type: string;
  has_edge_attrs?: boolean;
  declared_task_type?: string;
  declared_label_column?: string;
  schema_spec?: ExcelSchemaSpec;
  graph_count?: number;
  is_heterogeneous?: boolean;
  node_types?: string[];
  edge_types?: string[];
}

export interface DemoExcelInfo {
  id: string;
  name: string;
  description: string;
  filename: string;
  is_heterogeneous: boolean;
  tags: string[];
}

export interface SplitMetrics {
  accuracy: number | null;
  f1_score: number | null;
  precision: number | null;
  recall: number | null;
  mse: number | null;
  mae: number | null;
  r2_score: number | null;
  mape?: number | null;
}

export interface BestConfig {
  model_name: string;
  hidden_dim: number;
  num_layers: number;
  dropout: number;
  lr: number;
}

export interface LeaderboardEntry {
  trial: number;
  model: string;
  hidden_dim: number;
  num_layers: number;
  dropout: number;
  lr: number;
  val_loss: number;
}

export interface ProjectSummary {
  project_id: string;
  name: string;
  tags: string[];
  created_at: string;
  updated_at?: string;
  current_step: number;
  status: string;
  dataset_id?: string;
  task_id?: string;
}

export interface ProjectDetail extends ProjectSummary {
  task_type?: string;
  label_column?: string;
  dataset_summary?: DatasetSummary;
  task_status?: TaskStatus;
  dataset_ids?: string[];
  experiment_ids?: string[];
}

export interface ExperimentSummary {
  experiment_id: string;
  project_id: string;
  name: string;
  dataset_id: string;
  task_type?: string;
  label_column?: string;
  current_step: number;
  status: string;
  created_at: string;
  updated_at: string;
  run_count: number;
  best_metric?: number;
  best_model?: string;
}

export interface ExperimentDetail extends ExperimentSummary {
  dataset_summary?: DatasetSummary;
  runs: TaskStatus[];
}

export interface ColumnInfo {
  name: string;
  dtype: 'numeric' | 'categorical' | 'boolean';
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  // Set on hetero datasets where the same feature can repeat across node/edge types.
  node_type?: string | null;
  edge_type?: string | null;
  // Presence rate across graphs (fraction of graphs that contain this column).
  presence_pct?: number;
  low_presence_warning?: boolean;
}

export interface PerGraphFeatureSchemaEntry {
  graphs: Record<string, string[]>;
  union: string[];
  intersection: string[];
  presence_per_column: Record<string, number>;
  low_presence_columns: string[];
}

export interface GenericExploreData {
  num_nodes: number;
  num_edges: number;
  columns: ColumnInfo[];
  edge_columns?: ColumnInfo[];
  feature_correlation: Array<{ x: string; y: string; value: number }>;
  correlation_columns: string[];
  graph_count: number;
  avg_nodes_per_graph: number;
  avg_edges_per_graph: number;
  is_heterogeneous: boolean;
  node_types: string[];
  edge_types: string[];
  canonical_edges: string[][];
  // Per-graph feature schema breakdown (keyed by node/edge type or 'graph').
  per_graph_feature_schema?: Record<string, PerGraphFeatureSchemaEntry>;
  // Warnings from schema analysis (typo detection, low presence, etc.).
  schema_warnings?: string[];
}

export interface NumericColumnStats {
  column: string;
  dtype: 'numeric';
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
  outlier_count: number;
  distribution: Array<{ range: string; count: number }>;
}

export interface CategoricalColumnStats {
  column: string;
  dtype: 'categorical';
  value_counts: Array<{ name: string; count: number }>;
  top_value: string;
  top_count: number;
}

export type ColumnStats = NumericColumnStats | CategoricalColumnStats;

export interface LabelValidationResult {
  valid: boolean;
  message: string;
  num_classes?: number;
  class_distribution?: Array<{ label: string; count: number }>;
  value_range?: { min: number; max: number; mean: number; std: number };
  is_continuous?: boolean;
}

export type TaskPhase =
  | 'queued'
  | 'preprocessing'
  | 'hpo'
  | 'final_training'
  | 'completed'
  | 'failed';

export interface TaskStatus {
  task_id: string;
  project_id?: string;
  status: 'QUEUED' | 'PREPROCESSING' | 'TRAINING' | 'COMPLETED' | 'FAILED';
  progress: number;
  current_phase?: TaskPhase;
  current_trial?: number;
  total_trials?: number;
  device?: string;
  results?: {
    train_metrics: SplitMetrics;
    test_metrics: SplitMetrics;
    training_time_seconds: number;
  };
  best_config?: BestConfig;
  started_at?: string;
  completed_at?: string;
}

export interface TrainingEstimate {
  estimated_seconds: number;
  device: string;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface NodePrediction {
  node_id: string;
  true_label: string | number;
  predicted_label: string | number;
  correct?: boolean;
  confidence?: number;
}

export interface Report {
  task_type: string;
  train_metrics: SplitMetrics;
  val_metrics?: SplitMetrics;
  test_metrics: SplitMetrics;
  history: Array<{ epoch: number; loss: number; val_loss: number; accuracy?: number; lr?: number }>;
  confusion_matrix: ConfusionMatrix | null;
  residual_data?: Array<{ actual: number; predicted: number; error: number }>;
  node_predictions?: NodePrediction[];
  best_config?: BestConfig;
  leaderboard?: LeaderboardEntry[];
  is_heterogeneous?: boolean;
  // Multi-Y support — populated when the dataset declares >1 Y column.
  // For single-Y the fields stay empty/[] and train_metrics + test_metrics
  // + residual_data carry the data as before.
  label_columns?: string[];
  per_target_metrics?: Record<string, SplitMetrics>;
  per_target_residuals?: Record<string, Array<{ actual: number; predicted: number; error: number }>>;
}

export interface RegisteredModel {
  model_id: string;
  project_id: string;
  task_id: string;
  name: string;
  model_name: string;
  task_type: string;
  label_column: string;
  num_features: number;
  num_classes: number;
  best_config: BestConfig;
  train_metrics: SplitMetrics;
  test_metrics: SplitMetrics;
  file_path: string;
  registered_at: string;
  description: string;
}

export interface GraphSampleNode {
  id: string;
  label: string;
  node_type?: string | null;
  attributes: Record<string, number | string | null>;
}

export interface GraphSampleEdge {
  source: string;
  target: string;
  edge_type?: string | null;
  attributes: Record<string, number | string | null>;
}

export interface GraphIndexEntry {
  id: string;
  node_count: number;
  edge_count: number;
}

export interface GraphSampleData {
  nodes: GraphSampleNode[];
  edges: GraphSampleEdge[];
  num_nodes_total: number;
  num_edges_total: number;
  sample_size: number;
  graph_names?: string[];
  current_graph?: string | null;
  is_heterogeneous?: boolean;
  node_types?: string[];
  edge_types?: string[];
  // Structured graph index with node/edge counts per graph (preferred over graph_names).
  graph_index?: GraphIndexEntry[];
}

// ══════════════════════════════════════════════
// Project CRUD
// ══════════════════════════════════════════════

export const createProject = async (name: string, tags: string[]): Promise<ProjectSummary> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, tags }),
  });
  if (!res.ok) throw new Error(`Create project failed: ${res.statusText}`);
  return res.json();
};

export const listProjects = async (): Promise<ProjectSummary[]> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/`);
  if (!res.ok) throw new Error('List projects failed');
  return res.json();
};

export const getProject = async (projectId: string): Promise<ProjectDetail> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}`);
  if (!res.ok) throw new Error('Get project failed');
  return res.json();
};

export const deleteProject = async (projectId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Delete project failed');
};

export const updateProject = async (
  projectId: string,
  data: { name?: string; tags?: string[] },
): Promise<ProjectSummary> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

// ══════════════════════════════════════════════
// Excel Upload (single ingress path)
// ══════════════════════════════════════════════

export const uploadProjectExcel = async (
  projectId: string,
  file: File,
  datasetName: string = '',
): Promise<DatasetSummary> => {
  const formData = new FormData();
  formData.append('file', file);
  if (datasetName) formData.append('dataset_name', datasetName);
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/upload-excel`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Excel upload failed: ${res.statusText}`);
  }
  return res.json();
};

export const downloadSampleExcel = (): string =>
  `${API_BASE}/api/v1/projects/sample-excel`;

export const listDemoExcels = async (): Promise<DemoExcelInfo[]> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/demo-excels`);
  if (!res.ok) throw new Error('Failed to list demo Excel datasets');
  return res.json();
};

export const loadDemoExcel = async (
  projectId: string,
  demoId: string,
): Promise<DatasetSummary> => {
  const res = await fetch(
    `${API_BASE}/api/v1/projects/${projectId}/load-demo-excel?demo_id=${encodeURIComponent(demoId)}`,
    { method: 'POST' },
  );
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || 'Load demo failed');
  }
  return res.json();
};

export const downloadDemoExcel = (demoId: string): string =>
  `${API_BASE}/api/v1/projects/demo-excel/${encodeURIComponent(demoId)}`;

// ══════════════════════════════════════════════
// Explore
// ══════════════════════════════════════════════

export const getProjectExplore = async (projectId: string): Promise<GenericExploreData> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/explore`);
  if (!res.ok) throw new Error('Explore failed');
  return res.json();
};

export const analyzeColumn = async (
  projectId: string,
  columnName: string,
  overrideType?: string,
): Promise<ColumnStats> => {
  const params = new URLSearchParams();
  if (overrideType) params.set('override_type', overrideType);
  const url = `${API_BASE}/api/v1/projects/${encodeURIComponent(projectId)}/columns/${encodeURIComponent(columnName)}${params.toString() ? '?' + params : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Analyze column failed');
  return res.json();
};

export const getCorrelation = async (
  projectId: string,
  columns: string[],
): Promise<Array<{ x: string; y: string; value: number }>> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/correlation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ columns }),
  });
  if (!res.ok) throw new Error('Correlation failed');
  return res.json();
};

export const validateLabel = async (
  projectId: string,
  taskType: string,
  labelColumn: string,
): Promise<LabelValidationResult> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/validate-label`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_type: taskType, label_column: labelColumn }),
  });
  if (!res.ok) throw new Error('Validate label failed');
  return res.json();
};

export const imputeMissing = async (
  projectId: string,
  column: string,
  method: string,
): Promise<{ column: string; filled_count: number; method: string }> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/impute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ column, method }),
  });
  if (!res.ok) throw new Error('Impute failed');
  return res.json();
};

export const confirmData = async (
  projectId: string,
  taskType: string,
  labelColumn: string,
): Promise<ProjectSummary> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_type: taskType, label_column: labelColumn }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || 'Confirm failed');
  }
  return res.json();
};

export const getProjectGraphSample = async (
  projectId: string,
  opts: { limit?: number; graph_name?: string } | number = {},
  legacyGraphName?: string,
): Promise<GraphSampleData> => {
  // Support both old signature (projectId, limit, graphName) and new (projectId, opts).
  let limit = 500;
  let graphName: string | undefined;
  if (typeof opts === 'number') {
    limit = opts;
    graphName = legacyGraphName;
  } else {
    limit = opts.limit ?? 500;
    graphName = opts.graph_name;
  }
  const params = new URLSearchParams({ limit: String(limit) });
  if (graphName) params.set('graph_name', graphName);
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/graph-sample?${params}`);
  if (!res.ok) throw new Error('Failed to get graph sample');
  return res.json();
};

// ══════════════════════════════════════════════
// Training / Experiments / Report
// ══════════════════════════════════════════════

export const estimateTraining = async (
  projectId: string,
  nTrials: number,
): Promise<TrainingEstimate> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/estimate?n_trials=${nTrials}`);
  if (!res.ok) throw new Error('Estimate failed');
  return res.json();
};

export const startProjectTraining = async (
  projectId: string,
  models: string[],
  nTrials: number,
): Promise<TaskStatus> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ models, n_trials: nTrials }),
  });
  if (!res.ok) throw new Error('Start training failed');
  return res.json();
};

export const getProjectStatus = async (projectId: string): Promise<TaskStatus> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/status`);
  if (!res.ok) throw new Error('Status failed');
  return res.json();
};

export const getProjectReport = async (projectId: string): Promise<Report> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/report`);
  if (!res.ok) throw new Error('Report failed');
  return res.json();
};

export const listExperiments = async (projectId: string): Promise<TaskStatus[]> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/experiments`);
  if (!res.ok) throw new Error('Failed to list experiments');
  return res.json();
};

export const createExperiment = async (
  projectId: string,
  name: string,
  datasetId: string,
): Promise<ExperimentSummary> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, dataset_id: datasetId }),
  });
  if (!res.ok) throw new Error('Failed to create experiment');
  return res.json();
};

export const listProjectExperiments = async (projectId: string): Promise<ExperimentSummary[]> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/experiments/list`);
  if (!res.ok) throw new Error('Failed to list experiments');
  return res.json();
};

export const getExperimentDetail = async (
  projectId: string,
  experimentId: string,
): Promise<ExperimentDetail> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/experiments/${experimentId}`);
  if (!res.ok) throw new Error('Failed to get experiment');
  return res.json();
};

export const deleteExperiment = async (
  projectId: string,
  experimentId: string,
): Promise<void> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/experiments/${experimentId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete experiment');
};

export const getExperimentReport = async (projectId: string, taskId: string): Promise<Report> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/report/${taskId}`);
  if (!res.ok) throw new Error('Failed to get experiment report');
  return res.json();
};

// ══════════════════════════════════════════════
// Model registry
// ══════════════════════════════════════════════

export const listProjectModels = async (projectId: string): Promise<RegisteredModel[]> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/models`);
  if (!res.ok) throw new Error('Failed to list models');
  return res.json();
};

export const getModelDetail = async (projectId: string, modelId: string): Promise<RegisteredModel> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/models/${modelId}`);
  if (!res.ok) throw new Error('Failed to get model');
  return res.json();
};

export const updateModelInfo = async (
  projectId: string,
  modelId: string,
  data: { name?: string; description?: string },
): Promise<RegisteredModel> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/models/${modelId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update model');
  return res.json();
};

export const deleteModel = async (projectId: string, modelId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/api/v1/projects/${projectId}/models/${modelId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete model');
};
