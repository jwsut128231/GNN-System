"""API request/response schemas."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


VALID_TASK_TYPES = Literal[
    "node_classification", "node_regression",
    "graph_classification", "graph_regression",
]


# ── Dataset ────────────────────────────────────────────────────────────────

class ExcelSchemaEntry(BaseModel):
    xy: Literal["X", "Y"]
    level: Literal["Node", "Edge", "Graph"]
    type: str
    parameter: str
    weight: Optional[float] = None


class ExcelSchemaPayload(BaseModel):
    entries: list[ExcelSchemaEntry]
    is_heterogeneous: bool = False
    node_types: list[str] = []
    edge_types: list[str] = []


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    is_directed: bool
    task_type: str = "graph_regression"
    has_edge_attrs: bool = False
    # Excel-origin metadata
    declared_task_type: Optional[str] = None
    declared_label_column: Optional[str] = None
    schema_spec: Optional[dict] = None
    # Multi-graph / heterogeneity summary
    graph_count: int = 1
    is_heterogeneous: bool = False
    node_types: list[str] = []
    edge_types: list[str] = []
    # Multi-Y support: parallel lists. For single-Y both have length 1.
    label_columns: list[str] = []
    label_weights: list[float] = []


# ── Explore ────────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    presence_pct: float = 0.0
    low_presence_warning: bool = False
    unique_count: int
    # Populated for heterogeneous graphs (one entry per (type, column)).
    node_type: Optional[str] = None
    edge_type: Optional[str] = None


class GenericExploreData(BaseModel):
    num_nodes: int
    num_edges: int
    columns: list[ColumnInfo]
    edge_columns: list[ColumnInfo] = []
    feature_correlation: list[dict]
    correlation_columns: list[str]
    # Multi-graph / heterogeneity fields (Excel-only platform)
    graph_count: int = 1
    avg_nodes_per_graph: float = 0.0
    avg_edges_per_graph: float = 0.0
    is_heterogeneous: bool = False
    node_types: list[str] = []
    edge_types: list[str] = []
    canonical_edges: list[list[str]] = []
    per_graph_feature_schema: Optional[dict] = None
    schema_warnings: Optional[list[str]] = None


# ── Label validation / imputation ──────────────────────────────────────────

class LabelValidationRequest(BaseModel):
    task_type: VALID_TASK_TYPES
    label_column: str


class LabelValidationResult(BaseModel):
    valid: bool
    message: str
    num_classes: Optional[int] = None
    class_distribution: Optional[list[dict]] = None
    value_range: Optional[dict] = None
    is_continuous: Optional[bool] = None


class ImputationRequest(BaseModel):
    column: str
    method: Literal["mean", "median", "zero"]


class ImputationResult(BaseModel):
    column: str
    filled_count: int
    method: str


class ConfirmDataRequest(BaseModel):
    task_type: VALID_TASK_TYPES
    label_column: str


class CorrelationRequest(BaseModel):
    columns: list[str]


# ── Project ────────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    tags: list[str] = []


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    tags: Optional[list[str]] = None


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    tags: list[str]
    created_at: str
    updated_at: Optional[str] = None
    current_step: int
    status: str
    dataset_id: Optional[str] = None
    task_id: Optional[str] = None


class ProjectDetail(ProjectSummary):
    task_type: Optional[str] = None
    label_column: Optional[str] = None
    dataset_summary: Optional[DatasetSummary] = None
    task_status: Optional["TaskStatus"] = None
    dataset_ids: list[str] = []
    experiment_ids: list[str] = []


# ── Metrics ────────────────────────────────────────────────────────────────

class SplitMetrics(BaseModel):
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    mape: Optional[float] = None


class TaskResults(BaseModel):
    train_metrics: SplitMetrics
    test_metrics: SplitMetrics
    training_time_seconds: float


class BestConfig(BaseModel):
    model_name: str
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float


class LeaderboardEntry(BaseModel):
    trial: int
    model: str
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    val_loss: float


class TaskStatus(BaseModel):
    task_id: str
    project_id: Optional[str] = None
    status: Literal["QUEUED", "PREPROCESSING", "TRAINING", "COMPLETED", "FAILED"]
    progress: int
    # current_phase is a finer-grained label than ``status`` so the UI can
    # distinguish the HPO sweep from the final training run (both report
    # status="TRAINING"). Allowed values:
    #   "queued" | "preprocessing" | "hpo" | "final_training"
    #   | "completed" | "failed"
    current_phase: Optional[str] = None
    current_trial: Optional[int] = None
    total_trials: Optional[int] = None
    device: Optional[str] = None
    results: Optional[TaskResults] = None
    best_config: Optional[BestConfig] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class EpochHistory(BaseModel):
    epoch: int
    loss: float
    val_loss: float
    accuracy: Optional[float] = None
    lr: Optional[float] = None


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class NodePrediction(BaseModel):
    node_id: str
    true_label: Union[str, float]
    predicted_label: Union[str, float]
    correct: Optional[bool] = None
    confidence: Optional[float] = None


class Report(BaseModel):
    task_type: str = "graph_regression"
    train_metrics: SplitMetrics
    val_metrics: Optional[SplitMetrics] = None
    test_metrics: SplitMetrics
    history: list[EpochHistory]
    confusion_matrix: Optional[ConfusionMatrix] = None
    residual_data: Optional[list[dict]] = None
    node_predictions: Optional[list[NodePrediction]] = None
    best_config: Optional[BestConfig] = None
    leaderboard: Optional[list[LeaderboardEntry]] = None
    is_heterogeneous: bool = False
    # Multi-Y support. For single-Y label_columns has length 1 and
    # per_target_metrics / per_target_residuals stay empty (the legacy
    # test_metrics + residual_data fields carry the data).
    label_columns: list[str] = []
    per_target_metrics: dict[str, SplitMetrics] = {}
    per_target_residuals: dict[str, list[dict]] = {}


# ── Training ───────────────────────────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    models: list[str] = []
    n_trials: int = Field(default=20, ge=1, le=500)


class TrainingEstimate(BaseModel):
    estimated_seconds: float
    device: str


# ── Experiment hierarchy ──────────────────────────────────────────────────

class CreateExperimentRequest(BaseModel):
    name: str
    dataset_id: str


class ExperimentSummary(BaseModel):
    experiment_id: str
    project_id: str
    name: str
    dataset_id: str
    task_type: Optional[str] = None
    label_column: Optional[str] = None
    current_step: int = 1
    status: str = "created"
    created_at: str
    updated_at: str
    run_count: int = 0
    best_metric: Optional[float] = None
    best_model: Optional[str] = None


class ExperimentDetail(ExperimentSummary):
    dataset_summary: Optional[DatasetSummary] = None
    runs: list[TaskStatus] = []


# ── Model registry ─────────────────────────────────────────────────────────

class RegisteredModel(BaseModel):
    model_id: str
    project_id: str
    task_id: str
    name: str
    model_name: str
    task_type: str
    label_column: str
    num_features: int
    num_classes: int
    best_config: BestConfig
    train_metrics: SplitMetrics
    test_metrics: SplitMetrics
    file_path: str
    registered_at: str
    description: str = ""


class RegisterModelRequest(BaseModel):
    name: str = ""
    description: str = ""


ProjectDetail.model_rebuild()
ExperimentDetail.model_rebuild()
