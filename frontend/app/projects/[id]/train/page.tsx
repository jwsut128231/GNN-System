'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { sanitizeParam } from '@/lib/sanitize';
import {
    Button, Card, Tag, Slider, Alert, Spin, Space, Table, Tooltip, Progress, Typography, Checkbox, Row, Col, theme,
    Divider,
} from 'antd';
import {
    PlayCircleOutlined, AppstoreOutlined, ClockCircleOutlined,
    CheckCircleOutlined, HistoryOutlined, WarningOutlined, RocketOutlined,
} from '@ant-design/icons';

import {
    estimateTraining, startProjectTraining, getProjectStatus, getProject,
    listExperiments,
    TaskStatus, TrainingEstimate, ProjectDetail,
} from '@/lib/api';

const { Title, Text } = Typography;

const ALL_MODELS_HOMO = ['gcn', 'gat', 'sage', 'gin', 'mlp'];
// Heterogeneous graphs: GCN/GIN are excluded — both rely on assumptions
// (single relation type, inner MLP) that break under PyG's `to_hetero` transform.
// Backend skips them for hetero datasets, so the UI must too.
const ALL_MODELS_HETERO = ['gat', 'sage', 'mlp'];

function formatTime(seconds: number): string {
    if (seconds < 0) return '\u2014';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

export default function TrainPage() {
    const params = useParams();
    const router = useRouter();
    const projectId = sanitizeParam(params.id);
    const { token } = theme.useToken();

    const [project, setProject] = useState<ProjectDetail | null>(null);

    // v2: single checkbox list. "Select all" is the AutoML affordance — equivalent to sending
    // an empty models array to the backend (backend already treats that as "search all").
    const [selectedModels, setSelectedModels] = useState<string[]>(ALL_MODELS_HOMO);
    const [nTrials, setNTrials] = useState(150);
    const [estimate, setEstimate] = useState<TrainingEstimate | null>(null);
    const [estimateLoading, setEstimateLoading] = useState(false);

    const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
    const [training, setTraining] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);
    const pollRef = useRef<NodeJS.Timeout | null>(null);
    const logRef = useRef<HTMLDivElement>(null);
    const lastLogKey = useRef<string>('');

    const [elapsed, setElapsed] = useState(0);
    const [experiments, setExperiments] = useState<TaskStatus[]>([]);

    const hasEdgeAttrs = project?.dataset_summary?.has_edge_attrs;
    const isHetero = project?.dataset_summary?.is_heterogeneous ?? false;
    const availableModels = isHetero ? ALL_MODELS_HETERO : ALL_MODELS_HOMO;

    // Drop unsupported models (gcn / gin) when project is hetero — prevents the user
    // from picking a model the backend will silently skip during training.
    useEffect(() => {
        setSelectedModels(prev => {
            const next = prev.filter(m => availableModels.includes(m));
            return next.length === prev.length ? prev : (next.length > 0 ? next : availableModels);
        });
    }, [isHetero, availableModels]);

    useEffect(() => {
        if (!projectId) return;
        getProject(projectId).then(p => {
            setProject(p);
            if (p.task_status && p.task_status.status !== 'COMPLETED' && p.task_status.status !== 'FAILED') {
                setTaskStatus(p.task_status);
                setTraining(true);
            } else if (p.task_status) {
                setTaskStatus(p.task_status);
            }
        }).catch(console.error);
        listExperiments(projectId).then(setExperiments).catch(console.error);
    }, [projectId]);

    useEffect(() => {
        if (!projectId) return;
        let cancelled = false;
        // eslint-disable-next-line react-hooks/set-state-in-effect -- loading state for async fetch
        setEstimateLoading(true);
        estimateTraining(projectId, nTrials)
            .then(data => { if (!cancelled) setEstimate(data); })
            .catch(console.error)
            .finally(() => { if (!cancelled) setEstimateLoading(false); });
        return () => { cancelled = true; };
    }, [projectId, nTrials]);

    useEffect(() => {
        if (!training || !projectId) return;
        const poll = async () => {
            try {
                const status = await getProjectStatus(projectId);
                setTaskStatus(status);
                const key = `${status.status}|${status.progress}|${status.current_trial}`;
                if (key !== lastLogKey.current) {
                    lastLogKey.current = key;
                    const logLine = `[${new Date().toLocaleTimeString()}] ${status.status} - Progress: ${status.progress}%` +
                        (status.current_trial ? ` (Trial ${status.current_trial}/${status.total_trials})` : '');
                    setLogs(prev => [...prev, logLine]);
                }
                if (status.status === 'COMPLETED') {
                    setTraining(false);
                    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training completed!`]);
                    listExperiments(projectId).then(setExperiments).catch(console.error);
                } else if (status.status === 'FAILED') {
                    setTraining(false);
                    setError('Training failed. Check logs for details.');
                    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training FAILED`]);
                    listExperiments(projectId).then(setExperiments).catch(console.error);
                }
            } catch (err) {
                console.error(err);
            }
        };
        pollRef.current = setInterval(poll, 2000);
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, [training, projectId]);

    useEffect(() => {
        const startedAt = taskStatus?.started_at;
        const results = taskStatus?.results;
        if (!startedAt || (!training && taskStatus?.status !== 'COMPLETED')) return;
        if (taskStatus?.status === 'COMPLETED' && results) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- syncing elapsed from completed task
            setElapsed(results.training_time_seconds);
            return;
        }
        const startTime = new Date(startedAt).getTime();
        const tick = () => setElapsed(Math.floor((Date.now() - startTime) / 1000));
        tick();
        const interval = setInterval(tick, 1000);
        return () => clearInterval(interval);
    }, [taskStatus?.started_at, training, taskStatus?.status, taskStatus?.results]);

    useEffect(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [logs]);

    const handleStart = async () => {
        setError(null);
        setLogs([`[${new Date().toLocaleTimeString()}] Starting training...`]);
        lastLogKey.current = '';
        try {
            // Backwards-compat: sending [] tells backend "try all" — equivalent to AutoML.
            // Only send the explicit list when the user has narrowed the selection.
            const allSelected = selectedModels.length === availableModels.length;
            const models = allSelected ? [] : selectedModels;
            const status = await startProjectTraining(projectId, models, nTrials);
            setTaskStatus(status);
            setTraining(true);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Failed to start training');
        }
    };

    const isCompleted = taskStatus?.status === 'COMPLETED';
    const isFailed = taskStatus?.status === 'FAILED';
    const isRunning = training && !isCompleted && !isFailed;

    const progress = taskStatus?.progress || 0;
    const estimatedRemaining = progress > 0 && isRunning
        ? Math.max(0, elapsed * (100 - progress) / progress)
        : -1;

    const allSelected = selectedModels.length === availableModels.length;
    const noneSelected = selectedModels.length === 0;
    const showEdgeAttrWarning = hasEdgeAttrs && selectedModels.includes('mlp') && !allSelected;

    const toggleSelectAll = (checked: boolean) => {
        setSelectedModels(checked ? availableModels : []);
    };

    const experimentColumns = [
        { title: '#', dataIndex: 'index', key: 'index', width: 50 },
        { title: 'Model', dataIndex: 'model', key: 'model' },
        {
            title: 'Status', dataIndex: 'status', key: 'status',
            render: (v: string) => <Tag color={v === 'COMPLETED' ? 'green' : v === 'FAILED' ? 'red' : 'blue'}>{v}</Tag>,
        },
        { title: 'Metric', dataIndex: 'metric', key: 'metric' },
        { title: 'Time', dataIndex: 'time', key: 'time' },
        { title: 'Date', dataIndex: 'date', key: 'date' },
        {
            title: '', dataIndex: 'action', key: 'action',
            render: (_: unknown, record: { canView: boolean }) => record.canView ? <a>View</a> : null,
        },
    ];

    const experimentData = experiments.map((exp, i) => {
        const metric = exp.results?.test_metrics?.accuracy != null
            ? `Acc: ${(exp.results.test_metrics.accuracy * 100).toFixed(1)}%`
            : exp.results?.test_metrics?.mse != null
                ? `MSE: ${exp.results.test_metrics.mse.toFixed(4)}`
                : '\u2014';
        return {
            key: exp.task_id,
            index: i + 1,
            model: exp.best_config?.model_name?.toUpperCase() || '\u2014',
            status: exp.status,
            metric,
            time: exp.results?.training_time_seconds ? formatTime(exp.results.training_time_seconds) : '\u2014',
            date: exp.started_at ? new Date(exp.started_at).toLocaleString() : '\u2014',
            canView: exp.status === 'COMPLETED',
            taskId: exp.task_id,
        };
    });

    return (
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px' }}>
            <div className="page-header" style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
            }}>
                <div>
                    <Title level={3} style={{ margin: 0 }}>
                        <RocketOutlined style={{ marginRight: 8, color: token.colorPrimary }} />
                        Model Training
                    </Title>
                    <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                        Configure and run GNN model training with automated hyperparameter optimization.
                    </Text>
                </div>
                {experiments.length > 0 && (
                    <Tag icon={<HistoryOutlined />} color="blue" style={{ fontSize: 13, padding: '4px 12px' }}>
                        {experiments.length} Experiment{experiments.length !== 1 ? 's' : ''}
                    </Tag>
                )}
            </div>

            <Row gutter={24}>
                {/* Left: Configuration */}
                <Col xs={24} md={12}>
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                        <Card title="Model Families" size="small">
                            <Space direction="vertical" style={{ width: '100%' }}>
                                {/* v2: a single checkbox list. "Select all" is the AutoML switch — checking
                                    everything tells the backend to search across every family. Darren's
                                    feedback: IC designers don't want a separate Auto/Manual tab. */}
                                <Checkbox
                                    indeterminate={selectedModels.length > 0 && selectedModels.length < availableModels.length}
                                    checked={allSelected}
                                    onChange={(e) => toggleSelectAll(e.target.checked)}
                                >
                                    <Text strong>Select all</Text>
                                </Checkbox>
                                <Divider style={{ margin: '8px 0' }} />
                                <Checkbox.Group
                                    value={selectedModels}
                                    onChange={(vals) => setSelectedModels(vals as string[])}
                                    style={{ width: '100%' }}
                                >
                                    <Space direction="vertical" size={6}>
                                        {availableModels.map(m => (
                                            <Tooltip key={m} title={hasEdgeAttrs && m === 'mlp' ? 'MLP does not use edge attributes' : ''}>
                                                <Checkbox value={m}>{m.toUpperCase()}</Checkbox>
                                            </Tooltip>
                                        ))}
                                    </Space>
                                </Checkbox.Group>

                                {noneSelected && (
                                    <Alert
                                        type="warning"
                                        showIcon
                                        icon={<WarningOutlined />}
                                        message="Select at least one model family."
                                    />
                                )}

                                {showEdgeAttrWarning && (
                                    <Alert
                                        type="warning"
                                        showIcon
                                        icon={<WarningOutlined />}
                                        message="MLP baseline does not use edge attributes. Consider GCN, GAT, or GraphSAGE for better results with edge features."
                                    />
                                )}
                            </Space>
                        </Card>

                        <Card title="Optuna Trials" size="small">
                            <Slider
                                value={nTrials}
                                onChange={(val) => setNTrials(val)}
                                min={10}
                                max={300}
                                step={10}
                                disabled={isRunning}
                                tooltip={{ open: true }}
                            />
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Text type="secondary" style={{ fontSize: 12 }}>10</Text>
                                <Text type="secondary" style={{ fontSize: 12 }}>300</Text>
                            </div>
                        </Card>

                        <Card size="small">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Space>
                                    <ClockCircleOutlined />
                                    <Text type="secondary">Estimated Time:</Text>
                                    {estimateLoading ? (
                                        <Spin size="small" />
                                    ) : estimate ? (
                                        <Text strong>~{formatTime(estimate.estimated_seconds)}</Text>
                                    ) : null}
                                </Space>
                                <Tag icon={<AppstoreOutlined />} color={estimate?.device === 'cuda' ? 'green' : 'default'}>
                                    {estimate?.device?.toUpperCase() || 'CPU'}
                                </Tag>
                            </div>
                        </Card>

                        {!isRunning && (
                            <Button
                                type="primary"
                                size="large"
                                block
                                icon={<PlayCircleOutlined />}
                                onClick={handleStart}
                                disabled={noneSelected}
                            >
                                {experiments.length > 0 ? 'Start New Training' : 'Start Training'}
                            </Button>
                        )}

                        {isCompleted && (
                            <>
                                <Button
                                    type="primary"
                                    size="large"
                                    block
                                    icon={<CheckCircleOutlined />}
                                    onClick={() => router.push(`/projects/${projectId}/evaluate`)}
                                    style={{ background: token.colorSuccess, borderColor: token.colorSuccess }}
                                >
                                    View Latest Results
                                </Button>
                                <Button
                                    size="large"
                                    block
                                    icon={<RocketOutlined />}
                                    onClick={() => router.push(`/projects/${projectId}/models`)}
                                >
                                    Model Registry
                                </Button>
                            </>
                        )}

                        {error && <Alert type="error" showIcon message={error} />}
                    </Space>
                </Col>

                {/* Right: Progress & Logs */}
                <Col xs={24} md={12}>
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                        {taskStatus && (
                            <Card title="Training Progress" size="small" className="stat-card">
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                    <Space size={4}>
                                        <Tag color={isCompleted ? 'green' : isFailed ? 'red' : 'processing'}>
                                            {taskStatus.status}
                                        </Tag>
                                        {taskStatus.current_phase && !isCompleted && !isFailed && (
                                            <Tag color={
                                                taskStatus.current_phase === 'preprocessing' ? 'cyan' :
                                                taskStatus.current_phase === 'hpo' ? 'blue' :
                                                taskStatus.current_phase === 'final_training' ? 'geekblue' :
                                                'default'
                                            }>
                                                {taskStatus.current_phase === 'preprocessing' ? 'Phase 1/3 · Preprocessing' :
                                                 taskStatus.current_phase === 'hpo' ? 'Phase 2/3 · HPO Search' :
                                                 taskStatus.current_phase === 'final_training' ? 'Phase 3/3 · Final Training' :
                                                 taskStatus.current_phase}
                                            </Tag>
                                        )}
                                    </Space>
                                    <Text strong style={{ fontSize: 18, color: token.colorPrimary }}>
                                        {taskStatus.progress}%
                                    </Text>
                                </div>
                                <Progress
                                    percent={taskStatus.progress}
                                    showInfo={false}
                                    status={isCompleted ? 'success' : isFailed ? 'exception' : 'active'}
                                    strokeColor={
                                        isCompleted
                                            ? { from: '#10b981', to: '#34d399' }
                                            : isFailed
                                                ? token.colorError
                                                : { from: '#0891b2', to: '#06b6d4' }
                                    }
                                    strokeWidth={8}
                                />

                                <Row gutter={24} style={{ marginTop: 16 }}>
                                    <Col>
                                        <Text type="secondary" style={{ fontSize: 12 }}>Elapsed</Text>
                                        <div><Text strong>{formatTime(elapsed)}</Text></div>
                                    </Col>
                                    {estimatedRemaining >= 0 && (
                                        <Col>
                                            <Text type="secondary" style={{ fontSize: 12 }}>Remaining (est.)</Text>
                                            <div><Text strong>~{formatTime(estimatedRemaining)}</Text></div>
                                        </Col>
                                    )}
                                </Row>

                                {taskStatus.current_trial !== undefined && taskStatus.total_trials && (
                                    <Space style={{ marginTop: 12 }}>
                                        <Tag color="blue">Trial {taskStatus.current_trial} / {taskStatus.total_trials}</Tag>
                                        {taskStatus.device && (
                                            <Tag color={taskStatus.device === 'cuda' ? 'green' : 'default'}>
                                                {taskStatus.device.toUpperCase()}
                                            </Tag>
                                        )}
                                    </Space>
                                )}
                            </Card>
                        )}

                        {/* Training Log */}
                        <Card
                            title={
                                <Space>
                                    <div className={isRunning ? 'pulse-dot' : ''} style={{
                                        width: 10,
                                        height: 10,
                                        borderRadius: '50%',
                                        background: isRunning ? token.colorSuccess : token.colorBorder,
                                        boxShadow: isRunning ? `0 0 8px ${token.colorSuccessBg}` : 'none',
                                    }} />
                                    <Text type="secondary" style={{ fontSize: 12, letterSpacing: 1, fontWeight: 600 }}>TRAINING LOG</Text>
                                </Space>
                            }
                            size="small"
                        >
                            <div
                                ref={logRef}
                                style={{
                                    padding: 12,
                                    maxHeight: 400,
                                    minHeight: 200,
                                    overflowY: 'auto',
                                    background: token.colorBgLayout,
                                    borderRadius: 8,
                                    fontSize: 13,
                                    lineHeight: 1.8,
                                }}
                            >
                                {logs.length === 0 ? (
                                    <Text type="secondary">Waiting for training to start...</Text>
                                ) : (
                                    logs.map((line, i) => (
                                        <div key={i} style={{
                                            padding: '2px 0',
                                            color: line.includes('FAILED') ? token.colorError :
                                                   line.includes('completed') ? token.colorSuccess :
                                                   line.includes('Progress') ? token.colorPrimary :
                                                   token.colorText,
                                        }}>
                                            {line}
                                        </div>
                                    ))
                                )}
                            </div>
                        </Card>
                    </Space>
                </Col>
            </Row>

            {/* Experiment History */}
            {experiments.length > 0 && (
                <Card
                    title={
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <HistoryOutlined style={{ color: token.colorPrimary }} />
                            <span>Experiment History</span>
                            <Tag color="blue" style={{ marginLeft: 4 }}>{experiments.length}</Tag>
                        </div>
                    }
                    className="stat-card"
                    style={{
                        marginTop: 24,
                    }}
                >
                    <Table
                        columns={experimentColumns}
                        dataSource={experimentData}
                        pagination={experiments.length > 10 ? { pageSize: 10 } : false}
                        size="small"
                        onRow={(record) => ({
                            onClick: () => {
                                if (record.canView) {
                                    router.push(`/projects/${projectId}/evaluate?task_id=${record.taskId}`);
                                }
                            },
                            style: { cursor: record.canView ? 'pointer' : 'default' },
                        })}
                    />
                </Card>
            )}
        </div>
    );
}
