'use client';

/**
 * Predict page (v2 · new route)
 *
 * UX scaffold for running single-shot inference against a registered model in a project.
 * Implements the A-variant layout from .design-ref/project/GraphX Frontend Improvements v2.html:
 *   Left  — Input: file picker + model selector + Run button
 *   Centre — Summary strip (avg confidence · uncertain count · class distribution)
 *           + Graph visualization (hidden when node count > 200, per Darren's feedback)
 *           + Per-node predictions table
 *   Right — Confidence histogram + "Needs review" queue
 *
 * Backend wiring: no dedicated /predict endpoint exists yet. For now we reuse the last
 * task's report (`getProjectReport`) as a stand-in preview so the page renders meaningfully,
 * and mark the Run action with a TODO. Swap in POST /projects/:id/predict when it lands.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { sanitizeParam } from '@/lib/sanitize';
import {
    Card, Space, Typography, Button, Upload, Select, Alert, Tag, Table, Row, Col,
    theme, Statistic, Empty, List, Progress, Spin,
} from 'antd';
import type { UploadProps } from 'antd';
import {
    UploadOutlined, ThunderboltOutlined, WarningOutlined,
    ArrowLeftOutlined, DownloadOutlined,
} from '@ant-design/icons';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

import {
    getProject, listProjectModels, getProjectReport,
    ProjectDetail, RegisteredModel, Report, NodePrediction,
} from '@/lib/api';
import GraphPreview from '@/components/GraphPreview';
import type { GraphSampleData } from '@/lib/api';

const { Title, Text } = Typography;

// Graph visualization threshold — when predictions exceed this many nodes, drop the
// force-directed view and show list-only output to keep the browser responsive.
// Per Darren's feedback: IC designers don't need a massive graph they can't read anyway.
const GRAPH_VIZ_THRESHOLD = 200;

interface PredictionSummary {
    total: number;
    avgConfidence: number;
    lowConfidenceCount: number;
    classDistribution: Array<{ label: string; count: number }>;
    predictions: NodePrediction[];
    taskType: string;
}

function summarize(report: Report): PredictionSummary {
    const preds = report.node_predictions || [];
    const confidences = preds.map(p => p.confidence ?? 0).filter(c => c > 0);
    const avgConfidence = confidences.length
        ? confidences.reduce((s, c) => s + c, 0) / confidences.length
        : 0;
    const lowConfidenceCount = confidences.filter(c => c < 0.7).length;

    const counts = new Map<string, number>();
    preds.forEach(p => {
        const label = String(p.predicted_label);
        counts.set(label, (counts.get(label) || 0) + 1);
    });
    const classDistribution = Array.from(counts.entries())
        .map(([label, count]) => ({ label, count }))
        .sort((a, b) => b.count - a.count);

    return {
        total: preds.length,
        avgConfidence,
        lowConfidenceCount,
        classDistribution,
        predictions: preds,
        taskType: report.task_type,
    };
}

/** Bucket confidences into 10 bins (0–0.1, 0.1–0.2, …) for the right-column histogram. */
function confidenceHistogram(preds: NodePrediction[]): Array<{ bin: string; count: number }> {
    const buckets = new Array(10).fill(0);
    preds.forEach(p => {
        if (p.confidence == null) return;
        const idx = Math.min(9, Math.floor(p.confidence * 10));
        buckets[idx] += 1;
    });
    return buckets.map((count, i) => ({
        bin: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`,
        count,
    }));
}

export default function PredictPage() {
    const params = useParams();
    const router = useRouter();
    const projectId = sanitizeParam(params.id);
    const { token } = theme.useToken();

    const [project, setProject] = useState<ProjectDetail | null>(null);
    const [models, setModels] = useState<RegisteredModel[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string | undefined>();
    const [report, setReport] = useState<Report | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [inputFileName, setInputFileName] = useState<string | null>(null);

    useEffect(() => {
        if (!projectId) return;
        // eslint-disable-next-line react-hooks/set-state-in-effect -- loading state for async fetch
        setLoading(true);
        Promise.all([
            getProject(projectId),
            listProjectModels(projectId),
            // Stand-in for the future /predict endpoint. If no prior report exists, gracefully null.
            getProjectReport(projectId).catch(() => null),
        ])
            .then(([p, m, r]) => {
                setProject(p);
                setModels(m);
                if (m.length > 0) setSelectedModelId(m[0].model_id);
                setReport(r);
            })
            .catch(err => setError(err instanceof Error ? err.message : String(err)))
            .finally(() => setLoading(false));
    }, [projectId]);

    const summary = useMemo(() => (report ? summarize(report) : null), [report]);
    const showGraph = summary ? summary.total <= GRAPH_VIZ_THRESHOLD : false;
    const histogram = useMemo(() => (summary ? confidenceHistogram(summary.predictions) : []), [summary]);

    // Confidence is a classification concept. For regression tasks the model emits
    // continuous values, not class probabilities — hide all confidence-related UI.
    const isRegression = summary?.taskType?.endsWith('regression') ?? false;
    const isGraphTask = summary?.taskType?.startsWith('graph') ?? false;
    const itemLabel = isGraphTask ? 'Graph' : 'Node';

    // Reviewer queue: predictions with confidence below the 70% threshold.
    const reviewQueue = useMemo(
        () => (summary?.predictions || [])
            .filter(p => (p.confidence ?? 1) < 0.7)
            .sort((a, b) => (a.confidence ?? 0) - (b.confidence ?? 0))
            .slice(0, 20),
        [summary]
    );

    // Build a minimal GraphSampleData from node predictions so the existing GraphPreview
    // can render without a new backend call. Edges are not available from the Report type
    // — we render node-only for now and leave a TODO to fetch the graph sample alongside.
    const graphSample: GraphSampleData | null = useMemo(() => {
        if (!summary || !showGraph) return null;
        const nodes = summary.predictions.slice(0, GRAPH_VIZ_THRESHOLD).map(p => ({
            id: String(p.node_id),
            label: String(p.predicted_label),
            attributes: {
                predicted_label: String(p.predicted_label),
                true_label: String(p.true_label),
                confidence: p.confidence ?? 0,
            },
        }));
        return {
            nodes,
            edges: [],
            num_nodes_total: summary.total,
            num_edges_total: 0,
            sample_size: nodes.length,
        };
    }, [summary, showGraph]);

    const uploadProps: UploadProps = {
        beforeUpload: (file) => {
            setInputFileName(file.name);
            return false; // prevent auto-upload — just capture the name for now
        },
        maxCount: 1,
        onRemove: () => { setInputFileName(null); },
    };

    const handleRunInference = () => {
        // TODO: wire to POST /api/v1/projects/{id}/predict when the backend route lands.
        // For now the page displays the most recent training report as a preview.
        setError('Live inference API not yet implemented on the backend. Showing the latest report as a preview.');
    };

    if (loading) {
        return (
            <div style={{ padding: 48, display: 'flex', justifyContent: 'center' }}>
                <Spin size="large" />
            </div>
        );
    }

    if (!project) {
        return (
            <div style={{ padding: 24 }}>
                <Alert type="error" message="Project not found" />
            </div>
        );
    }

    if (models.length === 0) {
        return (
            <div style={{ padding: 24 }}>
                <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description={
                        <Space direction="vertical">
                            <Text strong>No registered models yet</Text>
                            <Text type="secondary">Complete training and register a model first.</Text>
                        </Space>
                    }
                >
                    <Button
                        icon={<ArrowLeftOutlined />}
                        onClick={() => router.push(`/projects/${projectId}/train`)}
                    >
                        Go to Training
                    </Button>
                </Empty>
            </div>
        );
    }

    const selectedModel = models.find(m => m.model_id === selectedModelId);

    return (
        <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                <div style={{ flex: 1 }}>
                    <Title level={3} style={{ margin: 0 }}>
                        <ThunderboltOutlined /> Predict
                    </Title>
                    <Text type="secondary">Run inference against a registered model</Text>
                </div>
                <Space>
                    <Tag color="blue">{project.task_type || 'task type unknown'}</Tag>
                </Space>
            </div>

            {error && <Alert type="info" showIcon message={error} style={{ marginBottom: 16 }} closable />}

            <Row gutter={16}>
                {/* ═══ Left · Input ═══ */}
                <Col xs={24} md={6}>
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                        <Card title="Input Data" size="small">
                            <Upload.Dragger {...uploadProps} style={{ padding: 12 }}>
                                <p><UploadOutlined style={{ fontSize: 24, color: token.colorPrimary }} /></p>
                                <p style={{ fontSize: 13, margin: 0 }}>
                                    {inputFileName || 'Drop Excel / CSV here'}
                                </p>
                            </Upload.Dragger>
                        </Card>

                        <Card title="Model" size="small">
                            <Select
                                value={selectedModelId}
                                onChange={setSelectedModelId}
                                style={{ width: '100%' }}
                                options={models.map(m => ({
                                    value: m.model_id,
                                    label: (
                                        <Space>
                                            <Text strong>{m.name || m.model_name}</Text>
                                            <Tag color="cyan">{m.model_name}</Tag>
                                        </Space>
                                    ),
                                }))}
                            />
                            {selectedModel && (
                                <div style={{ marginTop: 12, fontSize: 12 }}>
                                    <div><Text type="secondary">Features:</Text> {selectedModel.num_features}</div>
                                    {selectedModel.num_classes > 0 && (
                                        <div><Text type="secondary">Classes:</Text> {selectedModel.num_classes}</div>
                                    )}
                                    <div>
                                        <Text type="secondary">Registered:</Text>{' '}
                                        {new Date(selectedModel.registered_at).toLocaleDateString()}
                                    </div>
                                </div>
                            )}
                        </Card>

                        <Button
                            type="primary"
                            icon={<ThunderboltOutlined />}
                            size="large"
                            block
                            onClick={handleRunInference}
                            disabled={!selectedModelId}
                        >
                            Run Inference
                        </Button>

                        {report && (
                            <Button icon={<DownloadOutlined />} block>
                                Export predictions
                            </Button>
                        )}
                    </Space>
                </Col>

                {/* ═══ Centre · Results ═══ */}
                <Col xs={24} md={12}>
                    {!summary ? (
                        <Card><Empty description="No predictions yet — run inference or complete a training run." /></Card>
                    ) : (
                        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                            {/* Summary strip — regression skips confidence-derived stats. */}
                            <Row gutter={8}>
                                {!isRegression && (
                                    <>
                                        <Col span={8}>
                                            <Card size="small">
                                                <Statistic
                                                    title="Avg confidence"
                                                    value={summary.avgConfidence * 100}
                                                    precision={1}
                                                    suffix="%"
                                                />
                                            </Card>
                                        </Col>
                                        <Col span={8}>
                                            <Card size="small">
                                                <Statistic
                                                    title="Needs review"
                                                    value={summary.lowConfidenceCount}
                                                    valueStyle={{ color: summary.lowConfidenceCount > 0 ? token.colorWarning : token.colorSuccess }}
                                                />
                                            </Card>
                                        </Col>
                                    </>
                                )}
                                <Col span={isRegression ? 24 : 8}>
                                    <Card size="small">
                                        <Statistic title="Predictions" value={summary.total} />
                                    </Card>
                                </Col>
                            </Row>

                            {/* Graph view (node-count gated) */}
                            {showGraph && graphSample ? (
                                <Card size="small" title="Graph view">
                                    <GraphPreview graphSample={graphSample} height={360} />
                                </Card>
                            ) : (
                                <Alert
                                    type="warning"
                                    showIcon
                                    icon={<WarningOutlined />}
                                    message={`Graph too large for interactive preview (${summary.total} nodes > ${GRAPH_VIZ_THRESHOLD}).`}
                                    description="Showing list-only predictions below. Use Export or go back to Analyze for a sampled view."
                                />
                            )}

                            {/* Per-item predictions table — column set depends on task. */}
                            <Card size="small" title="Predictions">
                                <Table<NodePrediction>
                                    size="small"
                                    rowKey={(r) => String(r.node_id)}
                                    dataSource={summary.predictions}
                                    pagination={{ pageSize: 20, showSizeChanger: true }}
                                    columns={[
                                        { title: itemLabel, dataIndex: 'node_id', key: 'node_id', width: 100 },
                                        { title: 'True', dataIndex: 'true_label', key: 'true_label', width: 110 },
                                        {
                                            title: 'Predicted',
                                            dataIndex: 'predicted_label',
                                            key: 'predicted_label',
                                            render: (label: string | number, row) => (
                                                <Tag color={row.correct === false ? 'red' : row.correct === true ? 'green' : 'default'}>
                                                    {label}
                                                </Tag>
                                            ),
                                            width: 130,
                                        },
                                        ...(isRegression ? [] : [{
                                            title: 'Confidence',
                                            dataIndex: 'confidence',
                                            key: 'confidence',
                                            render: (c?: number) => c != null ? (
                                                <Progress
                                                    percent={c * 100}
                                                    size="small"
                                                    format={(v) => `${(v ?? 0).toFixed(1)}%`}
                                                    status={c < 0.7 ? 'exception' : 'success'}
                                                />
                                            ) : '—',
                                        }]),
                                    ]}
                                />
                            </Card>
                        </Space>
                    )}
                </Col>

                {/* ═══ Right · Confidence + Review Queue (classification only) ═══ */}
                <Col xs={24} md={6} style={{ display: isRegression ? 'none' : undefined }}>
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                        <Card title="Confidence histogram" size="small">
                            {histogram.length > 0 ? (
                                <ResponsiveContainer width="100%" height={180}>
                                    <BarChart data={histogram}>
                                        <CartesianGrid strokeDasharray="3 3" stroke={token.colorBorderSecondary} />
                                        <XAxis dataKey="bin" tick={{ fontSize: 9 }} />
                                        <YAxis tick={{ fontSize: 10 }} />
                                        <Tooltip />
                                        <Bar dataKey="count" fill={token.colorPrimary} />
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No data" />}
                        </Card>

                        <Card
                            title={`Needs review (${reviewQueue.length})`}
                            size="small"
                            extra={<Tag color="orange">conf &lt; 0.70</Tag>}
                        >
                            {reviewQueue.length > 0 ? (
                                <List
                                    size="small"
                                    dataSource={reviewQueue}
                                    renderItem={(p) => (
                                        <List.Item key={String(p.node_id)} style={{ padding: '6px 0' }}>
                                            <Space size={4} style={{ width: '100%', justifyContent: 'space-between' }}>
                                                <Text style={{ fontSize: 12 }}>
                                                    #{String(p.node_id).slice(0, 10)}
                                                </Text>
                                                <Tag color="orange" style={{ margin: 0 }}>
                                                    {((p.confidence ?? 0) * 100).toFixed(0)}%
                                                </Tag>
                                            </Space>
                                        </List.Item>
                                    )}
                                />
                            ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="All confident" />}
                        </Card>
                    </Space>
                </Col>
            </Row>
        </div>
    );
}
