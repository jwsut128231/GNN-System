'use client';

import React, { useState, useEffect } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import { sanitizeParam } from '@/lib/sanitize';
import { Card, Tag, Spin, Table, Space, Alert, Statistic, Row, Col, Typography, theme, Button, Input, Select } from 'antd';
import { TrophyOutlined, RocketOutlined, ArrowRightOutlined, CheckCircleOutlined, CloseCircleOutlined, SearchOutlined } from '@ant-design/icons';

import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
    ScatterChart, Scatter, ReferenceLine,
} from 'recharts';

import { getProjectReport, getExperimentReport, Report, SplitMetrics, NodePrediction } from '@/lib/api';

const { Title, Text } = Typography;

function MetricCard({ label, value, color }: { label: string; value: number | null; color?: string }) {
    if (value === null || value === undefined) return null;
    return (
        <Card size="small">
            <Statistic
                title={label}
                value={typeof value === 'number' ? (value < 1 ? value.toFixed(4) : value.toFixed(2)) : value}
                valueStyle={color ? { color } : undefined}
            />
        </Card>
    );
}

function MetricsRow({ label, metrics }: { label: string; metrics: SplitMetrics }) {
    const isClassification = metrics.accuracy !== null;
    return (
        <div>
            <Text strong style={{ display: 'block', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 2, fontSize: 12 }}>
                {label}
            </Text>
            <Row gutter={[12, 12]}>
                {isClassification ? (
                    <>
                        <Col xs={12} sm={6}><MetricCard label="Accuracy" value={metrics.accuracy} /></Col>
                        <Col xs={12} sm={6}><MetricCard label="F1 Score" value={metrics.f1_score} /></Col>
                        <Col xs={12} sm={6}><MetricCard label="Precision" value={metrics.precision} /></Col>
                        <Col xs={12} sm={6}><MetricCard label="Recall" value={metrics.recall} /></Col>
                    </>
                ) : (
                    <>
                        <Col xs={12} sm={6}><MetricCard label="MSE" value={metrics.mse} /></Col>
                        <Col xs={12} sm={6}><MetricCard label="MAE" value={metrics.mae} /></Col>
                        <Col xs={12} sm={6}><MetricCard label="R² Score" value={metrics.r2_score} /></Col>
                        <Col xs={12} sm={6}>
                            <Card size="small">
                                <Statistic
                                    title="MAPE"
                                    value={metrics.mape == null ? 'N/A' : `${(metrics.mape * 100).toFixed(2)}%`}
                                />
                            </Card>
                        </Col>
                    </>
                )}
            </Row>
        </div>
    );
}

export default function EvaluatePage() {
    const params = useParams();
    const searchParams = useSearchParams();
    const router = useRouter();
    const projectId = sanitizeParam(params.id);
    const taskIdParam = searchParams.get('task_id');
    const { token } = theme.useToken();

    const [report, setReport] = useState<Report | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [predFilter, setPredFilter] = useState<string>('all');
    const [predSearch, setPredSearch] = useState('');
    const [predPageSize, setPredPageSize] = useState(20);

    useEffect(() => {
        if (!projectId) return;
        let cancelled = false;
        const fetchReport = taskIdParam
            ? getExperimentReport(projectId, taskIdParam)
            : getProjectReport(projectId);
        fetchReport
            .then(data => { if (!cancelled) setReport(data); })
            .catch(err => { if (!cancelled) setError(err.message); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, [projectId, taskIdParam]);

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '64px 0' }}>
                <Spin size="large" />
            </div>
        );
    }

    if (error || !report) {
        return (
            <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px' }}>
                <Alert type="warning" showIcon message={error || 'No training results yet. Please train a model first.'} />
            </div>
        );
    }

    const isClassification = report.task_type.includes('classification');

    const bestConfigColumns = [
        { title: 'Model', dataIndex: 'model', key: 'model' },
        { title: 'Hidden Dim', dataIndex: 'hidden_dim', key: 'hidden_dim' },
        { title: 'Num Layers', dataIndex: 'num_layers', key: 'num_layers' },
        { title: 'Dropout', dataIndex: 'dropout', key: 'dropout' },
        { title: 'Learning Rate', dataIndex: 'lr', key: 'lr' },
    ];

    const bestConfigData = report.best_config ? [{
        key: '1',
        model: report.best_config.model_name.toUpperCase(),
        hidden_dim: report.best_config.hidden_dim,
        num_layers: report.best_config.num_layers,
        dropout: report.best_config.dropout,
        lr: report.best_config.lr,
    }] : [];

    const leaderboardColumns = [
        {
            title: 'Rank', dataIndex: 'rank', key: 'rank',
            render: (_: unknown, __: unknown, i: number) => i === 0 ? '\ud83e\udd47' : i === 1 ? '\ud83e\udd48' : i === 2 ? '\ud83e\udd49' : `#${i + 1}`,
        },
        { title: 'Trial', dataIndex: 'trial', key: 'trial' },
        { title: 'Model', dataIndex: 'model', key: 'model', render: (v: string) => <Text strong>{v.toUpperCase()}</Text> },
        { title: 'Hidden Dim', dataIndex: 'hidden_dim', key: 'hidden_dim' },
        { title: 'Layers', dataIndex: 'num_layers', key: 'num_layers' },
        { title: 'Dropout', dataIndex: 'dropout', key: 'dropout' },
        { title: 'LR', dataIndex: 'lr', key: 'lr' },
        { title: 'Val Loss', dataIndex: 'val_loss', key: 'val_loss', render: (v: number) => v.toFixed(4) },
    ];

    const leaderboardData = (report.leaderboard || []).map((entry) => ({
        key: `${entry.trial}`,
        trial: entry.trial,
        model: entry.model,
        hidden_dim: entry.hidden_dim,
        num_layers: entry.num_layers,
        dropout: entry.dropout,
        lr: entry.lr,
        val_loss: entry.val_loss,
    }));

    // Per-node prediction table
    const nodePredictions = report.node_predictions || [];
    const filteredPredictions = nodePredictions.filter(p => {
        if (predFilter === 'correct' && p.correct !== true) return false;
        if (predFilter === 'incorrect' && p.correct !== false) return false;
        if (predSearch && !p.node_id.toLowerCase().includes(predSearch.toLowerCase())) return false;
        return true;
    });

    const correctCount = nodePredictions.filter(p => p.correct === true).length;
    const incorrectCount = nodePredictions.filter(p => p.correct === false).length;

    const predColumns = [
        {
            title: 'Node ID', dataIndex: 'node_id', key: 'node_id',
            sorter: (a: NodePrediction, b: NodePrediction) => a.node_id.localeCompare(b.node_id, undefined, { numeric: true }),
        },
        { title: 'True Label', dataIndex: 'true_label', key: 'true_label' },
        { title: 'Predicted', dataIndex: 'predicted_label', key: 'predicted_label' },
        ...(isClassification ? [{
            title: 'Correct', dataIndex: 'correct', key: 'correct',
            render: (v: boolean) => v
                ? <Tag color="success" icon={<CheckCircleOutlined />}>Yes</Tag>
                : <Tag color="error" icon={<CloseCircleOutlined />}>No</Tag>,
            sorter: (a: NodePrediction, b: NodePrediction) => (a.correct === b.correct ? 0 : a.correct ? -1 : 1),
        }] : []),
        ...(nodePredictions.some(p => p.confidence != null) ? [{
            title: 'Confidence', dataIndex: 'confidence', key: 'confidence',
            render: (v: number | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '—',
            sorter: (a: NodePrediction, b: NodePrediction) => (a.confidence || 0) - (b.confidence || 0),
        }] : []),
    ];

    return (
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px' }}>
            <div className="page-header" style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
            }}>
                <div>
                    <Title level={3} style={{ margin: 0 }}>
                        <TrophyOutlined style={{ marginRight: 8, color: token.colorWarning }} />
                        Model Evaluation
                    </Title>
                    <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                        Task: {report.task_type.replace('_', ' ').toUpperCase()}
                    </Text>
                </div>
                <Space>
                    {report.best_config && (
                        <Tag icon={<TrophyOutlined />} color="gold" style={{ fontSize: 13, padding: '4px 12px' }}>
                            Best: {report.best_config.model_name.toUpperCase()}
                        </Tag>
                    )}
                    <Button
                        type="primary"
                        icon={<RocketOutlined />}
                        onClick={() => router.push(`/projects/${projectId}/models`)}
                    >
                        Model Registry <ArrowRightOutlined />
                    </Button>
                </Space>
            </div>

            <Space direction="vertical" size="large" style={{ width: '100%' }}>
                {/* Performance Metrics */}
                <Card
                    title={
                        report.label_columns && report.label_columns.length > 1
                            ? `Performance Metrics — Aggregate across ${report.label_columns.length} Y targets`
                            : "Performance Metrics"
                    }
                >
                    <Space direction="vertical" size="large" style={{ width: '100%' }}>
                        <MetricsRow label="Training" metrics={report.train_metrics} />
                        {report.val_metrics && (
                            <MetricsRow label="Validation" metrics={report.val_metrics} />
                        )}
                        <MetricsRow label="Test" metrics={report.test_metrics} />
                    </Space>
                </Card>

                {/* Per-Target Test Metrics (multi-Y only) */}
                {report.per_target_metrics
                    && Object.keys(report.per_target_metrics).length > 1 && (
                    <Card
                        title="Per-Target Test Metrics"
                        extra={
                            <Tag color="processing">
                                {Object.keys(report.per_target_metrics).length} Y targets
                            </Tag>
                        }
                    >
                        <Space direction="vertical" size="large" style={{ width: '100%' }}>
                            {Object.entries(report.per_target_metrics).map(([col, metrics]) => (
                                <div key={col}>
                                    <Text strong style={{ display: 'block', marginBottom: 8 }}>
                                        Y = {col}
                                    </Text>
                                    <MetricsRow label="Test" metrics={metrics} />
                                </div>
                            ))}
                        </Space>
                    </Card>
                )}

                {/* Confusion Matrix */}
                {isClassification && report.confusion_matrix && (
                    <Card title="Confusion Matrix">
                        {(() => {
                            const { labels, matrix } = report.confusion_matrix;
                            const n = labels.length;
                            const maxVal = Math.max(...matrix.flat(), 1);
                            return (
                                <div style={{ overflowX: 'auto' }}>
                                    <div style={{
                                        display: 'inline-flex',
                                        flexDirection: 'column',
                                        gap: 0,
                                        margin: '0 auto',
                                        minWidth: Math.max(500, 160 + n * 100),
                                    }}>
                                        <div style={{ textAlign: 'center', marginBottom: 8, paddingLeft: 160 }}>
                                            <Text type="secondary" style={{ fontSize: 13, fontWeight: 600, letterSpacing: 1 }}>PREDICTED</Text>
                                        </div>
                                        <div style={{ display: 'grid', gridTemplateColumns: `160px repeat(${n}, minmax(90px, 1fr))`, gap: 6 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 10 }}>
                                                <Text type="secondary" style={{ fontSize: 12 }}>Actual \ Pred.</Text>
                                            </div>
                                            {labels.map(label => (
                                                <div key={label} style={{ textAlign: 'center', padding: 10 }}>
                                                    <Text strong style={{ fontSize: 13 }}>{label}</Text>
                                                </div>
                                            ))}
                                        </div>
                                        {matrix.map((row, i) => (
                                            <div key={i} style={{ display: 'grid', gridTemplateColumns: `160px repeat(${n}, minmax(90px, 1fr))`, gap: 6 }}>
                                                <div style={{ display: 'flex', alignItems: 'center', padding: '10px 12px' }}>
                                                    <Text strong style={{ fontSize: 13 }}>{labels[i]}</Text>
                                                </div>
                                                {row.map((val, j) => {
                                                    const isDiag = i === j;
                                                    const intensity = val / maxVal;
                                                    const bg = isDiag
                                                        ? `color-mix(in srgb, ${token.colorSuccess} ${Math.round(intensity * 60 + 15)}%, ${token.colorBgContainer})`
                                                        : val > 0
                                                            ? `color-mix(in srgb, ${token.colorError} ${Math.round(intensity * 60 + 15)}%, ${token.colorBgContainer})`
                                                            : token.colorBgContainer;
                                                    return (
                                                        <div key={j} style={{
                                                            textAlign: 'center',
                                                            padding: '18px 14px',
                                                            borderRadius: token.borderRadius,
                                                            background: bg,
                                                            border: isDiag
                                                                ? `2px solid ${token.colorSuccess}40`
                                                                : `1px solid ${token.colorBorderSecondary}`,
                                                        }}>
                                                            <div style={{ fontSize: 20, fontWeight: 700, margin: 0, lineHeight: 1.2 }}>{val}</div>
                                                            <Text type="secondary" style={{ fontSize: 11 }}>
                                                                {(val / row.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
                                                            </Text>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        ))}
                                    </div>
                                    <div style={{ position: 'relative', marginTop: 8 }}>
                                        <Text type="secondary" style={{ fontSize: 13, fontWeight: 600, letterSpacing: 1, paddingLeft: 40 }}>ACTUAL</Text>
                                    </div>
                                </div>
                            );
                        })()}
                    </Card>
                )}

                {/* Per-Node Predictions Table */}
                {nodePredictions.length > 0 && (
                    <Card
                        title={
                            <Space>
                                <span>Per-Node Predictions</span>
                                <Tag color="blue">{nodePredictions.length} nodes</Tag>
                                {isClassification && (
                                    <>
                                        <Tag color="success">{correctCount} correct</Tag>
                                        <Tag color="error">{incorrectCount} incorrect</Tag>
                                    </>
                                )}
                            </Space>
                        }
                        extra={
                            <Space>
                                <Input
                                    placeholder="Search Node ID"
                                    prefix={<SearchOutlined />}
                                    value={predSearch}
                                    onChange={e => setPredSearch(e.target.value)}
                                    style={{ width: 160 }}
                                    size="small"
                                    allowClear
                                />
                                {isClassification && (
                                    <Select
                                        value={predFilter}
                                        onChange={setPredFilter}
                                        size="small"
                                        style={{ width: 120 }}
                                        options={[
                                            { value: 'all', label: 'All' },
                                            { value: 'correct', label: 'Correct' },
                                            { value: 'incorrect', label: 'Incorrect' },
                                        ]}
                                    />
                                )}
                            </Space>
                        }
                    >
                        <Table
                            columns={predColumns}
                            dataSource={filteredPredictions.map((p, i) => ({ ...p, key: `${p.node_id}-${i}` }))}
                            pagination={{
                                pageSize: predPageSize,
                                showSizeChanger: true,
                                pageSizeOptions: ['10', '20', '50', '100'],
                                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total}`,
                                onShowSizeChange: (_, size) => setPredPageSize(size),
                            }}
                            size="small"
                            scroll={{ y: 500 }}
                            rowClassName={(record: NodePrediction) =>
                                record.correct === true ? '' : record.correct === false ? 'ant-table-row-selected' : ''
                            }
                        />
                    </Card>
                )}

                {/* Residual Plot(s) — Error vs Predicted with y=0 reference */}
                {!isClassification && report.per_target_residuals
                    && Object.keys(report.per_target_residuals).length > 1 ? (
                    /* Multi-Y: one plot per target */
                    Object.entries(report.per_target_residuals).map(([col, points]) => {
                        const maxAbs = Math.max(...points.map((d) => Math.abs(d.error ?? 0))) || 1;
                        return (
                            <Card key={col} title={`Residual Plot — ${col} (Error vs Predicted)`}>
                                <ResponsiveContainer width="100%" height={350}>
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis type="number" dataKey="predicted" name="Predicted" tick={{ fontSize: 11 }} label={{ value: 'Predicted', position: 'insideBottom', offset: -10 }} />
                                        <YAxis type="number" dataKey="error" name="Error" domain={[-maxAbs, maxAbs]} tick={{ fontSize: 11 }} label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                        <ReferenceLine y={0} stroke="red" strokeDasharray="3 3" />
                                        <Scatter data={points} fill={token.colorPrimary} opacity={0.6} />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </Card>
                        );
                    })
                ) : (
                    !isClassification && report.residual_data && report.residual_data.length > 0 && (() => {
                        const maxAbs = Math.max(...report.residual_data.map((d) => Math.abs(d.error ?? 0))) || 1;
                        return (
                            <Card title="Residual Plot (Error vs Predicted)">
                                <ResponsiveContainer width="100%" height={350}>
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis type="number" dataKey="predicted" name="Predicted" tick={{ fontSize: 11 }} label={{ value: 'Predicted', position: 'insideBottom', offset: -10 }} />
                                        <YAxis type="number" dataKey="error" name="Error" domain={[-maxAbs, maxAbs]} tick={{ fontSize: 11 }} label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                        <ReferenceLine y={0} stroke="red" strokeDasharray="3 3" />
                                        <Scatter data={report.residual_data} fill={token.colorPrimary} opacity={0.6} />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </Card>
                        );
                    })()
                )}

                {/* Training History */}
                {report.history && report.history.length > 0 && (
                    <Card title="Training History">
                        <ResponsiveContainer width="100%" height={340}>
                            <LineChart data={report.history} margin={{ top: 5, right: 10, bottom: 25, left: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -15 }} />
                                <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                                {isClassification && (
                                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0, 1]} />
                                )}
                                <Tooltip />
                                <Legend verticalAlign="top" />
                                <Line type="monotone" dataKey="loss" name="Train Loss" stroke={token.colorPrimary} strokeWidth={2} dot={false} yAxisId="left" />
                                <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke={token.colorInfo} strokeWidth={2} dot={false} yAxisId="left" />
                                {isClassification && (
                                    <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke={token.colorSuccess} strokeWidth={2} dot={false} yAxisId="right" />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </Card>
                )}

                {/* Best Config */}
                {report.best_config && (
                    <Card title="Best Model Configuration">
                        <Table columns={bestConfigColumns} dataSource={bestConfigData} pagination={false} size="small" />
                    </Card>
                )}

                {/* Leaderboard */}
                {report.leaderboard && report.leaderboard.length > 0 && (
                    <Card title="Training Leaderboard">
                        <Table
                            columns={leaderboardColumns}
                            dataSource={leaderboardData}
                            pagination={false}
                            size="small"
                            scroll={{ y: 500 }}
                            rowClassName={(_, i) => i === 0 ? 'ant-table-row-selected' : ''}
                        />
                    </Card>
                )}
            </Space>
        </div>
    );
}
