'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { sanitizeParam } from '@/lib/sanitize';
import {
    Button, Card, Input, Space, Alert, Tag, Divider, Row, Col, Typography, theme,
} from 'antd';
import {
    CloudUploadOutlined, CheckCircleOutlined, DownloadOutlined,
    FileExcelOutlined, ExperimentOutlined, BranchesOutlined, ApartmentOutlined,
} from '@ant-design/icons';

import {
    uploadProjectExcel, downloadSampleExcel,
    listDemoExcels, loadDemoExcel, downloadDemoExcel,
    DemoExcelInfo,
} from '@/lib/api';

const { Title, Text } = Typography;

const TAG_COLORS: Record<string, string> = {
    'multi-graph': 'geekblue',
    'homogeneous': 'cyan',
    'heterogeneous': 'purple',
    'graph-regression': 'gold',
};

export default function UploadPage() {
    const params = useParams();
    const router = useRouter();
    const projectId = sanitizeParam(params.id);
    const { token } = theme.useToken();
    const excelInputRef = useRef<HTMLInputElement>(null);

    const [datasetName, setDatasetName] = useState('');
    const [excelUploading, setExcelUploading] = useState(false);
    const [loadingDemoId, setLoadingDemoId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [demos, setDemos] = useState<DemoExcelInfo[]>([]);

    useEffect(() => {
        listDemoExcels().then(setDemos).catch(console.error);
    }, []);

    const handleExcelSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setError(null);
        setExcelUploading(true);
        try {
            await uploadProjectExcel(projectId, file, datasetName);
            router.push(`/projects/${projectId}/explore`);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Excel upload failed');
        } finally {
            setExcelUploading(false);
            if (excelInputRef.current) excelInputRef.current.value = '';
        }
    };

    const handleLoadDemo = async (demoId: string) => {
        setLoadingDemoId(demoId);
        setError(null);
        try {
            await loadDemoExcel(projectId, demoId);
            router.push(`/projects/${projectId}/explore`);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Load demo failed');
        } finally {
            setLoadingDemoId(null);
        }
    };

    const isLoading = excelUploading || loadingDemoId !== null;

    return (
        <div style={{ maxWidth: 900, margin: '0 auto', padding: '40px 24px' }}>
            <div className="page-header">
                <Title level={3} style={{ margin: 0 }}>
                    <CloudUploadOutlined style={{ marginRight: 8, color: token.colorPrimary }} />
                    Upload Graph Data
                </Title>
                <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                    All data is uploaded as a single Excel workbook. The Parameter sheet declares
                    features (X) and labels (Y); task type and label column are auto-detected.
                </Text>
            </div>

            <Space direction="vertical" size="large" style={{ width: '100%', marginTop: 24 }}>
                {/* Demo Excels */}
                <Card
                    data-testid="demo-excel-card"
                    title={
                        <Space>
                            <ExperimentOutlined />
                            Demo Excel Datasets
                            <Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                                One-click load to explore the platform
                            </Text>
                        </Space>
                    }
                >
                    <Row gutter={[16, 16]}>
                        {demos.map((demo) => (
                            <Col xs={24} md={12} key={demo.id}>
                                <Card size="small">
                                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                                        <Space>
                                            {demo.is_heterogeneous ? <ApartmentOutlined /> : <BranchesOutlined />}
                                            <Text strong>{demo.name}</Text>
                                        </Space>
                                        <Text type="secondary" style={{ fontSize: 12 }}>
                                            {demo.description}
                                        </Text>
                                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                            {demo.tags.map((tag) => (
                                                <Tag key={tag} color={TAG_COLORS[tag] || 'default'}>{tag}</Tag>
                                            ))}
                                        </div>
                                        <Space>
                                            <Button
                                                size="small"
                                                type="primary"
                                                onClick={() => handleLoadDemo(demo.id)}
                                                disabled={isLoading}
                                                loading={loadingDemoId === demo.id}
                                                icon={<ExperimentOutlined />}
                                            >
                                                Load
                                            </Button>
                                            <Button
                                                size="small"
                                                href={downloadDemoExcel(demo.id)}
                                                download
                                                icon={<DownloadOutlined />}
                                            >
                                                Download
                                            </Button>
                                        </Space>
                                    </Space>
                                </Card>
                            </Col>
                        ))}
                    </Row>
                </Card>

                <Divider>OR UPLOAD YOUR OWN</Divider>

                <Input
                    placeholder="Dataset Name (optional) — auto-filled from filename"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                />

                <Card
                    data-testid="excel-upload-card"
                    title={
                        <Space>
                            <FileExcelOutlined style={{ color: token.colorSuccess }} />
                            Upload .xlsx File
                        </Space>
                    }
                    extra={
                        <Button
                            href={downloadSampleExcel()}
                            download
                            icon={<DownloadOutlined />}
                            size="small"
                        >
                            Empty Template
                        </Button>
                    }
                >
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                        <Text type="secondary" style={{ fontSize: 13 }}>
                            Fill the <code>Parameter</code> sheet to declare features and labels,
                            then fill the <code>Node</code>, <code>Edge</code>, and <code>Graph</code> sheets with data.
                            For heterogeneous graphs, add a <code>Type</code> column to <code>Node</code> / <code>Edge</code> rows
                            to distinguish node and edge types.
                        </Text>
                        <input
                            ref={excelInputRef}
                            type="file"
                            accept=".xlsx"
                            onChange={handleExcelSelect}
                            style={{ display: 'none' }}
                            data-testid="excel-file-input"
                        />
                        <Button
                            type="primary"
                            icon={excelUploading ? <CheckCircleOutlined /> : <FileExcelOutlined />}
                            size="large"
                            loading={excelUploading}
                            disabled={isLoading}
                            onClick={() => excelInputRef.current?.click()}
                            block
                        >
                            {excelUploading ? 'Uploading Excel...' : 'Select .xlsx File'}
                        </Button>
                    </Space>
                </Card>

                {error && <Alert type="error" showIcon message={error} />}
            </Space>
        </div>
    );
}
