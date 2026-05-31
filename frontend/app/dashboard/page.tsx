'use client';

import React, { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
    Button, Input, Card, Tag, Modal, Space, Skeleton, Row, Col, Typography, Empty, theme,
    Segmented, Table, Tooltip,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
    PlusOutlined, SearchOutlined, DeleteOutlined, EditOutlined,
    ExperimentOutlined, RocketOutlined, CheckCircleOutlined,
    ThunderboltOutlined, AppstoreOutlined, UnorderedListOutlined,
} from '@ant-design/icons';

import { useProject } from '@/contexts/ProjectContext';
import { useAuth } from '@/contexts/AuthContext';
import { deleteProject, updateProject, ProjectSummary } from '@/lib/api';
import AppHeader from '@/components/AppHeader';
import PageTransition from '@/components/PageTransition';

function timeAgo(dateStr: string): string {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 30) return `${days}d ago`;
    return new Date(dateStr).toLocaleDateString();
}

const { Title, Text } = Typography;

import { V2_STEP_LABELS as STEP_LABELS, V2_STEP_COUNT as STEP_COUNT, reachedFromLegacy } from '@/lib/progress';

const VIEW_STORAGE_KEY = 'dashboard.view';
type DashboardView = 'grid' | 'list';

const STATUS_TAG_COLOR: Record<string, string> = {
    created: 'default',
    data_uploaded: 'blue',
    data_confirmed: 'cyan',
    training: 'processing',
    completed: 'green',
    failed: 'red',
};

function getStepPath(project: ProjectSummary): string {
    const id = project.project_id;
    return `/projects/${id}`;
}

// Compact mini-rail used inside the list-view table cell.
function MiniRail({ reached }: { reached: number }) {
    const { token } = theme.useToken();
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ display: 'flex', gap: 2 }}>
                {STEP_LABELS.map((label, i) => (
                    <Tooltip key={label} title={label}>
                        <div style={{
                            width: 14, height: 4, borderRadius: 2,
                            background: i < reached
                                ? `linear-gradient(90deg, ${token.colorPrimary}, ${token.colorInfo})`
                                : token.colorFillSecondary,
                        }} />
                    </Tooltip>
                ))}
            </div>
            <span style={{ fontSize: 11, color: token.colorTextSecondary, fontFamily: 'monospace' }}>
                {reached}/{STEP_COUNT}
            </span>
        </div>
    );
}

interface ProjectListTableProps {
    projects: ProjectSummary[];
    onOpen: (project: ProjectSummary) => void;
    onEdit: (e: React.MouseEvent, project: ProjectSummary) => void;
    onDelete: (e: React.MouseEvent, project: ProjectSummary) => void;
}

function ProjectListTable({ projects, onOpen, onEdit, onDelete }: ProjectListTableProps) {
    const columns: ColumnsType<ProjectSummary> = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            render: (name: string) => <Typography.Text strong>{name}</Typography.Text>,
            width: 260,
        },
        {
            title: 'Tags',
            dataIndex: 'tags',
            key: 'tags',
            render: (tags?: string[]) => (
                <Space size={4} wrap>
                    {(tags || []).slice(0, 4).map((t) => <Tag key={t} style={{ borderRadius: 6 }}>{t}</Tag>)}
                    {(tags || []).length > 4 && <Tag>+{(tags || []).length - 4}</Tag>}
                </Space>
            ),
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            render: (status: string) => (
                <Tag color={STATUS_TAG_COLOR[status] || 'default'}>
                    {status.replace('_', ' ').toUpperCase()}
                </Tag>
            ),
            width: 150,
        },
        {
            title: 'Progress',
            key: 'progress',
            render: (_: unknown, record: ProjectSummary) => (
                <MiniRail reached={reachedFromLegacy(record.current_step, record.status)} />
            ),
            width: 190,
        },
        {
            title: 'Updated',
            key: 'updated',
            render: (_: unknown, record: ProjectSummary) => (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    {timeAgo(record.updated_at || record.created_at)}
                </Typography.Text>
            ),
            width: 120,
        },
        {
            title: '',
            key: 'actions',
            render: (_: unknown, record: ProjectSummary) => (
                <Space size={4} onClick={(e) => e.stopPropagation()}>
                    <Button type="text" size="small" icon={<EditOutlined />} onClick={(e) => onEdit(e, record)} />
                    <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={(e) => onDelete(e, record)} />
                </Space>
            ),
            width: 90,
            fixed: 'right',
        },
    ];

    return (
        <Table<ProjectSummary>
            rowKey="project_id"
            columns={columns}
            dataSource={projects}
            pagination={{ pageSize: 20, showSizeChanger: true }}
            onRow={(record) => ({
                onClick: () => onOpen(record),
                style: { cursor: 'pointer' },
            })}
        />
    );
}

export default function DashboardPage() {
    const router = useRouter();
    const { projects, loading, createNewProject, refreshProjects } = useProject();
    const { user } = useAuth();
    const { token } = theme.useToken();

    const [search, setSearch] = useState('');
    const [tagFilter, setTagFilter] = useState<string | null>(null);
    const [view, setView] = useState<DashboardView>('grid');

    // Restore user's last view preference (grid/list) from localStorage.
    React.useEffect(() => {
        try {
            const stored = localStorage.getItem(VIEW_STORAGE_KEY);
            if (stored === 'grid' || stored === 'list') setView(stored);
        } catch {
            // SSR / disabled localStorage — ignore
        }
    }, []);

    const handleViewChange = (v: string | number) => {
        const next = (v === 'list' ? 'list' : 'grid') as DashboardView;
        setView(next);
        try { localStorage.setItem(VIEW_STORAGE_KEY, next); } catch { /* ignore */ }
    };
    const [dialogOpen, setDialogOpen] = useState(false);
    const [newName, setNewName] = useState('');
    const [newTags, setNewTags] = useState<string[]>([]);
    const [tagInput, setTagInput] = useState('');
    const [creating, setCreating] = useState(false);
    const [editProject, setEditProject] = useState<ProjectSummary | null>(null);
    const [editName, setEditName] = useState('');
    const [saving, setSaving] = useState(false);

    const allTags = useMemo(() => {
        const tags = new Set<string>();
        projects.forEach(p => p.tags?.forEach(t => tags.add(t)));
        return Array.from(tags).sort();
    }, [projects]);

    const filtered = useMemo(() => {
        let result = projects;
        if (search) {
            const q = search.toLowerCase();
            result = result.filter(p =>
                p.name.toLowerCase().includes(q) ||
                p.tags?.some(t => t.toLowerCase().includes(q))
            );
        }
        if (tagFilter) {
            result = result.filter(p => p.tags?.includes(tagFilter));
        }
        return result;
    }, [projects, search, tagFilter]);

    const handleCreate = async () => {
        if (!newName.trim()) return;
        setCreating(true);
        try {
            const project = await createNewProject(newName.trim(), newTags);
            setDialogOpen(false);
            setNewName('');
            setNewTags([]);
            router.push(`/projects/${project.project_id}/upload`);
        } catch (err) {
            console.error(err);
        } finally {
            setCreating(false);
        }
    };

    const handleTagAdd = () => {
        const tag = tagInput.trim();
        if (tag && !newTags.includes(tag)) {
            setNewTags([...newTags, tag]);
        }
        setTagInput('');
    };

    const handleDelete = async (e: React.MouseEvent, projectId: string) => {
        e.stopPropagation();
        if (!confirm('Delete this project?')) return;
        try {
            await deleteProject(projectId);
            refreshProjects();
        } catch (err) {
            console.error(err);
        }
    };

    const [editTags, setEditTags] = useState<string[]>([]);
    const [editTagInput, setEditTagInput] = useState('');

    const handleEditOpen = (e: React.MouseEvent, project: ProjectSummary) => {
        e.stopPropagation();
        setEditProject(project);
        setEditName(project.name);
        setEditTags(project.tags || []);
        setEditTagInput('');
    };

    const handleEditTagAdd = () => {
        const tag = editTagInput.trim();
        if (tag && !editTags.includes(tag)) {
            setEditTags([...editTags, tag]);
        }
        setEditTagInput('');
    };

    const handleSaveEdit = async () => {
        if (!editProject || !editName.trim()) return;
        setSaving(true);
        try {
            await updateProject(editProject.project_id, { name: editName.trim(), tags: editTags });
            refreshProjects();
            setEditProject(null);
        } catch (err) {
            console.error(err);
        } finally {
            setSaving(false);
        }
    };

    const completedCount = projects.filter(p => p.status === 'completed').length;
    const trainingCount = projects.filter(p => p.status === 'training').length;

    return (
        <div>
            <AppHeader subtitle="PROJECT WORKSPACE" />

            <PageTransition>
            {/* Hero / Welcome Banner */}
            <div className="hero-gradient" style={{
                borderBottom: `1px solid ${token.colorBorderSecondary}`,
                padding: '36px 24px',
            }}>
                <div style={{ maxWidth: 1200, margin: '0 auto', position: 'relative', zIndex: 1 }}>
                    <Row gutter={24} align="middle">
                        <Col flex="auto">
                            <Title level={3} style={{ margin: 0 }}>
                                Welcome back{user ? `, ${user.name.split(' ')[0]}` : ''}
                            </Title>
                            <Text type="secondary" style={{ fontSize: 14, marginTop: 4, display: 'block' }}>
                                Manage your GNN training projects and experiments
                            </Text>
                        </Col>
                        <Col>
                            <Space size="large">
                                <div className="stat-card" style={{
                                    textAlign: 'center',
                                    padding: '12px 20px',
                                    borderRadius: 12,
                                    background: token.colorBgContainer,
                                    boxShadow: token.boxShadow,
                                    minWidth: 80,
                                }}>
                                    <div style={{ fontSize: 28, fontWeight: 700, color: token.colorPrimary, lineHeight: 1.2 }}>
                                        {projects.length}
                                    </div>
                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                        <ExperimentOutlined style={{ marginRight: 4 }} />Projects
                                    </Text>
                                </div>
                                <div className="stat-card stat-success" style={{
                                    textAlign: 'center',
                                    padding: '12px 20px',
                                    borderRadius: 12,
                                    background: token.colorBgContainer,
                                    boxShadow: token.boxShadow,
                                    minWidth: 80,
                                }}>
                                    <div style={{ fontSize: 28, fontWeight: 700, color: token.colorSuccess, lineHeight: 1.2 }}>
                                        {completedCount}
                                    </div>
                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                        <CheckCircleOutlined style={{ marginRight: 4 }} />Completed
                                    </Text>
                                </div>
                                <div className="stat-card stat-warning" style={{
                                    textAlign: 'center',
                                    padding: '12px 20px',
                                    borderRadius: 12,
                                    background: token.colorBgContainer,
                                    boxShadow: token.boxShadow,
                                    minWidth: 80,
                                }}>
                                    <div style={{ fontSize: 28, fontWeight: 700, color: token.colorWarning, lineHeight: 1.2 }}>
                                        {trainingCount}
                                    </div>
                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                        <RocketOutlined style={{ marginRight: 4 }} />Training
                                    </Text>
                                </div>
                            </Space>
                        </Col>
                    </Row>
                </div>
            </div>

            <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px' }}>
                {/* Toolbar */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
                    <Input
                        placeholder="Search projects..."
                        prefix={<SearchOutlined />}
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        style={{ maxWidth: 400, flex: 1, borderRadius: 10 }}
                        size="large"
                    />

                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', flex: 1 }}>
                        {tagFilter && (
                            <Tag closable onClose={() => setTagFilter(null)} color="blue">
                                Tag: {tagFilter}
                            </Tag>
                        )}
                        {allTags.slice(0, 8).map(tag => (
                            <Tag
                                key={tag}
                                color={tagFilter === tag ? 'blue' : undefined}
                                style={{ cursor: 'pointer', borderRadius: 6 }}
                                onClick={() => setTagFilter(tagFilter === tag ? null : tag)}
                            >
                                {tag}
                            </Tag>
                        ))}
                    </div>

                    <Segmented
                        value={view}
                        onChange={handleViewChange}
                        options={[
                            { value: 'grid', label: <span><AppstoreOutlined /> Grid</span> },
                            { value: 'list', label: <span><UnorderedListOutlined /> List</span> },
                        ]}
                        size="large"
                    />

                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setDialogOpen(true)}
                        size="large"
                        style={{
                            borderRadius: 10,
                            fontWeight: 600,
                            background: `linear-gradient(135deg, ${token.colorPrimary}, #06b6d4)`,
                            border: 'none',
                            boxShadow: '0 2px 8px rgba(8, 145, 178, 0.3)',
                        }}
                    >
                        New Project
                    </Button>
                </div>

                {/* Project Grid */}
                {loading && projects.length === 0 ? (
                    <Row gutter={[24, 24]}>
                        {[1, 2, 3].map(i => (
                            <Col xs={24} sm={12} md={8} key={i}>
                                <Card><Skeleton active paragraph={{ rows: 3 }} /></Card>
                            </Col>
                        ))}
                    </Row>
                ) : filtered.length === 0 ? (
                    <Empty
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                        description={
                            <Space direction="vertical" size={4}>
                                <Text strong>
                                    {projects.length === 0 ? 'No projects yet' : 'No matching projects'}
                                </Text>
                                <Text type="secondary" style={{ fontSize: 13 }}>
                                    {projects.length === 0
                                        ? 'Create your first project to start training GNN models'
                                        : 'Try adjusting your search or tag filters'}
                                </Text>
                            </Space>
                        }
                    >
                        {projects.length === 0 && (
                            <Button type="primary" icon={<PlusOutlined />} onClick={() => setDialogOpen(true)}>
                                Create First Project
                            </Button>
                        )}
                    </Empty>
                ) : view === 'list' ? (
                    <ProjectListTable
                        projects={filtered}
                        onOpen={(p) => router.push(getStepPath(p))}
                        onEdit={(e, p) => handleEditOpen(e, p)}
                        onDelete={(e, p) => handleDelete(e, p.project_id)}
                    />
                ) : (
                    <Row gutter={[24, 24]}>
                        {filtered.map((project) => {
                            const statusColor = project.status === 'completed' ? token.colorSuccess
                                : project.status === 'training' ? token.colorWarning
                                : token.colorPrimary;

                            return (
                                <Col xs={24} sm={12} md={8} key={project.project_id}>
                                    <Card
                                        hoverable
                                        className="card-hover-lift"
                                        onClick={() => router.push(getStepPath(project))}
                                        styles={{ body: { padding: 20 } }}
                                        style={{
                                            borderTop: `3px solid ${statusColor}`,
                                            overflow: 'hidden',
                                        }}
                                        extra={
                                            <Space size={4}>
                                                <Button
                                                    type="text"
                                                    size="small"
                                                    icon={<EditOutlined />}
                                                    onClick={(e) => handleEditOpen(e, project)}
                                                />
                                                <Button
                                                    type="text"
                                                    danger
                                                    size="small"
                                                    icon={<DeleteOutlined />}
                                                    onClick={(e) => handleDelete(e, project.project_id)}
                                                />
                                            </Space>
                                        }
                                        title={<Text strong ellipsis style={{ maxWidth: 200 }}>{project.name}</Text>}
                                    >
                                        {/* Tags */}
                                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 12, minHeight: 24 }}>
                                            {project.tags?.slice(0, 3).map(tag => (
                                                <Tag key={tag} style={{ borderRadius: 6 }}>{tag}</Tag>
                                            ))}
                                        </div>

                                        {/* Status & Date */}
                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                                            <Tag color={STATUS_TAG_COLOR[project.status] || 'default'}>
                                                {project.status.replace('_', ' ').toUpperCase()}
                                            </Tag>
                                            <Text type="secondary" style={{ fontSize: 12 }}>
                                                {timeAgo(project.updated_at || project.created_at)}
                                            </Text>
                                        </div>

                                        {/* Step progress — 6-step v2 mini-rail */}
                                        {(() => {
                                            const reached = reachedFromLegacy(project.current_step, project.status);
                                            return (
                                                <div>
                                                    <div style={{ display: 'flex', gap: 3, marginBottom: 6 }}>
                                                        {STEP_LABELS.map((label, i) => {
                                                            const isActive = i < reached;
                                                            return (
                                                                <Tooltip key={label} title={label}>
                                                                    <div style={{ flex: 1 }}>
                                                                        <div style={{
                                                                            height: 4,
                                                                            borderRadius: 2,
                                                                            background: isActive
                                                                                ? `linear-gradient(90deg, ${token.colorPrimary}, ${token.colorInfo})`
                                                                                : token.colorFillSecondary,
                                                                            transition: 'background 0.3s ease',
                                                                        }} />
                                                                    </div>
                                                                </Tooltip>
                                                            );
                                                        })}
                                                    </div>
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                        <Text type="secondary" style={{ fontSize: 10, fontWeight: 600, color: token.colorPrimary }}>
                                                            {STEP_LABELS[Math.max(0, reached - 1)]}
                                                        </Text>
                                                        <Text type="secondary" style={{ fontSize: 10, fontFamily: 'monospace' }}>
                                                            {reached}/{STEP_COUNT}
                                                        </Text>
                                                    </div>
                                                </div>
                                            );
                                        })()}
                                    </Card>
                                </Col>
                            );
                        })}
                    </Row>
                )}
            </div>

            {/* Create Project Modal */}
            <Modal
                title={
                    <Space>
                        <ThunderboltOutlined style={{ color: token.colorPrimary }} />
                        <span>Create New Project</span>
                    </Space>
                }
                open={dialogOpen}
                onCancel={() => setDialogOpen(false)}
                onOk={handleCreate}
                okText={creating ? 'Creating...' : 'Create'}
                okButtonProps={{ disabled: !newName.trim() || creating, loading: creating }}
            >
                <Space direction="vertical" size="middle" style={{ width: '100%', marginTop: 16 }}>
                    <div>
                        <Text type="secondary" style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, display: 'block' }}>
                            Project Name
                        </Text>
                        <Input
                            placeholder="e.g., SRAM Cell Classification"
                            value={newName}
                            onChange={e => setNewName(e.target.value)}
                            autoFocus
                            size="large"
                        />
                    </div>
                    <div>
                        <Text type="secondary" style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, display: 'block' }}>
                            Tags
                        </Text>
                        <Input
                            placeholder="Add tags (press Enter)"
                            value={tagInput}
                            onChange={e => setTagInput(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleTagAdd(); } }}
                        />
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 8 }}>
                            {newTags.map(tag => (
                                <Tag key={tag} closable onClose={() => setNewTags(newTags.filter(t => t !== tag))} color="blue">
                                    {tag}
                                </Tag>
                            ))}
                        </div>
                    </div>
                </Space>
            </Modal>

            {/* Edit Project Modal */}
            <Modal
                title="Edit Project"
                open={!!editProject}
                onCancel={() => setEditProject(null)}
                onOk={handleSaveEdit}
                okText={saving ? 'Saving...' : 'Save'}
                okButtonProps={{ disabled: !editName.trim() || saving, loading: saving }}
            >
                <Space direction="vertical" size="middle" style={{ width: '100%', marginTop: 16 }}>
                    <div>
                        <Text type="secondary" style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, display: 'block' }}>
                            Project Name
                        </Text>
                        <Input
                            placeholder="Project Name"
                            value={editName}
                            onChange={e => setEditName(e.target.value)}
                            autoFocus
                        />
                    </div>
                    <div>
                        <Text type="secondary" style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, display: 'block' }}>
                            Tags
                        </Text>
                        <Input
                            placeholder="Add tags (press Enter)"
                            value={editTagInput}
                            onChange={e => setEditTagInput(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleEditTagAdd(); } }}
                        />
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 8 }}>
                            {editTags.map(tag => (
                                <Tag key={tag} closable onClose={() => setEditTags(editTags.filter(t => t !== tag))} color="blue">
                                    {tag}
                                </Tag>
                            ))}
                        </div>
                    </div>
                </Space>
            </Modal>
            </PageTransition>
        </div>
    );
}
