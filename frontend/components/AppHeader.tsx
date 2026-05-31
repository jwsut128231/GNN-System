'use client';

import React, { useState } from 'react';
import {
    Button, Avatar, Breadcrumb, Steps, Divider, Tag, Typography, theme, Grid, Drawer, Space, Tooltip,
} from 'antd';
import {
    UserOutlined, LogoutOutlined,
    SunOutlined, MoonOutlined, MenuOutlined,
    LockOutlined, CheckOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { ColorModeContext } from '@/contexts/ColorModeContext';
import Image from 'next/image';

const { Text } = Typography;
const { useBreakpoint } = Grid;

// v2 pipeline: Create → Upload → Analyze → Train → Evaluate → Predict
// Register/Models is NOT in the pipeline per v2 — it's linked from Evaluate / project landing instead.
// Index 0 (Create) navigates back to /dashboard where projects are created.
export const STEPS = [
    { label: 'Create' },
    { label: 'Upload' },
    { label: 'Analyze' },
    { label: 'Train' },
    { label: 'Evaluate' },
    { label: 'Predict' },
];

export const STEP_PATHS = (projectId: string) => [
    `/dashboard`,
    `/projects/${projectId}/upload`,
    `/projects/${projectId}/explore`,
    `/projects/${projectId}/train`,
    `/projects/${projectId}/evaluate`,
    `/projects/${projectId}/predict`,
];

const STATUS_TAG_COLOR: Record<string, string> = {
    completed: 'green',
    failed: 'red',
    training: 'processing',
};

interface AppHeaderProps {
    subtitle?: string;
    projectName?: string;
    projectId?: string;
    projectStep?: number;
    projectStatus?: string;
}

export default function AppHeader({ subtitle, projectName, projectId, projectStep, projectStatus }: AppHeaderProps) {
    const router = useRouter();
    const { user, logout } = useAuth();
    const { mode, toggleColorMode } = React.useContext(ColorModeContext);
    const { token } = theme.useToken();
    const screens = useBreakpoint();
    const isMobile = !screens.md;
    const [drawerOpen, setDrawerOpen] = useState(false);

    const isProjectMode = !!(projectId && projectName);
    // Translate legacy 1..5 current_step into the 6-step v2 index space (0..5):
    // legacy 1(upload)→1, 2(analysis)→2, 3(training)→3, 4(evaluation)→4, 5(models)→4.
    // Create(index 0) maps to the dashboard; it's always reachable.
    const legacyStep = projectStep ?? 1;
    const activeIndex = isProjectMode ? Math.min(Math.max(legacyStep, 1), 4) : -1;
    const baseMaxIndex = activeIndex;
    // Predict(5) unlocks after Evaluate is reached AND status === 'completed' (a model exists).
    const maxReachableIndex = isProjectMode
        ? (projectStatus === 'completed' && baseMaxIndex >= 4 ? 5 : baseMaxIndex)
        : -1;

    const handleStepClick = (index: number) => {
        if (index > maxReachableIndex) return;
        if (index === 0) {
            router.push('/dashboard');
            return;
        }
        if (!projectId) return;
        router.push(STEP_PATHS(projectId)[index]);
    };

    // Build AntD Steps items with lock-on-future + check-on-completed semantics.
    const buildStepItems = () => STEPS.map((s, i) => {
        const locked = i > maxReachableIndex;
        const done = i < activeIndex;
        const isTraining = projectStatus === 'training' && i === 3; // new Train index
        const baseTitle = s.label;
        if (locked) {
            return {
                title: (
                    <Tooltip title="Complete previous steps to unlock">
                        <span style={{ color: token.colorTextDisabled }}>{baseTitle}</span>
                    </Tooltip>
                ),
                icon: <LockOutlined style={{ color: token.colorTextDisabled }} />,
                disabled: true as const,
            };
        }
        if (done) {
            return {
                title: baseTitle,
                icon: <CheckOutlined style={{ color: token.colorSuccess }} />,
            };
        }
        return {
            title: baseTitle,
            status: isTraining ? ('process' as const) : undefined,
        };
    });

    const logoSection = (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
            <Image src="/graphx-icon.svg" alt="GraphX.AI" width={32} height={32} style={{ borderRadius: 8 }} />
            <Button
                type="text"
                onClick={() => router.push('/dashboard')}
                style={{ fontWeight: 800, fontSize: '1rem', padding: '4px 4px' }}
            >
                <span className="gradient-text">GraphX.AI</span>
            </Button>
        </div>
    );

    const rightSection = (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
            <Button
                type="text"
                icon={mode === 'light' ? <MoonOutlined /> : <SunOutlined />}
                onClick={toggleColorMode}
                style={{
                    fontSize: 16,
                    borderRadius: 8,
                    width: 36,
                    height: 36,
                }}
                title={mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            />
            {user && (
                <>
                    <Divider type="vertical" style={{ height: 28, margin: '0 4px' }} />
                    <Avatar
                        src={user.avatar}
                        alt={user.name}
                        size={34}
                        style={{
                            border: `2px solid ${token.colorPrimary}30`,
                        }}
                        icon={<UserOutlined />}
                    />
                    <Text strong style={{ fontSize: 13 }}>{user.name}</Text>
                    <Button
                        type="text"
                        danger
                        icon={<LogoutOutlined />}
                        onClick={logout}
                        style={{ borderRadius: 8 }}
                    >
                        Logout
                    </Button>
                </>
            )}
        </div>
    );

    const middleSection = isProjectMode ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 0, overflow: 'hidden' }}>
            <Breadcrumb
                items={[
                    { title: <a onClick={(e) => { e.preventDefault(); router.push('/dashboard'); }}>Dashboard</a> },
                    {
                        title: (
                            <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'inline-block' }}>
                                {projectName}
                            </span>
                        ),
                    },
                    { title: <strong>{STEPS[activeIndex]?.label}</strong> },
                ]}
            />
            {projectStatus && (
                <Tag color={STATUS_TAG_COLOR[projectStatus] || 'default'}>
                    {projectStatus.replace('_', ' ').toUpperCase()}
                </Tag>
            )}
        </div>
    ) : subtitle ? (
        <div style={{ flex: 1 }}>
            <Text type="secondary" style={{ letterSpacing: 1, fontSize: 12, fontWeight: 500 }}>{subtitle}</Text>
        </div>
    ) : (
        <div style={{ flex: 1 }} />
    );

    if (isMobile) {
        return (
            <>
                <div className="glass-header" style={{
                    padding: '12px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    position: 'sticky',
                    top: 0,
                    zIndex: 100,
                }}>
                    {logoSection}
                    <div style={{ flex: 1 }} />
                    <Button
                        type="text"
                        icon={mode === 'light' ? <MoonOutlined /> : <SunOutlined />}
                        onClick={toggleColorMode}
                        style={{ fontSize: 16 }}
                    />
                    <Button
                        type="text"
                        icon={<MenuOutlined />}
                        onClick={() => setDrawerOpen(true)}
                    />
                </div>
                <Drawer
                    title="Menu"
                    placement="right"
                    open={drawerOpen}
                    onClose={() => setDrawerOpen(false)}
                    width={280}
                >
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                        {isProjectMode && (
                            <Steps
                                current={activeIndex}
                                direction="vertical"
                                size="small"
                                onChange={(i) => { handleStepClick(i); setDrawerOpen(false); }}
                                items={buildStepItems()}
                            />
                        )}
                        {user && (
                            <div>
                                <div style={{ fontWeight: 700 }}>{user.name}</div>
                                <div style={{ fontSize: 12, color: token.colorTextSecondary }}>{user.email}</div>
                                <Button
                                    type="text"
                                    danger
                                    icon={<LogoutOutlined />}
                                    onClick={() => { logout(); setDrawerOpen(false); }}
                                    style={{ marginTop: 8, padding: 0 }}
                                >
                                    Logout
                                </Button>
                            </div>
                        )}
                    </Space>
                </Drawer>
            </>
        );
    }

    return (
        <div className="glass-header" style={{
            padding: '10px 24px',
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            position: 'sticky',
            top: 0,
            zIndex: 100,
        }}>
            {logoSection}
            <Divider type="vertical" style={{ height: 28 }} />
            {middleSection}
            {isProjectMode && (
                <Steps
                    current={activeIndex}
                    size="small"
                    style={{ flex: 2, maxWidth: 720 }}
                    onChange={handleStepClick}
                    items={buildStepItems()}
                />
            )}
            {rightSection}
        </div>
    );
}
