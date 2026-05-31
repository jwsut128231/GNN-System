import { render, screen } from '@testing-library/react';
import { Space, Tag, Typography } from 'antd';
import { WarningOutlined } from '@ant-design/icons';

// Minimal inline components matching the Presence column renderer
// and PerGraphFeatureSchemaCard logic from explore/page.tsx.

const { Text } = Typography;

interface PresenceCellProps {
  presencePct?: number;
  lowPresence?: boolean;
}

function PresenceCell({ presencePct, lowPresence }: PresenceCellProps) {
  if (presencePct == null) return <Text type="secondary">—</Text>;
  return (
    <Space size={4}>
      <Text>{presencePct.toFixed(2)}%</Text>
      {lowPresence && (
        <Tag color="orange" icon={<WarningOutlined />} data-testid="low-presence-tag">
          low presence
        </Tag>
      )}
    </Space>
  );
}

interface PerGraphSchemaEntry {
  union: string[];
  intersection: string[];
  presence_per_column: Record<string, number>;
  low_presence_columns: string[];
}

function PerGraphBreakdown({ schema }: { schema: Record<string, PerGraphSchemaEntry> }) {
  return (
    <div data-testid="per-graph-schema">
      {Object.entries(schema).map(([typeName, entry]) => (
        <div key={typeName} data-testid={`schema-type-${typeName}`}>
          <span data-testid={`type-label-${typeName}`}>{typeName}</span>
          {entry.union.map(col => (
            <div key={col} data-testid={`col-${typeName}-${col}`}>
              <span>{col}</span>
              <span data-testid={`pct-${typeName}-${col}`}>
                {((entry.presence_per_column[col] ?? 0) * 100).toFixed(0)}% of graphs
              </span>
              {entry.low_presence_columns.includes(col) && (
                <span data-testid={`low-${typeName}-${col}`}>low presence</span>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

describe('Presence rate UI', () => {
  it('renders presence percentage without warning when low_presence_warning is false', () => {
    render(<PresenceCell presencePct={85} lowPresence={false} />);
    expect(screen.getByText('85.00%')).toBeInTheDocument();
    expect(screen.queryByTestId('low-presence-tag')).not.toBeInTheDocument();
  });

  it('renders low presence warning tag when low_presence_warning is true', () => {
    render(<PresenceCell presencePct={5} lowPresence={true} />);
    expect(screen.getByText('5.00%')).toBeInTheDocument();
    expect(screen.getByTestId('low-presence-tag')).toBeInTheDocument();
    expect(screen.getByText('low presence')).toBeInTheDocument();
  });

  it('renders dash when presencePct is undefined', () => {
    render(<PresenceCell />);
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.queryByTestId('low-presence-tag')).not.toBeInTheDocument();
  });
});

describe('Per-graph feature schema card', () => {
  const schema: Record<string, PerGraphSchemaEntry> = {
    node: {
      union: ['feature_a', 'feature_b', 'feature_c'],
      intersection: ['feature_a'],
      presence_per_column: {
        feature_a: 1.0,
        feature_b: 0.5,
        feature_c: 0.5,
      },
      low_presence_columns: ['feature_b', 'feature_c'],
    },
  };

  it('renders per-graph schema breakdown card', () => {
    render(<PerGraphBreakdown schema={schema} />);
    expect(screen.getByTestId('per-graph-schema')).toBeInTheDocument();
    expect(screen.getByTestId('schema-type-node')).toBeInTheDocument();
    expect(screen.getByTestId('type-label-node')).toHaveTextContent('node');
  });

  it('shows correct presence percentages for each column', () => {
    render(<PerGraphBreakdown schema={schema} />);
    expect(screen.getByTestId('pct-node-feature_a')).toHaveTextContent('100% of graphs');
    expect(screen.getByTestId('pct-node-feature_b')).toHaveTextContent('50% of graphs');
    expect(screen.getByTestId('pct-node-feature_c')).toHaveTextContent('50% of graphs');
  });

  it('marks low_presence_columns with warning indicator', () => {
    render(<PerGraphBreakdown schema={schema} />);
    expect(screen.getByTestId('low-node-feature_b')).toBeInTheDocument();
    expect(screen.getByTestId('low-node-feature_c')).toBeInTheDocument();
    // feature_a is in intersection (100%) — should NOT have low presence
    expect(screen.queryByTestId('low-node-feature_a')).not.toBeInTheDocument();
  });
});
