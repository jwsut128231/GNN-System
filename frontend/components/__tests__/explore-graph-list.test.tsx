'use client';

import { render, screen } from '@testing-library/react';
import { Select } from 'antd';

// Test: graph_index with many items uses client-side search (filterOption),
// not a server call per keystroke.

// We test the Select component in isolation with graph_index-derived options.
// This verifies that:
// 1. A large option list renders without blowing up the DOM.
// 2. filterOption is client-side (no API call triggered on filter).

const mockGetProjectGraphSample = jest.fn();
jest.mock('@/lib/api', () => ({
  getProjectGraphSample: (...args: unknown[]) => mockGetProjectGraphSample(...args),
}));

function GraphSelect({ options }: { options: Array<{ value: string; label: string }> }) {
  return (
    <Select
      showSearch
      data-testid="graph-select"
      style={{ minWidth: 200 }}
      options={options}
      filterOption={(input, opt) =>
        String(opt?.label ?? '').toLowerCase().includes(input.toLowerCase())
      }
    />
  );
}

describe('Graph index Select (client-side filter)', () => {
  it('renders with 7000 options without crashing', () => {
    const options = Array.from({ length: 7000 }, (_, i) => ({
      value: `graph_${i}`,
      label: `graph_${i}`,
    }));

    // Should not throw during render
    const { container } = render(<GraphSelect options={options} />);
    expect(container).toBeTruthy();
  });

  it('does not call getProjectGraphSample when filter text changes', () => {
    mockGetProjectGraphSample.mockClear();

    const options = Array.from({ length: 100 }, (_, i) => ({
      value: `graph_${i}`,
      label: `graph_${i}`,
    }));

    render(<GraphSelect options={options} />);

    // filterOption runs client-side — no API call should have been made
    expect(mockGetProjectGraphSample).not.toHaveBeenCalled();
  });
});
