'use client';
import * as React from 'react';
import { ConfigProvider, App, theme as antdTheme } from 'antd';
import { AntdRegistry } from '@ant-design/nextjs-registry';
import { getTheme } from '@/theme/theme';
import { ColorModeContext } from '@/contexts/ColorModeContext';

/** Syncs <html> data-theme attribute and <body> background/text color with active theme */
function BodyStyleSync({ mode }: { mode: 'light' | 'dark' }) {
  const { token } = antdTheme.useToken();

  React.useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', mode);
    // Also toggle .dark class for CSS selectors using that convention
    if (mode === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    document.body.style.backgroundColor = token.colorBgContainer;
    document.body.style.color = token.colorText;
  }, [token.colorBgContainer, token.colorText, mode]);

  return null;
}

export default function ThemeRegistry({ children }: { children: React.ReactNode }) {
  // Always initialize to 'light' so SSR and first client render agree (avoids hydration mismatch).
  // The useEffect below reads localStorage after mount and switches if needed.
  const [mode, setMode] = React.useState<'light' | 'dark'>('light');
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    const saved = localStorage.getItem('color_mode') as 'light' | 'dark' | null;
    if (saved) setMode(saved);
    setMounted(true);
  }, []);

  const colorMode = React.useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prev) => {
          const next = prev === 'light' ? 'dark' : 'light';
          localStorage.setItem('color_mode', next);
          return next;
        });
      },
      mode,
    }),
    [mode],
  );

  // Before mount, force light algorithm so ConfigProvider token values are stable between
  // SSR and first client paint. Dashboard inline styles (colorBgContainer, colorPrimary, etc.)
  // derive from these tokens — mismatching them is what triggers the visual layout glitch.
  const themeConfig = React.useMemo(
    () => mounted ? getTheme(mode) : { ...getTheme('light'), algorithm: antdTheme.defaultAlgorithm },
    [mode, mounted],
  );

  return (
    // suppressHydrationWarning: the data-theme attribute on <html> is written client-side
    // by BodyStyleSync; React would otherwise warn about the server/client attribute mismatch.
    <AntdRegistry>
      <ColorModeContext.Provider value={colorMode}>
        <ConfigProvider theme={themeConfig}>
          <App>
            <BodyStyleSync mode={mode} />
            <div suppressHydrationWarning>{children}</div>
          </App>
        </ConfigProvider>
      </ColorModeContext.Provider>
    </AntdRegistry>
  );
}
