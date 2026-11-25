import { useCallback, useEffect, useState } from 'react';
import {
  getNextManualTheme,
  parseThemePreference,
  resolveTheme,
  THEME_STORAGE_KEY,
  type ResolvedTheme,
  type ThemePreference,
} from '../utils/theme';

function readSystemPrefersDark(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function readStoredPreference(): ThemePreference {
  if (typeof window === 'undefined') return 'system';
  try {
    return parseThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return 'system';
  }
}

export function useTheme() {
  const [preference, setPreference] = useState<ThemePreference>(readStoredPreference);
  const [systemPrefersDark, setSystemPrefersDark] = useState<boolean>(readSystemPrefersDark);

  const resolvedTheme: ResolvedTheme = resolveTheme(preference, systemPrefersDark);
  const isDark = resolvedTheme === 'dark';

  // Keep OS preference synced in real time when the user changes system appearance
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (event: MediaQueryListEvent) => {
      setSystemPrefersDark(event.matches);
    };
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Apply .dark class and color-scheme to <html>
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', isDark);
    root.style.colorScheme = resolvedTheme;
  }, [isDark, resolvedTheme]);

  const setThemePreference = useCallback((next: ThemePreference) => {
    setPreference(next);
    try {
      if (next === 'system') {
        window.localStorage.removeItem(THEME_STORAGE_KEY);
      } else {
        window.localStorage.setItem(THEME_STORAGE_KEY, next);
      }
    } catch {
      // Storage may be restricted in sandboxed frames
    }
  }, []);

  const toggleTheme = useCallback(() => {
    const next = getNextManualTheme(resolvedTheme);
    setThemePreference(next);
  }, [resolvedTheme, setThemePreference]);

  const resetToSystem = useCallback(() => {
    setThemePreference('system');
  }, [setThemePreference]);

  return {
    preference,
    resolvedTheme,
    isDark,
    toggleTheme,
    setThemePreference,
    resetToSystem,
  };
}
