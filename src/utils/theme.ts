export type ThemePreference = 'system' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'gt-theme-preference';

/**
 * Resolves a stored user preference ('system' | 'light' | 'dark') against the
 * operating system's prefers-color-scheme setting.
 */
export function resolveTheme(
  preference: ThemePreference,
  systemPrefersDark: boolean
): ResolvedTheme {
  if (preference === 'dark') return 'dark';
  if (preference === 'light') return 'light';
  return systemPrefersDark ? 'dark' : 'light';
}

/**
 * Parses an arbitrary string from storage into a valid ThemePreference,
 * falling back to 'system' when unset or invalid.
 */
export function parseThemePreference(raw: string | null | undefined): ThemePreference {
  if (raw === 'light' || raw === 'dark' || raw === 'system') {
    return raw;
  }
  return 'system';
}

/**
 * Toggles directly to the opposite of the currently active resolved theme
 * so a single click always produces an immediate, predictable change.
 */
export function getNextManualTheme(resolvedTheme: ResolvedTheme): 'light' | 'dark' {
  return resolvedTheme === 'dark' ? 'light' : 'dark';
}
