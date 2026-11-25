import React, { useState } from 'react';
import { Menu, Moon, Sun, X } from 'lucide-react';
import type { ThemePreference } from '../utils/theme';
import { BrandLogo } from './BrandLogo';

export const WORKBENCH_URL = 'https://math12345678.github.io/gt-transformer-web/';

const NAV_LINKS = [
  { label: 'Overview', href: '#overview' },
  { label: '3D Helix', href: '#explorer-3d' },
  { label: 'Read Scan', href: '#walkthrough' },
  { label: 'Benchmarks', href: '#benchmarks' },
  { label: 'Pipeline', href: '#pipeline' },
  { label: 'Deploy', href: '#access' },
];

interface NavbarProps {
  isDark: boolean;
  preference: ThemePreference;
  onToggleTheme: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ isDark, preference, onToggleTheme }) => {
  const [mobileOpen, setMobileOpen] = useState(false);

  const themeTitle =
    preference === 'system'
      ? `Theme follows system (${isDark ? 'Dark' : 'Light'}) — click to switch`
      : `Switch to ${isDark ? 'light' : 'dark'} mode`;

  return (
    <header className="sticky top-0 z-40 bg-[#F6F6F3]/92 dark:bg-[#121316]/92 backdrop-blur-md border-b border-[#E6E5DF] dark:border-[#2B2E36] transition-colors duration-200">
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 h-16 flex items-center justify-between gap-6">
        {/* Zone 1: Brand lockup */}
        <a
          href="#overview"
          aria-label="Ground Truth — Return to top"
          className="inline-flex items-center focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#4F6BF6] rounded-sm shrink-0"
        >
          <BrandLogo />
        </a>

        {/* Zone 2: Concise single-line navigation links */}
        <nav
          aria-label="Primary navigation"
          className="hidden lg:flex items-center gap-6 text-[13.5px] font-medium text-[#5A5A62] dark:text-[#A8ADB8]"
        >
          {NAV_LINKS.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="whitespace-nowrap shrink-0 py-1 hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors duration-150 relative after:content-[''] after:absolute after:left-0 after:right-0 after:bottom-0 after:h-px after:bg-[#141416] dark:after:bg-[#2DD4A4] after:scale-x-0 hover:after:scale-x-100 after:transition-transform after:duration-150 after:origin-left focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#4F6BF6] rounded-xs"
            >
              {item.label}
            </a>
          ))}
        </nav>

        {/* Zone 3: Theme toggle + primary CTA + mobile menu toggle */}
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            data-testid="theme-toggle"
            onClick={onToggleTheme}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-pressed={isDark}
            title={themeTitle}
            className="inline-flex items-center justify-center w-9 h-9 rounded-full border border-[#E6E5DF] dark:border-[#2B2E36] bg-white dark:bg-[#1A1C21] text-[#141416] dark:text-[#2DD4A4] hover:border-[#141416] dark:hover:border-[#2DD4A4] transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            {isDark ? (
              <Sun className="w-4 h-4" aria-hidden="true" />
            ) : (
              <Moon className="w-4 h-4" aria-hidden="true" />
            )}
          </button>

          <a
            href={WORKBENCH_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:inline-flex items-center justify-center px-4 py-2 text-[13px] font-medium text-white dark:text-[#0E1013] bg-[#141416] dark:bg-[#2DD4A4] hover:bg-[#2B2B30] dark:hover:bg-[#34E0AE] rounded-full transition-colors duration-150 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            Launch Web Workbench →
          </a>

          <button
            type="button"
            onClick={() => setMobileOpen((prev) => !prev)}
            aria-label={mobileOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={mobileOpen}
            className="lg:hidden inline-flex items-center justify-center w-9 h-9 text-[#141416] dark:text-[#F3F4F6] bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-full hover:bg-[#F6F6F3] dark:hover:bg-[#23262E] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            {mobileOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {mobileOpen && (
        <div className="lg:hidden border-t border-[#E6E5DF] dark:border-[#2B2E36] bg-[#F6F6F3] dark:bg-[#121316] px-5 py-4 space-y-1">
          {NAV_LINKS.map((item) => (
            <a
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className="block py-2.5 text-[14.5px] font-medium text-[#141416] dark:text-[#F3F4F6] hover:text-[#4F6BF6] dark:hover:text-[#2DD4A4] transition-colors"
            >
              {item.label}
            </a>
          ))}
          <div className="pt-3 border-t border-[#E6E5DF] dark:border-[#2B2E36]">
            <a
              href={WORKBENCH_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex w-full items-center justify-center px-4 py-2.5 text-[13.5px] font-medium text-white dark:text-[#0E1013] bg-[#141416] dark:bg-[#2DD4A4] hover:bg-[#2B2B30] dark:hover:bg-[#34E0AE] rounded-full transition-colors"
            >
              Launch Web Workbench →
            </a>
          </div>
        </div>
      )}
    </header>
  );
};
