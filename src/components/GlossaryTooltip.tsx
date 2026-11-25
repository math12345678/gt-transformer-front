import React, { useId, useState } from 'react';
import { getGlossaryEntry } from '../utils/glossary';

interface GlossaryTooltipProps {
  termKey: string;
  children: React.ReactNode;
  align?: 'center' | 'left' | 'right';
}

/**
 * Hover and keyboard-focus technical glossary tooltip.
 * Keeps editorial copy uncluttered at rest using a subtle dotted teal underline
 * and reveals a high-contrast explanation popover on hover or focus.
 */
export const GlossaryTooltip: React.FC<GlossaryTooltipProps> = ({
  termKey,
  children,
  align = 'center',
}) => {
  const entry = getGlossaryEntry(termKey);
  const [open, setOpen] = useState(false);
  const tooltipId = useId();

  if (!entry) {
    return <>{children}</>;
  }

  const alignClasses =
    align === 'left'
      ? 'left-0'
      : align === 'right'
        ? 'right-0'
        : 'left-1/2 -translate-x-1/2';

  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          setOpen(false);
        }
      }}
    >
      <span
        tabIndex={0}
        role="button"
        aria-describedby={open ? tooltipId : undefined}
        aria-expanded={open}
        data-tooltip-term={termKey}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((prev) => !prev);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            e.stopPropagation();
            setOpen((prev) => !prev);
          }
        }}
        className="cursor-help font-medium text-[#141416] dark:text-[#F3F4F6] border-b border-dotted border-[#1B9E77] dark:border-[#2DD4A4] hover:text-[#1B9E77] dark:hover:text-[#2DD4A4] transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6] rounded-[2px]"
      >
        {children}
      </span>

      {open && (
        <span
          id={tooltipId}
          role="tooltip"
          data-testid={`tooltip-${termKey}`}
          className={`panel-settle pointer-events-none absolute bottom-full mb-2.5 z-50 w-[min(290px,80vw)] ${alignClasses} rounded-xl border border-[#2B2E36] bg-[#141416] dark:bg-[#0B0C0F] px-3.5 py-3 text-left shadow-[0_18px_40px_-14px_rgba(0,0,0,0.45)]`}
        >
          <span className="block font-mono text-[10px] uppercase tracking-[0.1em] text-[#2DD4A4]">
            {entry.category}
          </span>
          <span className="mt-1 block text-[13px] font-semibold text-white leading-snug">
            {entry.term}
          </span>
          <span className="mt-1.5 block text-[12px] font-normal leading-[1.55] text-[#D1D5DB]">
            {entry.summary}
          </span>
          {entry.formula && (
            <span className="mt-2 block border-t border-white/10 pt-1.5 font-mono text-[10.5px] text-[#7C93FF]">
              {entry.formula}
            </span>
          )}
        </span>
      )}
    </span>
  );
};
