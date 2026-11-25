import React from 'react';

interface BrandLogoProps {
  compact?: boolean;
}

/**
 * Custom monoline inline SVG logo:
 * The letters "GT" woven seamlessly into a horizontal DNA double-helix crossover,
 * paired with the "GROUND TRUTH" editorial wordmark. Adapts cleanly to light & dark charcoal modes.
 */
export const BrandLogo: React.FC<BrandLogoProps> = ({ compact = false }) => {
  return (
    <span className="inline-flex items-center gap-3 whitespace-nowrap shrink-0 select-none">
      <svg
        width="44"
        height="24"
        viewBox="0 0 44 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        className="shrink-0"
      >
        {/* Subtle base-pair rungs inside the helix loops */}
        <line x1="7.5" y1="7.2" x2="7.5" y2="16.8" className="stroke-[#E5E7EB] dark:stroke-[#2B2E36]" strokeWidth="1.2" />
        <line x1="12.5" y1="6.4" x2="12.5" y2="17.6" className="stroke-[#E5E7EB] dark:stroke-[#2B2E36]" strokeWidth="1.2" />
        <line x1="31.5" y1="6.4" x2="31.5" y2="17.6" className="stroke-[#E5E7EB] dark:stroke-[#2B2E36]" strokeWidth="1.2" />
        <line x1="36.5" y1="7.2" x2="36.5" y2="16.8" className="stroke-[#E5E7EB] dark:stroke-[#2B2E36]" strokeWidth="1.2" />

        {/* Upper-to-lower crossover strand forming the G-loop and crossing over */}
        <path
          d="M 16.5 6.8 C 12.5 3.6, 4.5 4.6, 4.5 12 C 4.5 19.4, 13.5 20.2, 18.2 15.2 L 24.8 8.4 C 28.4 4.8, 35.8 4.4, 41.5 6.8"
          className="stroke-[#111111] dark:stroke-[#F3F4F6]"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Horizontal crossbar completing the monoline "G" inside the left loop */}
        <path
          d="M 10.8 12.2 H 16.2 V 16.2"
          className="stroke-[#111111] dark:stroke-[#F3F4F6]"
          strokeWidth="1.65"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Lower-to-upper complementary recombination strand (deep teal -> luminous teal in dark mode) */}
        <path
          d="M 2.5 17.2 C 7.5 19.8, 14.2 18.8, 18.2 15.2 L 24.8 8.4 C 28.6 4.8, 36.2 4.8, 41.5 6.8"
          className="stroke-[#1B9E77] dark:stroke-[#2DD4A4]"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.24"
        />
        <path
          d="M 18.2 8.4 C 21.0 11.4, 22.8 13.4, 25.4 16.0 C 29.2 19.6, 36.2 19.6, 41.5 16.8"
          className="stroke-[#1B9E77] dark:stroke-[#2DD4A4]"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Monoline "T" top bar & stem integrated into the right helix phase */}
        <path
          d="M 28.2 6.8 H 39.2 M 33.7 6.8 V 18.6"
          className="stroke-[#1B9E77] dark:stroke-[#2DD4A4]"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Crisp crossover node marking the recombination breakpoint */}
        <circle
          cx="21.6"
          cy="11.8"
          r="2.1"
          className="fill-white dark:fill-[#121316] stroke-[#1B9E77] dark:stroke-[#2DD4A4]"
          strokeWidth="1.5"
        />
      </svg>

      {!compact && (
        <span className="text-[13.5px] font-semibold tracking-[0.14em] text-[#111111] dark:text-[#F3F4F6] uppercase">
          GROUND TRUTH
        </span>
      )}
    </span>
  );
};
