import React, { useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { useRevealOnScroll } from '../hooks/useRevealOnScroll';
import type { ScanFocusMode } from './GenomeScanPreview';

interface WalkthroughStep {
  id: number;
  code: string;
  title: string;
  headline: string;
  explanation: string;
  quantitativeNote: string;
  jumpLabel: string;
  jumpEventId: number;
  jumpMode: ScanFocusMode;
}

const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    id: 1,
    code: '01 · CALIBRATION',
    title: 'The z = 6.0 permutation threshold',
    headline: 'Separate true mosaic switches from mutation-rate noise',
    explanation:
      'For every alignment, Ground Truth generates 48 column-shuffled permutations that preserve site diversity while destroying spatial linkage. Only loci whose Flanking-KL divergence rises at least 6 standard deviations above that alignment-specific null (z ≥ 6.0) are called as candidate breakpoints.',
    quantitativeNote:
      'Null controls stay flat below z = 6.0 (0 / 30 FP); true breakpoints peak at z = 9.3–15.2 (64-seq stress) and z = 55.8–234.3 (8-seq validation).',
    jumpLabel: 'Highlight peak #7 (8,200 bp) in hero',
    jumpEventId: 7,
    jumpMode: 'flank_kl',
  },
  {
    id: 2,
    code: '02 · NEURAL CROSS-CHECK',
    title: 'Window Transformer boundary posterior',
    headline: 'Pinpoint sub-window coordinates independently of clade depth',
    explanation:
      'In parallel with the statistical scan, a 4-layer pre-norm Window Transformer (570,881 parameters, trained on 40,000 SantaSim alignments) scores circular informative-site windows. Because its input representation is S-invariant, a rare 1/64 singleton recombinant produces the same sharp neural logit as an ancestral clade switch.',
    quantitativeNote:
      'TF logits reach 2.81–4.35 at stress breakpoints (p_TF = 0.797–0.932 on 01_main_8seq_1500bp), anchoring 1.0 bp median error.',
    jumpLabel: 'Switch hero to Window Transformer view',
    jumpEventId: 6,
    jumpMode: 'transformer',
  },
  {
    id: 3,
    code: '03 · MOSAIC ATTRIBUTION',
    title: 'Carrier clades & co-descendant exclusion',
    headline: 'Identify external flanking parents without sibling contamination',
    explanation:
      'Colored bars beneath the curve track which parental clade each recombinant lineage matches across the genome. Before calling left and right parents, Stage 6 clusters all descendants sharing the event (for example Rec11..Rec13) and excludes those co-descendants from the donor pool so siblings never masquerade as parents.',
    quantitativeNote:
      'Proximal raw peaks within 75 bp merge into a single Level-1 event with 95% coordinate spans (e.g., 8,189..8,212 bp).',
    jumpLabel: 'Switch hero to Parental Mosaic Tracts',
    jumpEventId: 7,
    jumpMode: 'mosaic',
  },
  {
    id: 4,
    code: '04 · GUARDRAILS',
    title: 'Sibling ties & physical resolution limits',
    headline: 'Report ambiguity honestly rather than over-calling',
    explanation:
      'When two external candidate parents fall within 0.005 sequence similarity on a flanking tract, Ground Truth explicitly reports a tie ("P1 or P5") instead of breaking it arbitrarily. Equally, the statistical scan documents its physical blind spots: ±30 bp at alignment termini and very short tracts under 100 bp.',
    quantitativeNote:
      'Short 380 bp gene conversions (events #4 and #5, 4,800–5,180 bp) are still cleanly resolved above z = 9.3.',
    jumpLabel: 'Inspect 380 bp conversion (#4) in hero',
    jumpEventId: 4,
    jumpMode: 'flank_kl',
  },
];

const STRESS_REGIMES = [
  {
    label: 'Ancestral clade switch',
    coord: '1,150 bp',
    carriers: 'Rec01..04',
    detail: 'Shared deep crossover (CladeA → CladeB)',
    eventId: 1,
    mode: 'flank_kl' as ScanFocusMode,
  },
  {
    label: 'CRF cassette',
    coord: '2,600 & 3,350 bp',
    carriers: 'Rec05..07',
    detail: '750 bp internal cassette (CladeC ↔ CladeD)',
    eventId: 2,
    mode: 'mosaic' as ScanFocusMode,
  },
  {
    label: 'Short gene conversion',
    coord: '4,800 & 5,180 bp',
    carriers: 'Rec08..09',
    detail: '380 bp narrow donor tract (CladeB ↔ CladeE)',
    eventId: 4,
    mode: 'flank_kl' as ScanFocusMode,
  },
  {
    label: 'Rare 1 / 64 singleton',
    coord: '6,750 bp',
    carriers: 'Rec10',
    detail: 'Single sequence carrying CladeE → CladeF switch',
    eventId: 6,
    mode: 'transformer' as ScanFocusMode,
  },
  {
    label: 'Complex triple mosaic',
    coord: '8,200 · 9,150 · 10,100 bp',
    carriers: 'Rec11..13',
    detail: 'Four-parent chain (A → D → F → B)',
    eventId: 7,
    mode: 'mosaic' as ScanFocusMode,
  },
  {
    label: 'Drifted IUPAC / gap switch',
    coord: '11,150 bp',
    carriers: 'Rec14..15',
    detail: 'Divergent tract with ambiguous bases & gaps',
    eventId: 10,
    mode: 'flank_kl' as ScanFocusMode,
  },
];

const LocusAnatomyDiagram: React.FC<{ activeStepId: number; isDark: boolean }> = ({
  activeStepId,
  isDark,
}) => {
  const highlightThreshold = activeStepId === 1;
  const highlightTf = activeStepId === 2;
  const highlightMosaic = activeStepId === 3;
  const highlightGuardrail = activeStepId === 4;

  const ink = isDark ? '#F3F4F6' : '#141416';
  const muted = isDark ? '#A8ADB8' : '#5A5A62';
  const teal = isDark ? '#2DD4A4' : '#1B9E77';
  const cobalt = isDark ? '#7C93FF' : '#4F6BF6';
  const violet = isDark ? '#A78BFA' : '#7570B3';
  const surface = isDark ? '#1A1C21' : '#FFFFFF';
  const baseline = isDark ? '#323640' : '#D9D8D2';
  const leftParentFill = isDark ? '#3B4252' : '#141416';

  return (
    <svg
      viewBox="0 0 580 278"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Zoomed diagram of the 8,200 bp breakpoint illustrating flanking windows, z equals 6.0 null threshold, Window Transformer peak, and parental clade switch from Clade A to Clade D"
      className="w-full h-auto block select-none"
    >
      {/* Flanking window bands [b-w, b) and [b, b+w) */}
      <rect
        x="48"
        y="30"
        width="242"
        height="136"
        rx="6"
        fill={isDark ? '#94A3B8' : '#141416'}
        fillOpacity={highlightMosaic ? '0.11' : '0.05'}
      />
      <rect
        x="290"
        y="30"
        width="242"
        height="136"
        rx="6"
        fill={violet}
        fillOpacity={highlightMosaic ? '0.14' : '0.06'}
      />

      <text
        x="169"
        y="48"
        textAnchor="middle"
        fill={muted}
        fontSize="10"
        fontFamily="JetBrains Mono, monospace"
      >
        Left flank [b − w, b) · CladeA
      </text>
      <text
        x="411"
        y="48"
        textAnchor="middle"
        fill={violet}
        fontSize="10"
        fontFamily="JetBrains Mono, monospace"
      >
        Right flank [b, b + w) · CladeD
      </text>

      {/* 95% Confidence Span band */}
      <rect
        x="272"
        y="30"
        width="36"
        height="136"
        fill={teal}
        fillOpacity={highlightGuardrail ? '0.22' : '0.12'}
      />
      <text
        x="290"
        y="22"
        textAnchor="middle"
        fill={ink}
        fontSize="10"
        fontWeight="600"
        fontFamily="JetBrains Mono, monospace"
      >
        b = 8,200 bp (95% CI: 8,189..8,212)
      </text>

      {/* Baseline */}
      <line x1="48" y1="166" x2="532" y2="166" stroke={baseline} strokeWidth="1" />

      {/* Dashed z = 6.0 permutation threshold */}
      <line
        x1="48"
        y1="126"
        x2="532"
        y2="126"
        stroke={highlightThreshold ? ink : muted}
        strokeWidth={highlightThreshold ? '1.6' : '1'}
        strokeDasharray="5 4"
      />
      <text
        x="528"
        y="119"
        textAnchor="end"
        fill={highlightThreshold ? ink : muted}
        fontSize="9.5"
        fontWeight={highlightThreshold ? '600' : '400'}
        fontFamily="JetBrains Mono, monospace"
        paintOrder="stroke"
        stroke={surface}
        strokeWidth="3.5"
        strokeLinejoin="round"
      >
        z = 6.0 (48× shuffled null)
      </text>

      {/* Shuffled null envelope band */}
      <rect
        x="48"
        y="148"
        width="484"
        height="18"
        fill={teal}
        fillOpacity={highlightThreshold ? '0.18' : '0.09'}
      />
      <text
        x="56"
        y="160"
        fill={muted}
        fontSize="8.5"
        fontFamily="JetBrains Mono, monospace"
      >
        Clonal / mutation-noise floor (z &lt; 2.5)
      </text>

      {/* Window Transformer curve */}
      <path
        d="M 48 162 C 130 162, 224 160, 256 142 C 272 128, 281 78, 290 68 C 299 78, 308 128, 324 142 C 356 160, 450 162, 532 162"
        stroke={cobalt}
        strokeWidth={highlightTf ? '2.5' : '1.3'}
        strokeDasharray={highlightTf ? undefined : '3 3'}
        strokeOpacity={highlightTf ? '1' : '0.5'}
        fill="none"
      />

      {/* Flanking-KL divergence curve */}
      <path
        d="M 48 156 C 124 155, 206 152, 248 130 C 268 118, 279 64, 290 54 C 301 64, 312 118, 332 130 C 374 152, 456 155, 532 156"
        stroke={teal}
        strokeWidth={highlightTf ? '1.35' : '2.2'}
        strokeOpacity={highlightTf ? '0.45' : '1'}
        fill="none"
      />

      {/* Peak center line & marker */}
      <line x1="290" y1="30" x2="290" y2="246" stroke={ink} strokeWidth="1.2" strokeDasharray="2 2" opacity="0.7" />
      <circle
        cx="290"
        cy={highlightTf ? 68 : 54}
        r="5"
        fill={ink}
        stroke={highlightTf ? cobalt : teal}
        strokeWidth="2"
      />

      <text
        x="306"
        y="70"
        fill={highlightTf ? cobalt : teal}
        fontSize="10"
        fontWeight="600"
        fontFamily="JetBrains Mono, monospace"
      >
        {highlightTf ? 'TF logit = 4.35' : 'Flanking-KL z = 15.2'}
      </text>

      {/* Recombinant carrier track below */}
      <text
        x="48"
        y="196"
        fill={muted}
        fontSize="9"
        fontFamily="JetBrains Mono, monospace"
        letterSpacing="0.06em"
      >
        CARRIER CLADE · Rec11..Rec13 (CO-DESCENDANTS EXCLUDED FROM DONOR POOL)
      </text>

      <rect
        x="48"
        y="206"
        width="240"
        height="22"
        rx="4"
        fill={leftParentFill}
        fillOpacity={highlightMosaic ? '0.96' : '0.84'}
      />
      <text
        x="168"
        y="220.5"
        textAnchor="middle"
        fill="#FFFFFF"
        fontSize="10"
        fontFamily="JetBrains Mono, monospace"
      >
        External Left Parent: CladeA
      </text>

      <rect
        x="292"
        y="206"
        width="240"
        height="22"
        rx="4"
        fill={violet}
        fillOpacity={highlightMosaic ? '0.96' : '0.84'}
      />
      <text
        x="412"
        y="220.5"
        textAnchor="middle"
        fill="#FFFFFF"
        fontSize="10"
        fontFamily="JetBrains Mono, monospace"
      >
        External Right Parent: CladeD
      </text>

      {/* Coordinate axis labels */}
      <text x="48" y="262" fill={muted} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
        7,800 bp
      </text>
      <text
        x="290"
        y="262"
        textAnchor="middle"
        fill={ink}
        fontSize="9.5"
        fontWeight="600"
        fontFamily="JetBrains Mono, monospace"
      >
        8,200 bp
      </text>
      <text
        x="532"
        y="262"
        textAnchor="end"
        fill={muted}
        fontSize="9.5"
        fontFamily="JetBrains Mono, monospace"
      >
        8,600 bp
      </text>
    </svg>
  );
};

interface ScanWalkthroughSectionProps {
  onJumpToHero: (eventId: number, mode: ScanFocusMode) => void;
  isDark: boolean;
}

export const ScanWalkthroughSection: React.FC<ScanWalkthroughSectionProps> = ({
  onJumpToHero,
  isDark,
}) => {
  const [activeStepId, setActiveStepId] = useState<number>(1);
  const { ref, isVisible } = useRevealOnScroll<HTMLElement>();
  const activeStep = WALKTHROUGH_STEPS.find((s) => s.id === activeStepId) ?? WALKTHROUGH_STEPS[0];

  return (
    <section
      id="walkthrough"
      ref={ref}
      aria-labelledby="walkthrough-heading"
      className={`scroll-mt-16 bg-white dark:bg-[#16181D] border-t border-[#E6E5DF] dark:border-[#2B2E36] transition-colors duration-200 reveal-section ${
        isVisible ? 'reveal-visible' : 'reveal-hidden'
      }`}
    >
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 py-20 sm:py-24">
        {/* Editorial Header */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 pb-12 border-b border-[#E6E5DF] dark:border-[#2B2E36] items-end">
          <div className="lg:col-span-8">
            <p className="font-mono text-[11px] tracking-[0.14em] uppercase text-[#1B9E77] dark:text-[#2DD4A4] mb-3">
              Guided Interpretation · Reading the Profile
            </p>
            <h2
              id="walkthrough-heading"
              className="text-[28px] sm:text-[38px] font-medium tracking-[-0.03em] leading-[1.12] text-[#141416] dark:text-[#F3F4F6] [text-wrap:balance]"
            >
              How to read a Ground Truth recombination scan.
            </h2>
          </div>
          <div className="lg:col-span-4 lg:text-right">
            <p className="text-[14px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
              Every breakpoint call combines statistical null calibration, neural localization, and clade-aware parent verification.
            </p>
          </div>
        </div>

        {/* Lead Interactive Walkthrough Split */}
        <div className="mt-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <ol className="lg:col-span-5 divide-y divide-[#ECEBE6] dark:divide-[#2B2E36] border-y border-[#ECEBE6] dark:border-[#2B2E36]">
            {WALKTHROUGH_STEPS.map((step) => {
              const isSelected = step.id === activeStepId;
              return (
                <li key={step.id}>
                  <button
                    type="button"
                    onClick={() => setActiveStepId(step.id)}
                    aria-pressed={isSelected}
                    className={`w-full text-left py-4 px-4 transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6] ${
                      isSelected
                        ? 'bg-[#F6F6F3] dark:bg-[#1E2128]'
                        : 'bg-white dark:bg-[#16181D] hover:bg-[#FAF9F7] dark:hover:bg-[#1A1C21]'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`font-mono text-[11px] font-semibold tracking-[0.06em] ${
                          isSelected
                            ? 'text-[#4F6BF6] dark:text-[#7C93FF]'
                            : 'text-[#1B9E77] dark:text-[#2DD4A4]'
                        }`}
                      >
                        {step.code}
                      </span>
                      <span className="font-mono text-[11px] text-[#5A5A62] dark:text-[#A8ADB8]">
                        {isSelected ? 'Active' : 'Inspect'}
                      </span>
                    </div>
                    <h3 className="mt-1.5 text-[16.5px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                      {step.title}
                    </h3>
                    <p className="mt-1 text-[13.5px] leading-[1.55] text-[#5A5A62] dark:text-[#A8ADB8]">
                      {step.headline}
                    </p>
                  </button>
                </li>
              );
            })}
          </ol>

          <div
            key={activeStep.id}
            className="panel-settle lg:col-span-7 bg-[#FAF9F7] dark:bg-[#121316] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-7"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[#E6E5DF] dark:border-[#2B2E36]">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.1em] text-[#4F6BF6] dark:text-[#7C93FF]">
                  {activeStep.code}
                </p>
                <h3 className="mt-1 text-[19px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                  {activeStep.title}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => onJumpToHero(activeStep.jumpEventId, activeStep.jumpMode)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-[#141416] dark:text-[#F3F4F6] bg-white dark:bg-[#1A1C21] border border-[#DEDDD7] dark:border-[#323640] hover:border-[#141416] dark:hover:border-[#2DD4A4] rounded-lg transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
              >
                <span>{activeStep.jumpLabel}</span>
                <ArrowUpRight className="w-3.5 h-3.5 text-[#4F6BF6] dark:text-[#2DD4A4]" aria-hidden="true" />
              </button>
            </div>

            <div className="py-5 bg-white dark:bg-[#1A1C21] rounded-xl border border-[#ECEBE6] dark:border-[#2B2E36] my-5 px-3">
              <LocusAnatomyDiagram activeStepId={activeStep.id} isDark={isDark} />
            </div>

            <p className="text-[14.5px] leading-[1.68] text-[#3F3F46] dark:text-[#D1D5DB]">
              {activeStep.explanation}
            </p>
            <p className="mt-4 pt-4 border-t border-[#E6E5DF] dark:border-[#2B2E36] font-mono text-[12px] leading-[1.6] text-[#141416] dark:text-[#2DD4A4]">
              {activeStep.quantitativeNote}
            </p>
          </div>
        </div>

        {/* Mixed-Size Editorial Research Stories Row */}
        <div className="mt-12 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          <article className="lg:col-span-7 bg-[#F6F6F3] dark:bg-[#121316] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-8 flex flex-col justify-between">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#1B9E77] dark:text-[#2DD4A4]">
                Inside the Hero Scan · 64 seq × 12,000 bp
              </p>
              <h3 className="mt-2 text-[21px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                Six recombination regimes tested in a single alignment.
              </h3>
              <p className="mt-2.5 text-[14px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
                Select any regime below to jump directly to its peak and parental mosaic track in the interactive profile above.
              </p>

              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4">
                {STRESS_REGIMES.map((regime) => (
                  <button
                    key={regime.label}
                    type="button"
                    onClick={() => onJumpToHero(regime.eventId, regime.mode)}
                    className="group text-left pt-3 border-t border-[#DFDED8] dark:border-[#2B2E36] hover:border-[#141416] dark:hover:border-[#2DD4A4] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[13.5px] font-semibold text-[#141416] dark:text-[#F3F4F6] group-hover:text-[#4F6BF6] dark:group-hover:text-[#2DD4A4] transition-colors">
                        {regime.label}
                      </span>
                      <span className="font-mono text-[11px] text-[#1B9E77] dark:text-[#2DD4A4] tabular-nums shrink-0">
                        {regime.coord}
                      </span>
                    </div>
                    <p className="mt-1 text-[12.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
                      <span className="font-mono text-[#141416] dark:text-[#E5E7EB]">
                        {regime.carriers}
                      </span>{' '}
                      · {regime.detail}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </article>

          <article className="lg:col-span-5 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-8 flex flex-col justify-between">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#5A5A62] dark:text-[#A8ADB8]">
                Resolution Boundaries
              </p>
              <h3 className="mt-2 text-[21px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                Where the detector stays conservative.
              </h3>
              <p className="mt-2.5 text-[14px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
                Publication-grade recombination inference requires transparent physical limits rather than silent extrapolation.
              </p>

              <dl className="mt-6 divide-y divide-[#ECEBE6] dark:divide-[#2B2E36] border-t border-[#ECEBE6] dark:border-[#2B2E36] text-[13px]">
                <div className="py-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <dt className="font-mono text-[#141416] dark:text-[#2DD4A4] whitespace-nowrap">
                    ±30 bp alignment termini
                  </dt>
                  <dd className="text-[#5A5A62] dark:text-[#A8ADB8] whitespace-nowrap">
                    Edge dead zone
                  </dd>
                </div>
                <div className="py-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <dt className="font-mono text-[#141416] dark:text-[#2DD4A4] whitespace-nowrap">
                    &lt;100 bp or &lt;10 PIS
                  </dt>
                  <dd className="text-[#5A5A62] dark:text-[#A8ADB8] whitespace-nowrap">
                    Attenuated power
                  </dd>
                </div>
                <div className="py-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <dt className="font-mono text-[#141416] dark:text-[#2DD4A4] whitespace-nowrap">
                    Δ similarity ≤ 0.005
                  </dt>
                  <dd className="text-[#5A5A62] dark:text-[#A8ADB8] whitespace-nowrap">
                    Reported as &ldquo;P1 or P5&rdquo;
                  </dd>
                </div>
                <div className="py-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <dt className="font-mono text-[#141416] dark:text-[#2DD4A4] whitespace-nowrap">
                    5..500 seq · ≤50 kb
                  </dt>
                  <dd className="text-[#5A5A62] dark:text-[#A8ADB8] whitespace-nowrap">
                    Upfront schema check
                  </dd>
                </div>
              </dl>
            </div>

            <p className="mt-4 text-[12.5px] leading-[1.55] text-[#5A5A62] dark:text-[#A8ADB8]">
              Malformed alignments (unequal lengths, &lt;5 sequences, duplicate headers, all-gap files) are rejected before queuing.
            </p>
          </article>
        </div>
      </div>
    </section>
  );
};
