import React, { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { useRevealOnScroll } from '../hooks/useRevealOnScroll';
import { GlossaryTooltip } from './GlossaryTooltip';

interface PipelineContract {
  inputTitle: string;
  inputSpec: string;
  opTitle: string;
  opSpec: string;
  outputTitle: string;
  outputSpec: string;
}

interface PipelineStage {
  id: number;
  stepNumber: string;
  stages: string;
  title: string;
  renderDescription: () => React.ReactNode;
  specLabel: string;
  contract: PipelineContract;
  invariants: string[];
}

interface MicroStage {
  num: string;
  label: string;
  parentStepId: number;
}

const MICRO_STAGES: MicroStage[] = [
  { num: '1', label: 'FASTA Validate', parentStepId: 1 },
  { num: '2', label: 'Informative Encode', parentStepId: 1 },
  { num: '3', label: 'Window Transformer', parentStepId: 2 },
  { num: '4', label: 'Flanking-KL Scan', parentStepId: 2 },
  { num: '5', label: '48× Permutation Null', parentStepId: 2 },
  { num: '6', label: 'Attribution & .rdp', parentStepId: 3 },
];

const PIPELINE_STEPS: PipelineStage[] = [
  {
    id: 1,
    stepNumber: '01',
    stages: 'Stages 1–2',
    title: 'Informative-Site Encoding',
    renderDescription: () => (
      <>
        Filters aligned FASTA (S × L) to{' '}
        <GlossaryTooltip termKey="parsimony-informative" align="left">
          parsimony-informative sites
        </GlossaryTooltip>{' '}
        and builds an{' '}
        <GlossaryTooltip termKey="s-invariant" align="left">
          S-invariant
        </GlossaryTooltip>{' '}
        <GlossaryTooltip termKey="circular-window" align="left">
          circular window
        </GlossaryTooltip>{' '}
        representation.
      </>
    ),
    specLabel: 'Input Normalization & Window Tensorization',
    contract: {
      inputTitle: 'Aligned FASTA (S × L)',
      inputSpec: '5–500 sequences · ≤50,000 bp · ≤20 MB · IUPAC & gap safe',
      opTitle: 'Parsimony Filter + Circular Window',
      opSpec: 'W = 314 informative sites · 4-gamete & pairwise discordance',
      outputTitle: 'S-Invariant Window Tensor',
      outputSpec: 'Fixed-width feature tensor + 1-based genomic index map',
    },
    invariants: [
      'Rejects unequal lengths, duplicate headers, <5 sequences, and all-gap files upfront',
      'Retains parsimony-informative columns while preserving exact 1-based genomic coordinates',
      'Circular 4-gamete window encoding decouples neural runtime from sequence count S',
    ],
  },
  {
    id: 2,
    stepNumber: '02',
    stages: 'Stages 3–5',
    title: 'Dual Neural + Statistical Scan',
    renderDescription: () => (
      <>
        Runs a 4-layer{' '}
        <GlossaryTooltip termKey="window-transformer" align="left">
          Window Transformer
        </GlossaryTooltip>{' '}
        (trained on 40,000 SantaSim alignments) in parallel with a multiscale{' '}
        <GlossaryTooltip termKey="flanking-kl" align="left">
          Flanking-KL divergence
        </GlossaryTooltip>{' '}
        scan calibrated against 48{' '}
        <GlossaryTooltip termKey="column-shuffled" align="left">
          column-shuffled permutations
        </GlossaryTooltip>
        .
      </>
    ),
    specLabel: 'Hybrid Neural Logit + Permutation Null Fusion',
    contract: {
      inputTitle: 'Window Tensor + PIS Matrix',
      inputSpec: 'ckpt_N40000 (570,881 params) + informative-site array',
      opTitle: 'Parallel Transformer || Flanking-KL',
      opSpec: '4-layer pre-norm logits + [b−w, b) vs [b, b+w) KL over 48 shuffles',
      outputTitle: 'Calibrated Breakpoint Profile',
      outputSpec: 'Null-calibrated z-scores (z ≥ 6.0), p_TF logits & 95% spans',
    },
    invariants: [
      '4-layer pre-norm Window Transformer (570,881 parameters, ckpt_N40000)',
      'Symmetric flanking window [b−w, b) vs [b, b+w) soft-parent KL divergence',
      'Empirical z-score null computed from 48 column-shuffled permutations (0 / 30 null FP)',
    ],
  },
  {
    id: 3,
    stepNumber: '03',
    stages: 'Stage 6',
    title: 'Hierarchical Attribution',
    renderDescription: () => (
      <>
        Performs{' '}
        <GlossaryTooltip termKey="level-1-consolidation" align="left">
          Level-1 consolidation
        </GlossaryTooltip>{' '}
        (≤75 bp), infers recombinant carrier clades, excludes{' '}
        <GlossaryTooltip termKey="co-descendants" align="left">
          co-descendants
        </GlossaryTooltip>{' '}
        to identify external parents, and exports directly to{' '}
        <GlossaryTooltip termKey="native-rdp5" align="left">
          native RDP5 (.rdp)
        </GlossaryTooltip>{' '}
        projects.
      </>
    ),
    specLabel: 'Connected-Component Consolidation & Export',
    contract: {
      inputTitle: 'Thresholded Peaks (z ≥ 6.0)',
      inputSpec: 'Candidate loci + left/right flanking similarity matrices',
      opTitle: 'Connected-Component Attribution',
      opSpec: 'Merge ≤75 bp · cluster carriers · exclude co-descendants · 0.005 tie check',
      outputTitle: 'Level-1 Events & Research Exports',
      outputSpec: 'Native .rdp, GFF3, BED, FASTA mosaic tracts, JSON & TSV',
    },
    invariants: [
      'Consolidates proximal peaks within 75 bp into single Level-1 events (>20× fewer duplicates)',
      'Separates recombinant carrier clades from co-descendant siblings before calling external parents',
      'Reports sibling ties within 0.005 similarity explicitly ("P1 or P5") and writes native .rdp projects',
    ],
  },
];

const StageSchematic: React.FC<{ activeId: number; isDark: boolean }> = ({ activeId, isDark }) => {
  const ink = isDark ? '#F3F4F6' : '#141416';
  const muted = isDark ? '#A8ADB8' : '#5A5A62';
  const teal = isDark ? '#2DD4A4' : '#1B9E77';
  const cobalt = isDark ? '#7C93FF' : '#4F6BF6';
  const violet = isDark ? '#A78BFA' : '#7570B3';
  const panelBg = isDark ? '#16181D' : '#FAF9F7';
  const cardBg = isDark ? '#1E2128' : '#FFFFFF';
  const border = isDark ? '#323640' : '#E6E5DF';
  const seqTrack = isDark ? '#3A3E49' : '#DCDCD5';

  if (activeId === 1) {
    return (
      <svg
        viewBox="0 0 540 232"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Schematic of Stage 1 and 2 informative-site filtering and S-invariant circular window encoding"
        className="w-full h-auto"
      >
        <rect x="20" y="28" width="160" height="150" rx="10" fill={panelBg} stroke={border} />
        <text x="36" y="52" fill={ink} fontSize="11" fontWeight="600" fontFamily="Plus Jakarta Sans, sans-serif">
          Aligned FASTA (S × L)
        </text>
        {[72, 90, 108, 126, 144, 162].map((y, row) => (
          <g key={y}>
            <line x1="36" y1={y} x2="164" y2={y} stroke={seqTrack} strokeWidth="3" strokeLinecap="round" />
            <rect x="64" y={y - 3} width="5" height="6" rx="1" fill={row < 3 ? teal : cobalt} />
            <rect x="104" y={y - 3} width="5" height="6" rx="1" fill={row % 2 === 0 ? ink : violet} />
            <rect x="138" y={y - 3} width="5" height="6" rx="1" fill={row >= 3 ? teal : cobalt} />
          </g>
        ))}
        <text x="100" y="200" textAnchor="middle" fill={muted} fontSize="10" fontFamily="JetBrains Mono, monospace">
          Parsimony-informative sites
        </text>

        <path
          d="M 194 103 H 242 M 235 96 L 243 103 L 235 110"
          stroke={ink}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <text x="218" y="90" textAnchor="middle" fill={teal} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
          filter
        </text>

        <rect x="256" y="28" width="264" height="150" rx="10" fill={cardBg} stroke={border} />
        <text x="276" y="52" fill={ink} fontSize="11" fontWeight="600" fontFamily="Plus Jakarta Sans, sans-serif">
          S-Invariant Circular Window (W = 314)
        </text>
        <circle cx="332" cy="112" r="38" stroke={teal} strokeWidth="1.5" strokeDasharray="4 3" />
        <circle cx="332" cy="112" r="22" stroke={cobalt} strokeWidth="1.5" />
        <circle cx="332" cy="112" r="8" fill={ink} />

        <rect x="394" y="78" width="104" height="14" rx="4" fill={isDark ? '#232B45' : '#EEF1FE'} />
        <text x="404" y="88" fill={cobalt} fontSize="9" fontFamily="JetBrains Mono, monospace">
          4-gamete density
        </text>
        <rect x="394" y="102" width="104" height="14" rx="4" fill={isDark ? '#142F28' : '#E9F7F2'} />
        <text x="404" y="112" fill={teal} fontSize="9" fontFamily="JetBrains Mono, monospace">
          pairwise discord
        </text>
        <rect x="394" y="126" width="104" height="14" rx="4" fill={isDark ? '#262A34' : '#F4F3EF'} />
        <text x="404" y="136" fill={ink} fontSize="9" fontFamily="JetBrains Mono, monospace">
          coord index map
        </text>

        <text x="388" y="200" textAnchor="middle" fill={muted} fontSize="10" fontFamily="JetBrains Mono, monospace">
          Depth-independent tensor input
        </text>
      </svg>
    );
  }

  if (activeId === 2) {
    return (
      <svg
        viewBox="0 0 540 232"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Schematic of Stages 3 to 5 parallel Window Transformer and 48-permutation Flanking-KL scan"
        className="w-full h-auto"
      >
        <rect x="20" y="22" width="210" height="76" rx="10" fill={cardBg} stroke={cobalt} strokeOpacity="0.55" />
        <text x="36" y="46" fill={cobalt} fontSize="11" fontWeight="600" fontFamily="Plus Jakarta Sans, sans-serif">
          4-Layer Window Transformer
        </text>
        <text x="36" y="66" fill={muted} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
          40,000 SantaSim alignments
        </text>
        <text x="36" y="82" fill={muted} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
          Boundary logit score (p_TF)
        </text>

        <rect x="20" y="118" width="210" height="76" rx="10" fill={cardBg} stroke={teal} strokeOpacity="0.6" />
        <text x="36" y="142" fill={teal} fontSize="11" fontWeight="600" fontFamily="Plus Jakarta Sans, sans-serif">
          Multiscale Flanking-KL Scan
        </text>
        <text x="36" y="162" fill={muted} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
          48 column-shuffled permutations
        </text>
        <text x="36" y="178" fill={muted} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
          Null-calibrated z-score (z ≥ 6.0)
        </text>

        <path d="M 230 60 H 268 V 108 H 294" stroke={ink} strokeWidth="1.4" />
        <path d="M 230 156 H 268 V 108 H 294" stroke={ink} strokeWidth="1.4" />
        <path d="M 288 102 L 296 108 L 288 114" stroke={ink} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />

        <rect x="304" y="28" width="216" height="158" rx="10" fill={panelBg} stroke={border} />
        <text x="320" y="52" fill={ink} fontSize="11" fontWeight="600" fontFamily="Plus Jakarta Sans, sans-serif">
          Calibrated Hybrid Profile
        </text>
        <line x1="320" y1="122" x2="504" y2="122" stroke={muted} strokeWidth="1" strokeDasharray="4 3" />
        <text x="504" y="116" textAnchor="end" fill={muted} fontSize="8.5" fontFamily="JetBrains Mono, monospace">
          z = 6.0
        </text>
        <path
          d="M 320 156 L 348 154 L 368 76 L 386 152 L 424 150 L 444 86 L 462 153 L 504 155"
          stroke={teal}
          strokeWidth="2"
          strokeLinejoin="round"
        />
        <path
          d="M 320 158 L 350 157 L 368 84 L 384 156 L 426 154 L 444 94 L 460 156 L 504 158"
          stroke={cobalt}
          strokeWidth="1.35"
          strokeDasharray="3 2"
          strokeLinejoin="round"
        />
        <circle cx="368" cy="76" r="4" fill={ink} stroke={teal} strokeWidth="1.5" />
        <circle cx="444" cy="86" r="4" fill={ink} stroke={teal} strokeWidth="1.5" />
        <text x="412" y="208" textAnchor="middle" fill={muted} fontSize="10" fontFamily="JetBrains Mono, monospace">
          0 / 30 null false positives
        </text>
      </svg>
    );
  }

  return (
    <svg
      viewBox="0 0 540 232"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Schematic of Stage 6 peak consolidation, carrier clade attribution, and native RDP5 project export"
      className="w-full h-auto"
    >
      <rect x="20" y="28" width="318" height="154" rx="10" fill={panelBg} stroke={border} />
      <text x="36" y="52" fill={ink} fontSize="11" fontWeight="600" fontFamily="Plus Jakarta Sans, sans-serif">
        Level-1 Event Consolidation (≤75 bp)
      </text>

      <rect x="36" y="74" width="94" height="18" rx="4" fill={isDark ? '#475569' : '#141416'} />
      <rect x="134" y="74" width="92" height="18" rx="4" fill={teal} />
      <rect x="230" y="74" width="90" height="18" rx="4" fill={cobalt} />

      <rect x="36" y="102" width="94" height="18" rx="4" fill={isDark ? '#475569' : '#141416'} fillOpacity="0.72" />
      <rect x="134" y="102" width="92" height="18" rx="4" fill={teal} fillOpacity="0.82" />
      <rect x="230" y="102" width="90" height="18" rx="4" fill={cobalt} fillOpacity="0.82" />

      <line x1="132" y1="64" x2="132" y2="134" stroke={ink} strokeWidth="1.2" strokeDasharray="3 2" />
      <line x1="228" y1="64" x2="228" y2="134" stroke={ink} strokeWidth="1.2" strokeDasharray="3 2" />

      <text x="83" y="156" textAnchor="middle" fill={ink} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
        Clade A (L)
      </text>
      <text x="180" y="156" textAnchor="middle" fill={teal} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
        Clade B (Donor)
      </text>
      <text x="275" y="156" textAnchor="middle" fill={cobalt} fontSize="9.5" fontFamily="JetBrains Mono, monospace">
        Clade D (R)
      </text>

      <path d="M 350 105 H 386 M 379 98 L 387 105 L 379 112" stroke={ink} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />

      <rect x="398" y="36" width="122" height="138" rx="10" fill={isDark ? '#0B0C0F' : '#141416'} stroke={isDark ? '#2B2E36' : 'none'} />
      <text x="459" y="78" textAnchor="middle" fill="#34D399" fontSize="20" fontWeight="600" fontFamily="JetBrains Mono, monospace">
        .rdp
      </text>
      <text x="459" y="102" textAnchor="middle" fill="#FFFFFF" fontSize="10.5" fontFamily="Plus Jakarta Sans, sans-serif">
        Native RDP5 Project
      </text>
      <text x="459" y="130" textAnchor="middle" fill="#A1A1AA" fontSize="9" fontFamily="JetBrains Mono, monospace">
        GFF3 · BED · FASTA
      </text>
      <text x="459" y="146" textAnchor="middle" fill="#A1A1AA" fontSize="9" fontFamily="JetBrains Mono, monospace">
        JSON · TSV
      </text>

      <text x="179" y="204" textAnchor="middle" fill={muted} fontSize="10" fontFamily="JetBrains Mono, monospace">
        Co-descendants excluded · &gt;20× fewer duplicate calls
      </text>
    </svg>
  );
};

interface PipelineSectionProps {
  isDark: boolean;
}

export const PipelineSection: React.FC<PipelineSectionProps> = ({ isDark }) => {
  const [activeStageId, setActiveStageId] = useState<number>(2);
  const { ref, isVisible } = useRevealOnScroll<HTMLElement>();
  const activeStage = PIPELINE_STEPS.find((s) => s.id === activeStageId) ?? PIPELINE_STEPS[0];

  return (
    <section
      id="pipeline"
      ref={ref}
      aria-labelledby="pipeline-heading"
      className={`scroll-mt-16 bg-white dark:bg-[#16181D] border-t border-[#E6E5DF] dark:border-[#2B2E36] transition-colors duration-200 reveal-section ${
        isVisible ? 'reveal-visible' : 'reveal-hidden'
      }`}
    >
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 py-20 sm:py-24">
        {/* Centered Editorial Header */}
        <div className="mx-auto max-w-[760px] text-center mb-12">
          <p className="font-mono text-[11px] tracking-[0.14em] uppercase text-[#1B9E77] dark:text-[#2DD4A4] mb-3">
            How It Works · End-to-End Data Flow
          </p>
          <h2
            id="pipeline-heading"
            className="text-[30px] sm:text-[40px] font-medium tracking-[-0.03em] leading-[1.12] text-[#141416] dark:text-[#F3F4F6] [text-wrap:balance]"
          >
            Three-stage{' '}
            <span className="inline-flex items-center bg-[#141416] dark:bg-[#1F232B] dark:border dark:border-[#323640] text-white rounded-full px-4 py-0.5 font-normal">
              hybrid
            </span>{' '}
            inference architecture.
          </h2>
          <p className="mt-4 text-[15.5px] leading-[1.65] text-[#5A5A62] dark:text-[#A8ADB8]">
            Follow exact inputs, intermediate representations, and outputs across all six internal pipeline stages—hover any underlined term for a technical definition.
          </p>
        </div>

        {/* Clickable 6-Stage End-to-End Pipeline Ribbon */}
        <div
          aria-label="Six pipeline stages"
          className="mb-10 bg-[#FAF9F7] dark:bg-[#121316] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-xl p-2.5 overflow-x-auto"
        >
          <div className="flex items-center justify-between gap-1.5 min-w-[660px]">
            {MICRO_STAGES.map((stage, idx) => {
              const isHighlighted = stage.parentStepId === activeStageId;
              return (
                <React.Fragment key={stage.num}>
                  <button
                    type="button"
                    onClick={() => setActiveStageId(stage.parentStepId)}
                    className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-[12px] font-medium transition-colors duration-150 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6] ${
                      isHighlighted
                        ? 'bg-[#141416] dark:bg-[#2DD4A4] text-white dark:text-[#0E1013] font-semibold'
                        : 'bg-white dark:bg-[#1A1C21] text-[#5A5A62] dark:text-[#A8ADB8] border border-[#ECEBE6] dark:border-[#2B2E36] hover:border-[#141416] dark:hover:border-[#2DD4A4] hover:text-[#141416] dark:hover:text-[#F3F4F6]'
                    }`}
                  >
                    <span
                      className={`font-mono text-[10.5px] ${
                        isHighlighted
                          ? 'text-[#34D399] dark:text-[#0E1013]'
                          : 'text-[#1B9E77] dark:text-[#2DD4A4]'
                      }`}
                    >
                      S{stage.num}
                    </span>
                    <span>{stage.label}</span>
                  </button>
                  {idx < MICRO_STAGES.length - 1 && (
                    <ArrowRight className="w-3.5 h-3.5 text-[#A1A1AA] dark:text-[#5A5A62] shrink-0" aria-hidden="true" />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Asymmetric Interactive Architectural Split */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: 3 Architectural Steps with Hover Technical Tooltips */}
          <ol className="lg:col-span-5 space-y-2.5">
            {PIPELINE_STEPS.map((step) => {
              const isSelected = step.id === activeStageId;
              return (
                <li
                  key={step.id}
                  onClick={() => setActiveStageId(step.id)}
                  className={`cursor-pointer p-5 rounded-2xl border transition-all duration-150 ${
                    isSelected
                      ? 'bg-[#F6F6F3] dark:bg-[#1E2128] border-[#D9D8D0] dark:border-[#3A3E49] shadow-[0_12px_32px_-18px_rgba(20,20,22,0.12)]'
                      : 'bg-white dark:bg-[#16181D] border-[#ECEBE6] dark:border-[#2B2E36] hover:bg-[#FAF9F7] dark:hover:bg-[#1A1C21] hover:border-[#DEDDD7] dark:hover:border-[#323640]'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => setActiveStageId(step.id)}
                    aria-pressed={isSelected}
                    className="w-full text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6] rounded-sm"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span
                        className={`font-mono text-[12px] font-semibold tabular-nums ${
                          isSelected
                            ? 'text-[#4F6BF6] dark:text-[#7C93FF]'
                            : 'text-[#1B9E77] dark:text-[#2DD4A4]'
                        }`}
                      >
                        Step {step.stepNumber}
                      </span>
                      <span className="font-mono text-[11px] text-[#5A5A62] dark:text-[#A8ADB8]">
                        {step.stages}
                      </span>
                    </div>

                    <h3 className="mt-2.5 text-[18px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                      {step.title}
                    </h3>
                  </button>

                  <p className="mt-2 text-[14px] leading-[1.65] text-[#5A5A62] dark:text-[#A8ADB8]">
                    {step.renderDescription()}
                  </p>
                </li>
              );
            })}
          </ol>

          {/* Right Column: Interactive Schematic + Explicit Input → Operation → Output Data Contract */}
          <div
            key={activeStage.id}
            data-testid="pipeline-stage-panel"
            className="panel-settle lg:col-span-7 bg-[#FAF9F7] dark:bg-[#121316] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-8 shadow-[0_20px_50px_-24px_rgba(20,20,22,0.08)]"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 pb-5 border-b border-[#E6E5DF] dark:border-[#2B2E36]">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.1em] text-[#4F6BF6] dark:text-[#7C93FF]">
                  Step {activeStage.stepNumber} · {activeStage.stages}
                </p>
                <h3 className="mt-1 text-[18px] font-semibold text-[#141416] dark:text-[#F3F4F6]">
                  {activeStage.specLabel}
                </h3>
              </div>
              <span className="font-mono text-[11.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
                Input → Transform → Output
              </span>
            </div>

            {/* Explicit 3-Column Stage Data Contract */}
            <div className="mt-5 grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-xl p-3.5">
                <span className="block font-mono text-[10px] uppercase tracking-[0.08em] text-[#5A5A62] dark:text-[#A8ADB8]">
                  Input
                </span>
                <p className="mt-1 text-[13px] font-semibold text-[#141416] dark:text-[#F3F4F6]">
                  {activeStage.contract.inputTitle}
                </p>
                <p className="mt-1 font-mono text-[11px] leading-[1.5] text-[#5A5A62] dark:text-[#A8ADB8]">
                  {activeStage.contract.inputSpec}
                </p>
              </div>
              <div className="bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-xl p-3.5">
                <span className="block font-mono text-[10px] uppercase tracking-[0.08em] text-[#4F6BF6] dark:text-[#7C93FF]">
                  Transformation
                </span>
                <p className="mt-1 text-[13px] font-semibold text-[#141416] dark:text-[#F3F4F6]">
                  {activeStage.contract.opTitle}
                </p>
                <p className="mt-1 font-mono text-[11px] leading-[1.5] text-[#5A5A62] dark:text-[#A8ADB8]">
                  {activeStage.contract.opSpec}
                </p>
              </div>
              <div className="bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-xl p-3.5">
                <span className="block font-mono text-[10px] uppercase tracking-[0.08em] text-[#1B9E77] dark:text-[#2DD4A4]">
                  Output
                </span>
                <p className="mt-1 text-[13px] font-semibold text-[#141416] dark:text-[#F3F4F6]">
                  {activeStage.contract.outputTitle}
                </p>
                <p className="mt-1 font-mono text-[11px] leading-[1.5] text-[#5A5A62] dark:text-[#A8ADB8]">
                  {activeStage.contract.outputSpec}
                </p>
              </div>
            </div>

            {/* Vector Architectural Schematic */}
            <div className="my-5 p-3 bg-white dark:bg-[#1A1C21] border border-[#ECEBE6] dark:border-[#2B2E36] rounded-xl">
              <StageSchematic activeId={activeStage.id} isDark={isDark} />
            </div>

            <ul className="pt-4 border-t border-[#E6E5DF] dark:border-[#2B2E36] space-y-2.5 text-[13.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
              {activeStage.invariants.map((item) => (
                <li key={item} className="flex items-baseline gap-2.5">
                  <span className="font-mono text-[#1B9E77] dark:text-[#2DD4A4] shrink-0">—</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};
