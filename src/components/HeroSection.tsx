import React from 'react';
import { Activity, ArrowUpRight, Layers, Sparkles } from 'lucide-react';
import heroHelixImg from '../assets/images/hero_recombination_helix_1795626289030.jpg';
import { GenomeScanPreview, type ScanFocusMode } from './GenomeScanPreview';
import { WORKBENCH_URL } from './Navbar';

interface HeroTabMeta {
  id: ScanFocusMode;
  label: string;
  leftTitle: string;
  rightCopy: string;
  linkLabel: string;
  linkHref: string;
}

const HERO_TABS: HeroTabMeta[] = [
  {
    id: 'flank_kl',
    label: 'Flanking-KL scan',
    leftTitle: 'Permutation-calibrated divergence',
    rightCopy:
      'Multiscale Flanking-KL divergence is calibrated against 48 column-shuffled permutations per alignment, suppressing false positives across clonal and hyper-mutated controls.',
    linkLabel: 'How to read the scan',
    linkHref: '#walkthrough',
  },
  {
    id: 'mosaic',
    label: 'Parental mosaic tracts',
    leftTitle: 'Co-descendant exclusion',
    rightCopy:
      'Stage 6 merges redundant calls within 75 bp, clusters recombinant carrier clades, and excludes co-descendants to report true external flanking parents directly in native .rdp format.',
    linkLabel: 'Follow pipeline I/O',
    linkHref: '#pipeline',
  },
  {
    id: 'transformer',
    label: 'Window Transformer',
    leftTitle: 'S-invariant neural scoring',
    rightCopy:
      'A 4-layer Window Transformer trained on 40,000 SantaSim viral alignments evaluates circular informative-site windows independently of sequence count.',
    linkLabel: 'Inspect benchmarks',
    linkHref: '#benchmarks',
  },
];

const RESEARCH_INDEX = [
  {
    num: '01',
    title: '3D Molecular Helix',
    summary: 'Interactive Three.js recombination caliper & strand split',
    href: '#explorer-3d',
  },
  {
    num: '02',
    title: 'Read the Scan',
    summary: 'Interpreting z ≥ 6.0 peaks, TF logits, and parent ties',
    href: '#walkthrough',
  },
  {
    num: '03',
    title: 'Benchmarks',
    summary: '0.88 s Sarbecovirus CPU, 0/30 null FP, and scaling ledger',
    href: '#benchmarks',
  },
  {
    num: '04',
    title: 'Pipeline & Deploy',
    summary: 'Hover glossary on Stages 1–6 + Web, macOS & Rust CLI matrix',
    href: '#pipeline',
  },
];

interface HeroSectionProps {
  activeTab: ScanFocusMode;
  onChangeTab: (mode: ScanFocusMode) => void;
  selectedEventId: number;
  onSelectEvent: (eventId: number) => void;
  isDark: boolean;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  activeTab,
  onChangeTab,
  selectedEventId,
  onSelectEvent,
  isDark,
}) => {
  const activeMeta = HERO_TABS.find((t) => t.id === activeTab) ?? HERO_TABS[0];

  return (
    <section
      id="overview"
      aria-labelledby="hero-heading"
      className="scroll-mt-16 relative overflow-hidden bg-[#F6F6F3] dark:bg-[#121316] transition-colors duration-200"
    >
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 pt-12 pb-14 sm:pt-16 sm:pb-18">
        {/* Centered Monumental Editorial Lead Spotlight */}
        <div className="mx-auto max-w-[840px] text-center">
          <p className="hero-stagger hero-delay-1 font-mono text-[11.5px] font-medium tracking-[0.14em] uppercase text-[#1B9E77] dark:text-[#2DD4A4]">
            GT-Transformer · recombx hybrid engine
          </p>

          <h1
            id="hero-heading"
            className="hero-stagger hero-delay-2 mt-5 text-[34px] sm:text-[48px] lg:text-[56px] font-medium tracking-[-0.035em] leading-[1.22] text-[#141416] dark:text-[#F3F4F6] [text-wrap:balance]"
          >
            Fast, calibrated{' '}
            <span className="inline-flex items-center gap-2.5 align-baseline bg-[#141416] dark:bg-[#1F232B] dark:border dark:border-[#323640] text-white rounded-full px-5 sm:px-6 py-0.5 my-1 mx-0.5 font-normal leading-[1.12]">
              <span>recombination</span>
              <svg
                width="26"
                height="20"
                viewBox="0 0 26 20"
                fill="none"
                aria-hidden="true"
                className="w-5 h-4 sm:w-6 sm:h-5 shrink-0"
              >
                <path
                  d="M2 5.5 C 7 5.5, 9.5 14.5, 14.5 14.5 C 19.5 14.5, 21.5 5.5, 24 5.5"
                  stroke="#FFFFFF"
                  strokeWidth="1.75"
                  strokeLinecap="round"
                />
                <path
                  d="M2 14.5 C 7 14.5, 9.5 5.5, 14.5 5.5 C 19.5 5.5, 21.5 14.5, 24 14.5"
                  stroke="#34D399"
                  strokeWidth="1.75"
                  strokeLinecap="round"
                />
                <circle cx="10.7" cy="10" r="2" fill="#141416" stroke="#34D399" strokeWidth="1.4" />
              </svg>
            </span>{' '}
            inference for viral genomes.
          </h1>

          <p className="hero-stagger hero-delay-3 mt-5 mx-auto max-w-[58ch] text-[15.5px] sm:text-[16.5px] leading-[1.65] text-[#5A5A62] dark:text-[#A8ADB8]">
            Uniting a sequence-count-independent Window Transformer with permutation-calibrated Flanking-KL divergence to localize breakpoints, identify recombinant carriers, and attribute parental lineages in seconds.
          </p>

          {/* Segmented Scan Mode Rail */}
          <div className="hero-stagger hero-delay-3 mt-7 flex justify-center">
            <div
              role="tablist"
              aria-label="Inspect recombination inference layers"
              className="inline-flex flex-wrap items-center justify-center gap-1 p-1.5 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl sm:rounded-full shadow-[0_8px_24px_-12px_rgba(20,20,22,0.08)]"
            >
              {HERO_TABS.map((tab) => {
                const isActive = tab.id === activeTab;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    onClick={() => onChangeTab(tab.id)}
                    className={`px-3 sm:px-4 py-1.5 text-[12px] sm:text-[13px] font-medium rounded-full transition-colors duration-200 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6] ${
                      isActive
                        ? 'bg-[#EEF1FE] dark:bg-[#232B45] text-[#4F6BF6] dark:text-[#7C93FF]'
                        : 'text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6]'
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Sculptural Scientific Centerpiece Stage: Sweeping 3D Helix + Interactive Instrument + Flanking Editorial Notes */}
        <div className="hero-stagger hero-delay-4 relative mt-8 sm:mt-10">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -inset-x-10 -top-10 -bottom-10 flex items-center justify-center overflow-hidden"
          >
            <img
              src={heroHelixImg}
              alt=""
              referrerPolicy="no-referrer"
              className={`w-full max-w-[1180px] h-auto object-contain [mask-image:radial-gradient(ellipse_78%_76%_at_50%_50%,black_45%,transparent_90%)] select-none transition-opacity duration-300 ${
                isDark
                  ? 'opacity-35 invert hue-rotate-180 mix-blend-screen brightness-95 contrast-115'
                  : 'opacity-95 mix-blend-multiply brightness-[1.05] contrast-[1.06]'
              }`}
            />
          </div>

          <div className="relative z-10 grid grid-cols-1 xl:grid-cols-12 gap-6 xl:gap-8 items-center py-4 sm:py-6">
            {/* Left Editorial Annotation */}
            <div className="hidden xl:flex xl:col-span-3 flex-col items-start pr-2">
              <div
                key={`left-${activeTab}`}
                className="panel-settle bg-[#F6F6F3]/88 dark:bg-[#1A1C21]/92 backdrop-blur-xs rounded-xl p-4 border border-[#E6E5DF]/90 dark:border-[#2B2E36]"
              >
                <p className="text-[20px] font-medium tracking-[-0.02em] leading-snug text-[#141416] dark:text-[#F3F4F6]">
                  {activeMeta.leftTitle}
                </p>
                <div className="mt-4 w-10 h-10 rounded-full bg-[#141416] dark:bg-[#2DD4A4] text-white dark:text-[#0E1013] flex items-center justify-center">
                  {activeTab === 'flank_kl' && <Activity className="w-4 h-4" aria-hidden="true" />}
                  {activeTab === 'mosaic' && <Layers className="w-4 h-4" aria-hidden="true" />}
                  {activeTab === 'transformer' && <Sparkles className="w-4 h-4" aria-hidden="true" />}
                </div>
              </div>
            </div>

            {/* Central Interactive Genome Scan Frame */}
            <div className="xl:col-span-6 w-full max-w-[680px] mx-auto">
              <GenomeScanPreview
                focusMode={activeTab}
                selectedEventId={selectedEventId}
                onSelectEvent={onSelectEvent}
                isDark={isDark}
              />
            </div>

            {/* Right Editorial Explanation + Deep Jump Link */}
            <div className="hidden xl:flex xl:col-span-3 flex-col items-start pl-2">
              <div
                key={`right-${activeTab}`}
                className="panel-settle bg-[#F6F6F3]/88 dark:bg-[#1A1C21]/92 backdrop-blur-xs rounded-xl p-4 border border-[#E6E5DF]/90 dark:border-[#2B2E36]"
              >
                <p className="text-[13.5px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
                  {activeMeta.rightCopy}
                </p>
                <a
                  href={activeMeta.linkHref}
                  className="mt-4 inline-flex items-center gap-1.5 text-[13px] font-medium text-[#4F6BF6] dark:text-[#2DD4A4] hover:text-[#3A53D6] dark:hover:text-[#5EEAD4] transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6] rounded-xs"
                >
                  <span>{activeMeta.linkLabel}</span>
                  <span aria-hidden="true">▸</span>
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Centered Primary & Secondary Actions */}
        <div className="hero-stagger hero-delay-5 mt-9 flex flex-col sm:flex-row items-center justify-center gap-3.5">
          <a
            href={WORKBENCH_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex w-full sm:w-auto items-center justify-center px-7 py-3 text-[14px] font-medium text-white bg-[#4F6BF6] hover:bg-[#3F5AE3] active:bg-[#334CD0] rounded-xl transition-colors duration-150 whitespace-nowrap shrink-0 shadow-[0_10px_24px_-10px_rgba(79,107,246,0.55)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            Launch Web Workbench →
          </a>

          <a
            href="#access"
            className="inline-flex w-full sm:w-auto items-center justify-center px-6 py-3 text-[14px] font-medium text-[#141416] dark:text-[#F3F4F6] bg-white dark:bg-[#1A1C21] border border-[#DEDDD7] dark:border-[#323640] hover:border-[#141416] dark:hover:border-[#2DD4A4] rounded-xl transition-colors duration-150 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            Download Desktop App / CLI
          </a>
        </div>

        {/* Editorial Research Directory Strip */}
        <nav
          aria-label="Explore research sections"
          className="hero-stagger hero-delay-5 mt-14 pt-6 border-t border-[#DFDED8] dark:border-[#2B2E36]"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 lg:gap-0 lg:divide-x lg:divide-[#DFDED8] dark:lg:divide-[#2B2E36]">
            {RESEARCH_INDEX.map((item) => (
              <a
                key={item.num}
                href={item.href}
                className="group lg:px-6 first:lg:pl-0 last:lg:pr-0 flex items-start justify-between gap-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#4F6BF6]"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-semibold text-[#1B9E77] dark:text-[#2DD4A4] tabular-nums">
                      {item.num}
                    </span>
                    <span className="text-[14px] font-semibold text-[#141416] dark:text-[#F3F4F6] group-hover:text-[#4F6BF6] dark:group-hover:text-[#2DD4A4] transition-colors">
                      {item.title}
                    </span>
                  </div>
                  <p className="mt-1 text-[12.5px] leading-[1.5] text-[#5A5A62] dark:text-[#A8ADB8]">
                    {item.summary}
                  </p>
                </div>
                <ArrowUpRight
                  className="w-4 h-4 shrink-0 text-[#8B8B94] group-hover:text-[#4F6BF6] dark:group-hover:text-[#2DD4A4] transition-transform duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 mt-0.5"
                  aria-hidden="true"
                />
              </a>
            ))}
          </div>
        </nav>
      </div>
    </section>
  );
};
