import React from 'react';
import { useRevealOnScroll } from '../hooks/useRevealOnScroll';

interface MetricItem {
  value: string;
  label: string;
  context: string;
}

const KEY_METRICS: MetricItem[] = [
  {
    value: '0.88 s',
    label: 'Sarbecovirus CPU Runtime',
    context: '19-seq, 30.6 kb Sarbecovirus scan on CPU (Rust recombx)',
  },
  {
    value: '100% Specificity',
    label: 'Permutation-Calibrated Null',
    context: '0 / 30 false positives across clonal & high-mutation controls',
  },
  {
    value: '1.0 bp Median Error',
    label: 'Sub-Window Localization',
    context: '10 / 10 breakpoints recovered on 64-seq, 12 kb stress test',
  },
  {
    value: '>20× Fewer Duplicate Calls',
    label: 'Hierarchical Consolidation',
    context: 'Level-1 consolidation vs. raw 7-method RDP5 output',
  },
];

const BENCHMARK_ROWS = [
  {
    dataset: 'Sarbecovirus empirical alignment',
    dimensions: '19 seq × 30,600 bp',
    engine: 'Rust recombx (hybrid)',
    outcome: '0.88 s wall-clock CPU',
    metric: 'Full-genome scan',
  },
  {
    dataset: 'Multi-clade mosaic stress test',
    dimensions: '64 seq × 12,000 bp',
    engine: 'GT-Transformer + Flanking-KL',
    outcome: '10 / 10 breakpoints recovered',
    metric: '1.0 bp median error',
  },
  {
    dataset: 'Clonal & high-mutation null suite',
    dimensions: '30 control alignments',
    engine: '48× column-shuffled null',
    outcome: '0 / 30 false positives',
    metric: '100.0% specificity',
  },
  {
    dataset: 'Event deduplication vs. RDP5',
    dimensions: '7-method raw output',
    engine: 'Stage 6 connected components',
    outcome: '>20× fewer redundant calls',
    metric: 'Native .rdp export',
  },
  {
    dataset: 'Validation suite · 01_main (8-seq)',
    dimensions: '8 seq × 1,500 bp',
    engine: 'Python CPU · ckpt_N40000',
    outcome: '4 / 4 within ±6 bp · 0 FP',
    metric: '1.17 s wall time',
  },
  {
    dataset: 'CPU scaling · 60-seq alignment',
    dimensions: '60 seq × 3,000 bp',
    engine: 'Python CPU · 48 permutations',
    outcome: 'Calibrated hybrid profile',
    metric: '6.41 s wall time',
  },
  {
    dataset: 'CPU scaling · 96-seq alignment',
    dimensions: '96 seq × 3,536 bp',
    engine: 'Python CPU · full attribution',
    outcome: 'Complete carrier & parent calls',
    metric: '23.85 s wall time',
  },
];

export const MetricsStrip: React.FC = () => {
  const { ref, isVisible } = useRevealOnScroll<HTMLElement>();

  return (
    <section
      id="benchmarks"
      ref={ref}
      aria-labelledby="benchmarks-heading"
      className={`scroll-mt-16 bg-[#F6F6F3] dark:bg-[#121316] border-t border-[#E6E5DF] dark:border-[#2B2E36] transition-colors duration-200 reveal-section ${
        isVisible ? 'reveal-visible' : 'reveal-hidden'
      }`}
    >
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 py-20 sm:py-24">
        {/* Open Editorial Intro Row */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 pb-12 border-b border-[#DFDED8] dark:border-[#2B2E36] items-end">
          <div className="lg:col-span-8">
            <p className="font-mono text-[11px] tracking-[0.14em] uppercase text-[#1B9E77] dark:text-[#2DD4A4] mb-3">
              Empirical Calibration &amp; Benchmarks
            </p>
            <h2
              id="benchmarks-heading"
              className="text-[28px] sm:text-[38px] font-medium tracking-[-0.03em] leading-[1.12] text-[#141416] dark:text-[#F3F4F6] [text-wrap:balance]"
            >
              Measured on viral genomes and adversarial null controls.
            </h2>
          </div>
          <div className="lg:col-span-4 lg:text-right">
            <p className="font-mono text-[12px] text-[#5A5A62] dark:text-[#2DD4A4] tabular-nums">
              Checkpoint: ckpt_N40000 · z ≥ 6.0
            </p>
            <p className="mt-1 text-[13px] text-[#5A5A62] dark:text-[#A8ADB8]">
              Verified on CPU without GPU acceleration.
            </p>
          </div>
        </div>

        {/* 4 Open Hairline-Divided Headline Stat Columns */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-[#DFDED8] dark:divide-[#2B2E36] border-b border-[#DFDED8] dark:border-[#2B2E36]">
          {KEY_METRICS.map((item, idx) => (
            <div
              key={item.value}
              className={`py-8 sm:px-6 first:sm:pl-0 last:lg:pr-0 flex flex-col justify-between ${
                idx >= 2 ? 'sm:border-t lg:border-t-0 border-[#DFDED8] dark:border-[#2B2E36]' : ''
              }`}
            >
              <div>
                <p className="font-mono text-[22px] xl:text-[24px] font-semibold tracking-[-0.02em] leading-snug text-[#141416] dark:text-[#F3F4F6] tabular-nums">
                  {item.value}
                </p>
                <p className="mt-2 text-[12.5px] font-medium text-[#1B9E77] dark:text-[#2DD4A4]">
                  {item.label}
                </p>
              </div>
              <p className="mt-4 text-[13.5px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
                {item.context}
              </p>
            </div>
          ))}
        </div>

        {/* Mixed-Size Editorial Context Grid: What Each Benchmark Measures */}
        <div className="mt-12 space-y-6">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h3 className="text-[16px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
              What each benchmark measures
            </h3>
            <span className="font-mono text-[11.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
              Empirical viral, synthetic stress, null control &amp; formatting suites
            </span>
          </div>

          {/* Row 1: Asymmetric 7 / 5 split */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <article className="lg:col-span-7 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-8 flex flex-col justify-between">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#1B9E77] dark:text-[#2DD4A4]">
                  Localization &amp; Throughput · Empirical + Synthetic
                </p>
                <h4 className="mt-2 text-[21px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                  Full-genome coronavirus speed paired with 1.0 bp breakpoint precision.
                </h4>
                <p className="mt-3 text-[14.5px] leading-[1.65] text-[#5A5A62] dark:text-[#A8ADB8]">
                  The Sarbecovirus test measures end-to-end wall-clock throughput on a real 19-sequence, 30.6 kb coronavirus alignment (0.88 s on CPU with Rust{' '}
                  <code className="text-[13px] text-[#141416] dark:text-[#2DD4A4]">recombx</code>). To evaluate exact base-pair accuracy where true breakpoints are known, the 64-sequence, 12 kb stress test embeds 10 diverse crossovers—including a 380 bp gene conversion and a 1/64 singleton—recovering 10 / 10 events with 1.0 bp median error.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-[#ECEBE6] dark:border-[#2B2E36] grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-[12px]">
                <div>
                  <span className="block text-[#5A5A62] dark:text-[#A8ADB8] text-[10.5px]">
                    01_main (8 × 1.5 kb)
                  </span>
                  <span className="text-[#141416] dark:text-[#F3F4F6] font-semibold">
                    4 / 4 within ±6 bp
                  </span>
                </div>
                <div>
                  <span className="block text-[#5A5A62] dark:text-[#A8ADB8] text-[10.5px]">
                    64 × 12 kb stress
                  </span>
                  <span className="text-[#1B9E77] dark:text-[#2DD4A4] font-semibold">
                    10 / 10 · 1.0 bp med
                  </span>
                </div>
                <div>
                  <span className="block text-[#5A5A62] dark:text-[#A8ADB8] text-[10.5px]">
                    19 × 30.6 kb SARS-rel
                  </span>
                  <span className="text-[#141416] dark:text-[#F3F4F6] font-semibold">
                    0.88 s CPU (Rust)
                  </span>
                </div>
              </div>
            </article>

            <article className="lg:col-span-5 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-8 flex flex-col justify-between">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#4F6BF6] dark:text-[#7C93FF]">
                  False-Positive Control · 30 Null Alignments
                </p>
                <h4 className="mt-2 text-[21px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                  Zero false positives across clonal &amp; high-mutation regimes.
                </h4>
                <p className="mt-3 text-[14px] leading-[1.65] text-[#5A5A62] dark:text-[#A8ADB8]">
                  Mutation hotspots frequently trigger uncalibrated sliding-window scans. Ground Truth evaluates 30 negative controls spanning low-diversity clonal outbreaks and extreme substitution rates (μ up to 0.25), testing whether 48 column-shuffled permutations hold every null peak below z = 6.0.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-[#ECEBE6] dark:border-[#2B2E36] font-mono text-[11.5px] text-[#5A5A62] dark:text-[#A8ADB8] space-y-1">
                <p className="text-[#141416] dark:text-[#F3F4F6] font-medium">
                  Available calibrated checkpoints:
                </p>
                <p className="dark:text-[#2DD4A4]">
                  ckpt_N40000 (default) · ckpt_N10000_extreme (OOD) · ckpt_N10000_v2
                </p>
              </div>
            </article>
          </div>

          {/* Row 2: Asymmetric 5 / 7 split */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <article className="lg:col-span-5 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-6 sm:p-8 flex flex-col justify-between">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#5A5A62] dark:text-[#A8ADB8]">
                  Parser Robustness · Fixtures 08a–09h
                </p>
                <h4 className="mt-2 text-[20px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                  Deterministic across real-world FASTA quirks.
                </h4>
                <p className="mt-3 text-[14px] leading-[1.65] text-[#5A5A62] dark:text-[#A8ADB8]">
                  Validates that sequence formatting never shifts breakpoint coordinates: wrapped lowercase bases, CRLF endings, embedded spaces/digits, special header characters, and IUPAC ambiguity codes produce 5 / 5 identical calls, while 8 / 8 malformed inputs fail fast before inference.
                </p>
              </div>
              <p className="mt-5 pt-4 border-t border-[#ECEBE6] dark:border-[#2B2E36] font-mono text-[12px] text-[#1B9E77] dark:text-[#2DD4A4]">
                5 / 5 format invariant · 8 / 8 malformed rejected
              </p>
            </article>

            <article className="lg:col-span-7 bg-[#141416] dark:bg-[#1E222B] border border-transparent dark:border-[#323640] text-white rounded-2xl p-6 sm:p-8 flex flex-col justify-between">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#34D399]">
                  Event Consolidation · Stage 6 vs. 7-Method RDP5
                </p>
                <h4 className="mt-2 text-[21px] font-semibold tracking-tight text-white">
                  &gt;20× fewer redundant breakpoint calls without losing RDP5 compatibility.
                </h4>
                <p className="mt-3 text-[14.5px] leading-[1.65] text-[#C4C4CC]">
                  Classical 7-method recombination suites report overlapping hits independently per triplet, requiring hours of manual deduplication. Stage 6 merges proximal peaks within 75 bp via connected components, groups shared carrier clades, excludes co-descendants from parent attribution, and writes directly to native{' '}
                  <code className="text-[13px] text-[#34D399]">.rdp</code> project files for downstream inspection in RDP5.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-white/15 flex flex-wrap items-center justify-between gap-3 font-mono text-[12px]">
                <span className="text-[#34D399]">Level-1 consolidated events</span>
                <span className="text-[#A1A1AA]">Exports: .rdp · GFF3 · BED · FASTA · JSON · TSV</span>
              </div>
            </article>
          </div>
        </div>

        {/* Expanded Validation & CPU Scaling Ledger */}
        <div className="mt-12">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
            <h3 className="text-[15px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
              Validation &amp; CPU Timing Ledger
            </h3>
            <span className="font-mono text-[11.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
              7 verified benchmark batteries
            </span>
          </div>

          <div className="border border-[#E6E5DF] dark:border-[#2B2E36] rounded-xl overflow-hidden bg-white dark:bg-[#1A1C21]">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[#FAF9F7] dark:bg-[#16181D] border-b border-[#E6E5DF] dark:border-[#2B2E36] text-[11px] font-mono uppercase tracking-[0.06em] text-[#5A5A62] dark:text-[#A8ADB8]">
                    <th className="py-3.5 px-5 font-medium">Benchmark Suite</th>
                    <th className="py-3.5 px-5 font-medium">Scale</th>
                    <th className="py-3.5 px-5 font-medium">Calibration / Engine</th>
                    <th className="py-3.5 px-5 font-medium">Result</th>
                    <th className="py-3.5 px-5 font-medium text-right">Key Metric</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#ECEBE6] dark:divide-[#2B2E36] bg-white dark:bg-[#1A1C21] text-[13.5px]">
                  {BENCHMARK_ROWS.map((row) => (
                    <tr
                      key={row.dataset}
                      className="hover:bg-[#FAF9F7] dark:hover:bg-[#20232B] transition-colors"
                    >
                      <td className="py-3.5 px-5 font-medium text-[#141416] dark:text-[#F3F4F6]">
                        {row.dataset}
                      </td>
                      <td className="py-3.5 px-5 font-mono text-[12px] text-[#5A5A62] dark:text-[#A8ADB8] tabular-nums whitespace-nowrap">
                        {row.dimensions}
                      </td>
                      <td className="py-3.5 px-5 text-[#5A5A62] dark:text-[#A8ADB8]">
                        {row.engine}
                      </td>
                      <td className="py-3.5 px-5 font-mono text-[12px] text-[#141416] dark:text-[#E5E7EB] tabular-nums whitespace-nowrap">
                        {row.outcome}
                      </td>
                      <td className="py-3.5 px-5 font-mono text-[12px] font-medium text-[#1B9E77] dark:text-[#2DD4A4] tabular-nums text-right whitespace-nowrap">
                        {row.metric}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
