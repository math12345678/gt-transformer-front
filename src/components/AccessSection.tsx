import React, { useState } from 'react';
import { Check, Copy, Download, ExternalLink, Terminal } from 'lucide-react';
import capsidImg from '../assets/images/sarbecovirus_capsid_1795626289236.jpg';
import { useRevealOnScroll } from '../hooks/useRevealOnScroll';
import { WORKBENCH_URL } from './Navbar';

export const CLI_COMMAND =
  'recombx detect --mode hybrid --permutations 48 --format json alignment.fasta';

const COMPARISON_ROWS = [
  {
    dimension: 'Primary Use Case',
    web: 'Interactive exploration, visual triage & teaching',
    mac: 'Private pre-publication genomes on lab laptops',
    cli: 'Batch surveillance, HPC clusters, Nextflow & Snakemake',
  },
  {
    dimension: 'Execution Engine',
    web: 'Browser UI backed by ckpt_N40000 hybrid stack',
    mac: 'Bundled native engine (Apple Silicon & Intel)',
    cli: 'Zero-dependency Rust binary with embedded ONNX',
  },
  {
    dimension: 'Representative Runtime',
    web: 'Seconds per viral alignment (zero local install)',
    mac: 'Local CPU execution with no queue contention',
    cli: '0.88 s on 19-seq, 30.6 kb Sarbecovirus (CPU)',
  },
  {
    dimension: 'Data Locality',
    web: 'Fast browser upload & interactive session',
    mac: '100% offline — FASTA never leaves your Mac',
    cli: '100% local / air-gapped cluster execution',
  },
  {
    dimension: 'Visuals & Export Formats',
    web: '5 interactive plots, SimPlot, .rdp, GFF3, BED, FASTA',
    mac: 'Full local workbench UI + native .rdp & genomic exports',
    cli: 'Deterministic JSON, TSV, GFF3, BED & native .rdp',
  },
];

export const AccessSection: React.FC = () => {
  const [copied, setCopied] = useState(false);
  const [dmgNoticeVisible, setDmgNoticeVisible] = useState(false);
  const { ref, isVisible } = useRevealOnScroll<HTMLElement>();

  const handleCopyCommand = async () => {
    try {
      await navigator.clipboard.writeText(CLI_COMMAND);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = CLI_COMMAND;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section
      id="access"
      ref={ref}
      aria-labelledby="access-heading"
      className={`scroll-mt-16 bg-[#F6F6F3] dark:bg-[#121316] border-t border-[#E6E5DF] dark:border-[#2B2E36] transition-colors duration-200 reveal-section ${
        isVisible ? 'reveal-visible' : 'reveal-hidden'
      }`}
    >
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 py-20 sm:py-24">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 pb-12 border-b border-[#DFDED8] dark:border-[#2B2E36] items-end">
          <div className="lg:col-span-8">
            <p className="font-mono text-[11px] tracking-[0.14em] uppercase text-[#1B9E77] dark:text-[#2DD4A4] mb-3">
              Deployment &amp; Platform Comparison
            </p>
            <h2
              id="access-heading"
              className="text-[28px] sm:text-[38px] font-medium tracking-[-0.03em] leading-[1.12] text-[#141416] dark:text-[#F3F4F6] [text-wrap:balance]"
            >
              Run in the browser, offline on macOS, or inside high-throughput pipelines.
            </h2>
          </div>
          <div className="lg:col-span-4 lg:text-right">
            <p className="text-[14px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
              Identical calibrated checkpoints and deterministic Level-1 consolidation across every target.
            </p>
          </div>
        </div>

        {/* Asymmetric 2+1 Deployment Panels */}
        <div className="mt-10 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Panel 1 (Wide Featured): Web Workbench with bespoke sculpted viral capsid render */}
          <article className="lg:col-span-7 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-7 sm:p-8 flex flex-col justify-between overflow-hidden">
            <div className="grid grid-cols-1 sm:grid-cols-12 gap-6 items-center">
              <div className="sm:col-span-7">
                <p className="font-mono text-[11px] text-[#4F6BF6] dark:text-[#7C93FF] uppercase tracking-[0.1em]">
                  Browser Workbench
                </p>
                <h3 className="mt-2 text-[22px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                  Web Workbench
                </h3>
                <p className="mt-3 text-[14.5px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
                  Interactive 5-plot browser interface with zoomable breakpoint scans, mosaic tract maps, SimPlot, and{' '}
                  <code className="text-[12.5px] text-[#141416] dark:text-[#2DD4A4]">.rdp</code> export.
                </p>
                <ul className="mt-5 space-y-2 text-[13px] text-[#5A5A62] dark:text-[#A8ADB8]">
                  <li className="flex items-baseline gap-2">
                    <span className="font-mono text-[#1B9E77] dark:text-[#2DD4A4]">—</span>
                    <span>Zero installation; runs directly on ckpt_N40000</span>
                  </li>
                  <li className="flex items-baseline gap-2">
                    <span className="font-mono text-[#1B9E77] dark:text-[#2DD4A4]">—</span>
                    <span>Export GFF3, BED, FASTA tracts, and native RDP5 projects</span>
                  </li>
                </ul>
              </div>

              <div className="sm:col-span-5 flex items-center justify-center">
                <div className="rounded-full dark:bg-[#E7E9E4] dark:p-2 transition-colors">
                  <img
                    src={capsidImg}
                    alt="Sculpted 3D viral capsid protein structure"
                    referrerPolicy="no-referrer"
                    className="w-44 h-44 sm:w-44 sm:h-44 object-contain mix-blend-multiply brightness-[1.06] contrast-[1.06] [mask-image:radial-gradient(circle_at_50%_50%,black_44%,transparent_72%)] select-none"
                  />
                </div>
              </div>
            </div>

            <div className="mt-7 pt-5 border-t border-[#ECEBE6] dark:border-[#2B2E36] flex flex-wrap items-center justify-between gap-4">
              <a
                href={WORKBENCH_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-6 py-2.5 text-[13.5px] font-medium text-white bg-[#4F6BF6] hover:bg-[#3F5AE3] rounded-xl transition-colors duration-150 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
              >
                <span>Open Web Workbench →</span>
                <ExternalLink className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
              </a>
              <span className="font-mono text-[11.5px] text-[#5A5A62] dark:text-[#2DD4A4]">
                Live · 570,881 params
              </span>
            </div>
          </article>

          {/* Panel 2 (Focused): macOS Desktop App */}
          <article
            id="download-dmg"
            className="scroll-mt-24 lg:col-span-5 bg-white dark:bg-[#1A1C21] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-2xl p-7 sm:p-8 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between gap-3">
                <p className="font-mono text-[11px] text-[#5A5A62] dark:text-[#A8ADB8] uppercase tracking-[0.1em]">
                  Offline Native App
                </p>
                <span className="font-mono text-[11px] text-[#5A5A62] dark:text-[#2DD4A4]">
                  Universal .dmg
                </span>
              </div>

              <h3 className="mt-2 text-[22px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
                macOS Desktop App
              </h3>

              <p className="mt-3 text-[14.5px] leading-[1.6] text-[#5A5A62] dark:text-[#A8ADB8]">
                Standalone native{' '}
                <code className="text-[12.5px] text-[#141416] dark:text-[#2DD4A4]">.dmg</code> app bundling the local inference engine for offline, private genomic analysis.
              </p>

              <ul className="mt-5 space-y-2 text-[13px] text-[#5A5A62] dark:text-[#A8ADB8]">
                <li className="flex items-baseline gap-2">
                  <span className="font-mono text-[#1B9E77] dark:text-[#2DD4A4]">—</span>
                  <span>Pre-publication viral genomes stay strictly on your workstation</span>
                </li>
                <li className="flex items-baseline gap-2">
                  <span className="font-mono text-[#1B9E77] dark:text-[#2DD4A4]">—</span>
                  <span>Bundled checkpoints for Apple Silicon &amp; Intel Macs</span>
                </li>
              </ul>
            </div>

            <div className="mt-7 pt-5 border-t border-[#ECEBE6] dark:border-[#2B2E36]">
              <a
                href="#download-dmg"
                onClick={() => setDmgNoticeVisible(true)}
                className="inline-flex w-full items-center justify-center gap-2 px-5 py-2.5 text-[13.5px] font-medium text-[#141416] dark:text-[#F3F4F6] bg-[#F6F6F3] dark:bg-[#121316] border border-[#DEDDD7] dark:border-[#323640] hover:border-[#141416] dark:hover:border-[#2DD4A4] hover:bg-white dark:hover:bg-[#1E2128] rounded-xl transition-colors duration-150 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
              >
                <Download className="w-4 h-4 shrink-0 text-[#5A5A62] dark:text-[#2DD4A4]" aria-hidden="true" />
                <span>Download for macOS (.dmg)</span>
              </a>

              {dmgNoticeVisible && (
                <p
                  role="status"
                  className="panel-settle mt-3 text-[12px] leading-relaxed text-[#5A5A62] dark:text-[#A8ADB8] font-mono bg-[#FAF9F7] dark:bg-[#121316] border border-[#E6E5DF] dark:border-[#2B2E36] rounded-lg px-3 py-2"
                >
                  macOS universal .dmg installer is packaged per release. Use the live Web Workbench or Rust CLI for immediate inference.
                </p>
              )}
            </div>
          </article>

          {/* Panel 3 (Full-Width Dark Architectural Console): Rust CLI (recombx) */}
          <article className="lg:col-span-12 bg-[#141416] dark:bg-[#1A1D24] border border-transparent dark:border-[#323640] text-white rounded-2xl p-7 sm:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-center">
            <div className="lg:col-span-5">
              <div className="flex items-center gap-3">
                <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#34D399]">
                  Rust CLI · recombx
                </p>
                <span className="font-mono text-[11px] text-[#A1A1AA]">· ONNX embedded</span>
              </div>
              <h3 className="mt-2 text-[22px] font-semibold tracking-tight text-white">
                Rust CLI (recombx)
              </h3>
              <p className="mt-2.5 text-[14px] leading-[1.6] text-[#C4C4CC]">
                Zero-dependency compiled binary with embedded ONNX weights for reproducible HPC, Nextflow, and Snakemake workflows.
              </p>
            </div>

            <div className="lg:col-span-7">
              <div className="rounded-xl border border-white/15 bg-[#0F1115] overflow-hidden">
                <div className="px-4 py-2.5 border-b border-white/10 flex items-center justify-between gap-3">
                  <span className="inline-flex items-center gap-2 font-mono text-[11px] text-[#A1A1AA]">
                    <Terminal className="w-3.5 h-3.5 text-[#34D399]" aria-hidden="true" />
                    <span>Hybrid CPU inference</span>
                  </span>
                  <button
                    type="button"
                    onClick={handleCopyCommand}
                    aria-label="Copy recombx CLI command"
                    className="inline-flex items-center gap-1.5 px-3 py-1 text-[11.5px] font-mono text-white bg-white/10 hover:bg-white/15 border border-white/15 rounded-md transition-colors duration-150 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-[#34D399]" aria-hidden="true" />
                        <span className="text-[#34D399]">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-[#C4C4CC]" aria-hidden="true" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="p-4 text-[12.5px] sm:text-[13px] leading-[1.65] font-mono text-[#F4F4F6] whitespace-pre-wrap break-words">
                  <code>{CLI_COMMAND}</code>
                </pre>
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11.5px] font-mono text-[#A1A1AA]">
                <span>Outputs: JSON · TSV · GFF3 · .rdp</span>
                <span className="text-[#34D399]">0.88 s on 19-seq, 30.6 kb Sarbecovirus CPU scan</span>
              </div>
            </div>
          </article>
        </div>

        {/* Clear Side-by-Side Platform Comparison Matrix */}
        <div className="mt-12">
          <div className="flex flex-wrap items-baseline justify-between gap-3 mb-4">
            <h3 className="text-[16px] font-semibold tracking-tight text-[#141416] dark:text-[#F3F4F6]">
              Choose the right Ground Truth surface
            </h3>
            <span className="font-mono text-[11.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
              Web Workbench · macOS App · Rust CLI (recombx)
            </span>
          </div>

          <div className="border border-[#E6E5DF] dark:border-[#2B2E36] rounded-xl overflow-hidden bg-white dark:bg-[#1A1C21]">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[#FAF9F7] dark:bg-[#16181D] border-b border-[#E6E5DF] dark:border-[#2B2E36] text-[11.5px] font-mono uppercase tracking-[0.06em] text-[#5A5A62] dark:text-[#A8ADB8]">
                    <th className="py-3.5 px-5 font-medium w-[190px]">Capability</th>
                    <th className="py-3.5 px-5 font-semibold text-[#4F6BF6] dark:text-[#7C93FF]">
                      Web Workbench
                    </th>
                    <th className="py-3.5 px-5 font-semibold text-[#141416] dark:text-[#F3F4F6]">
                      macOS Desktop App
                    </th>
                    <th className="py-3.5 px-5 font-semibold text-[#1B9E77] dark:text-[#2DD4A4]">
                      Rust CLI (recombx)
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#ECEBE6] dark:divide-[#2B2E36] text-[13.5px]">
                  {COMPARISON_ROWS.map((row) => (
                    <tr
                      key={row.dimension}
                      className="hover:bg-[#FAF9F7] dark:hover:bg-[#20232B] transition-colors"
                    >
                      <th
                        scope="row"
                        className="py-3.5 px-5 font-mono text-[12px] font-medium text-[#141416] dark:text-[#F3F4F6] align-top whitespace-nowrap"
                      >
                        {row.dimension}
                      </th>
                      <td className="py-3.5 px-5 text-[#5A5A62] dark:text-[#A8ADB8] align-top">
                        {row.web}
                      </td>
                      <td className="py-3.5 px-5 text-[#5A5A62] dark:text-[#A8ADB8] align-top">
                        {row.mac}
                      </td>
                      <td className="py-3.5 px-5 text-[#141416] dark:text-[#2DD4A4] align-top">
                        {row.cli}
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
