import React, { useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import {
  CLADE_COLORS,
  GENOME_LENGTH_BP,
  MOSAIC_TRACKS,
  STRESS_TEST_BREAKPOINTS,
  Z_THRESHOLD,
  buildPolylinePath,
  buildTransformerPath,
  formatBp,
  generateGenomeScanProfile,
  projectBpToX,
  projectZToY,
} from '../utils/genomeScan';

export type ScanFocusMode = 'flank_kl' | 'mosaic' | 'transformer';

interface GenomeScanPreviewProps {
  focusMode: ScanFocusMode;
  selectedEventId: number;
  onSelectEvent: (eventId: number) => void;
  isDark: boolean;
}

const SVG_WIDTH = 660;
const SVG_HEIGHT = 324;
const PLOT_LEFT = 64;
const PLOT_RIGHT = 642;
const PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT;
const PLOT_TOP = 24;
const PLOT_HEIGHT = 152;
const PLOT_BOTTOM = PLOT_TOP + PLOT_HEIGHT;

const MOSAIC_TOP = 210;
const MOSAIC_ROW_HEIGHT = 16;
const MOSAIC_ROW_GAP = 8;
const MOSAIC_BOTTOM =
  MOSAIC_TOP + MOSAIC_TRACKS.length * (MOSAIC_ROW_HEIGHT + MOSAIC_ROW_GAP) - MOSAIC_ROW_GAP;

const X_TICKS = [0, 3000, 6000, 9000, 12000];
const Y_TICKS = [0, 6, 12, 16];

export const GenomeScanPreview: React.FC<GenomeScanPreviewProps> = ({
  focusMode,
  selectedEventId,
  onSelectEvent,
  isDark,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [sweepKey, setSweepKey] = useState<number>(0);
  const [hasEnteredView, setHasEnteredView] = useState<boolean>(false);

  useEffect(() => {
    if (hasEnteredView) return;
    const el = containerRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      setHasEnteredView(true);
      setSweepKey(1);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setHasEnteredView(true);
            setSweepKey(1);
            observer.disconnect();
            break;
          }
        }
      },
      { threshold: 0.2 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [hasEnteredView]);

  const scanPoints = useMemo(
    () => generateGenomeScanProfile(STRESS_TEST_BREAKPOINTS, GENOME_LENGTH_BP, 40),
    []
  );

  const klPath = useMemo(
    () => buildPolylinePath(scanPoints, PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT, 17),
    [scanPoints]
  );

  const tfPath = useMemo(
    () => buildTransformerPath(scanPoints, PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT),
    [scanPoints]
  );

  const klFillPath = useMemo(() => {
    if (!klPath) return '';
    return `${klPath} L${PLOT_RIGHT.toFixed(1)},${PLOT_BOTTOM.toFixed(1)} L${PLOT_LEFT.toFixed(1)},${PLOT_BOTTOM.toFixed(1)} Z`;
  }, [klPath]);

  const thresholdY = projectZToY(Z_THRESHOLD, PLOT_TOP, PLOT_HEIGHT, 17);
  const selectedEvent =
    STRESS_TEST_BREAKPOINTS.find((ev) => ev.id === selectedEventId) ?? STRESS_TEST_BREAKPOINTS[0];
  const selectedX = projectBpToX(selectedEvent.bp, PLOT_LEFT, PLOT_WIDTH);
  const selectedY = projectZToY(selectedEvent.zScore, PLOT_TOP, PLOT_HEIGHT, 17);

  const klPrimary = focusMode === 'flank_kl';
  const tfPrimary = focusMode === 'transformer';
  const mosaicPrimary = focusMode === 'mosaic';

  const tealStroke = isDark ? '#2DD4A4' : '#1B9E77';
  const cobaltStroke = isDark ? '#7C93FF' : '#4F6BF6';
  const violetStroke = isDark ? '#A78BFA' : '#7570B3';
  const gridStroke = isDark ? '#252830' : '#F1F0EC';
  const vGridStroke = isDark ? '#20232B' : '#F4F3EF';
  const axisStroke = isDark ? '#3A3E49' : '#D9D8D2';
  const mutedText = isDark ? '#A8ADB8' : '#5A5A62';
  const inkText = isDark ? '#F3F4F6' : '#141416';
  const plotSurface = isDark ? '#1A1C21' : '#FFFFFF';

  return (
    <div
      ref={containerRef}
      data-testid="genome-scan-preview"
      className="bg-white/95 dark:bg-[#1A1C21]/95 backdrop-blur-sm border border-[#E5E4DE] dark:border-[#2B2E36] rounded-2xl overflow-hidden shadow-[0_24px_60px_-24px_rgba(20,20,22,0.14)] dark:shadow-[0_24px_60px_-20px_rgba(0,0,0,0.55)] transition-colors duration-200"
    >
      {/* Crisp Instrument Header */}
      <div className="px-5 py-3.5 bg-[#FAF9F7] dark:bg-[#16181D] border-b border-[#ECEBE6] dark:border-[#2B2E36] flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[12.5px] font-semibold text-[#141416] dark:text-[#F3F4F6] tracking-tight truncate">
            GT-Transformer · Live Recombination Profile
          </p>
          <p className="text-[11px] font-mono text-[#5A5A62] dark:text-[#A8ADB8] tabular-nums mt-0.5 truncate">
            64 seq · 12 kb · 48 perm · ckpt_N40000
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <span className="hidden sm:inline font-mono text-[11px] text-[#1B9E77] dark:text-[#2DD4A4] whitespace-nowrap tabular-nums">
            10 / 10 recovered
          </span>
          <button
            type="button"
            onClick={() => setSweepKey((k) => k + 1)}
            aria-label="Replay genome scan sweep"
            title="Replay scan sweep"
            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-mono text-[#5A5A62] dark:text-[#A8ADB8] bg-white dark:bg-[#1E2128] border border-[#E5E4DE] dark:border-[#323640] hover:border-[#141416] dark:hover:border-[#2DD4A4] hover:text-[#141416] dark:hover:text-[#F3F4F6] rounded-md transition-colors duration-150 whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4F6BF6]"
          >
            <RotateCcw className="w-3 h-3" aria-hidden="true" />
            <span>Sweep</span>
          </button>
        </div>
      </div>

      {/* Interactive SVG Breakpoint Scan & Mosaic Tracts */}
      <div className="px-3.5 pt-4 pb-2 bg-white dark:bg-[#1A1C21]">
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          role="img"
          aria-label="Genome-wide Flanking-KL z-score scan across 0 to 12,000 bp showing 10 breakpoints above the z equals 6.0 threshold and parental mosaic tracts underneath"
          className="w-full h-auto block select-none"
        >
          <defs>
            <linearGradient id="klAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tealStroke} stopOpacity={klPrimary ? (isDark ? '0.24' : '0.18') : '0.08'} />
              <stop offset="100%" stopColor={tealStroke} stopOpacity="0.01" />
            </linearGradient>
            <linearGradient id="scanSweepTrail" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={tealStroke} stopOpacity="0" />
              <stop offset="70%" stopColor={tealStroke} stopOpacity="0.16" />
              <stop offset="100%" stopColor={cobaltStroke} stopOpacity="0.34" />
            </linearGradient>
            <clipPath id="scanPlotClip">
              <rect
                x={PLOT_LEFT}
                y={PLOT_TOP}
                width={PLOT_WIDTH}
                height={MOSAIC_BOTTOM - PLOT_TOP + 6}
              />
            </clipPath>
          </defs>

          {/* Horizontal Y gridlines */}
          {Y_TICKS.map((tick) => {
            const y = projectZToY(tick, PLOT_TOP, PLOT_HEIGHT, 17);
            return (
              <g key={tick}>
                <line
                  x1={PLOT_LEFT}
                  y1={y}
                  x2={PLOT_RIGHT}
                  y2={y}
                  stroke={gridStroke}
                  strokeWidth="1"
                />
                <text
                  x={PLOT_LEFT - 9}
                  y={y + 3.5}
                  textAnchor="end"
                  fill={mutedText}
                  className="font-mono"
                  fontSize="9.5"
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {/* Vertical guide lines */}
          {X_TICKS.map((bp) => {
            const x = projectBpToX(bp, PLOT_LEFT, PLOT_WIDTH);
            return (
              <line
                key={bp}
                x1={x}
                y1={PLOT_TOP}
                x2={x}
                y2={PLOT_BOTTOM}
                stroke={vGridStroke}
                strokeWidth="1"
              />
            );
          })}

          {/* Y-axis label */}
          <text
            x={16}
            y={PLOT_TOP + PLOT_HEIGHT / 2}
            textAnchor="middle"
            transform={`rotate(-90 16 ${PLOT_TOP + PLOT_HEIGHT / 2})`}
            fill={mutedText}
            className="font-mono"
            fontSize="9.5"
          >
            {tfPrimary ? 'TF Posterior / KL z' : 'Flanking-KL z'}
          </text>

          {/* Shaded area under Flanking-KL curve */}
          <path d={klFillPath} fill="url(#klAreaGrad)" className="transition-opacity duration-300" />

          {/* Dashed z = 6.0 calibration threshold */}
          <line
            x1={PLOT_LEFT}
            y1={thresholdY}
            x2={PLOT_RIGHT}
            y2={thresholdY}
            stroke={mutedText}
            strokeWidth="1"
            strokeDasharray="4 3"
          />
          <text
            x={PLOT_RIGHT - 4}
            y={thresholdY - 5}
            textAnchor="end"
            fill={mutedText}
            className="font-mono"
            fontSize="9"
            paintOrder="stroke"
            stroke={plotSurface}
            strokeWidth="3"
            strokeLinejoin="round"
          >
            z = 6.0 null threshold
          </text>

          {/* Window Transformer posterior curve */}
          <path
            d={tfPath}
            fill="none"
            stroke={tfPrimary ? cobaltStroke : violetStroke}
            strokeWidth={tfPrimary ? '2.1' : '1.15'}
            strokeOpacity={tfPrimary ? '0.96' : '0.48'}
            strokeDasharray={tfPrimary ? undefined : '2.5 2.5'}
            strokeLinejoin="round"
            strokeLinecap="round"
            className="transition-all duration-300 ease-out"
          />

          {/* Flanking-KL divergence curve */}
          <path
            d={klPath}
            fill="none"
            stroke={tealStroke}
            strokeWidth={klPrimary ? '1.9' : '1.2'}
            strokeOpacity={tfPrimary ? '0.42' : '1'}
            strokeLinejoin="round"
            strokeLinecap="round"
            className="transition-all duration-300 ease-out"
          />

          {/* Selected Breakpoint vertical crosshair */}
          <line
            x1={selectedX}
            y1={PLOT_TOP}
            x2={selectedX}
            y2={MOSAIC_BOTTOM + 4}
            stroke={inkText}
            strokeWidth="1"
            strokeDasharray="2 2"
            opacity="0.45"
          />

          {/* Interactive Peak Markers */}
          {STRESS_TEST_BREAKPOINTS.map((ev) => {
            const cx = projectBpToX(ev.bp, PLOT_LEFT, PLOT_WIDTH);
            const cy = projectZToY(ev.zScore, PLOT_TOP, PLOT_HEIGHT, 17);
            const isSelected = ev.id === selectedEvent.id;
            const ringColor = tfPrimary ? cobaltStroke : tealStroke;
            return (
              <g
                key={ev.id}
                onMouseEnter={() => onSelectEvent(ev.id)}
                onClick={() => onSelectEvent(ev.id)}
                className="cursor-pointer"
              >
                <circle cx={cx} cy={cy} r={11} fill="transparent" />
                {isSelected && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={7}
                    fill={ringColor}
                    fillOpacity="0.18"
                    stroke={ringColor}
                    strokeWidth="1"
                  />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 3.5 : 2.5}
                  fill={isSelected ? inkText : plotSurface}
                  stroke={ringColor}
                  strokeWidth="1.5"
                />
              </g>
            );
          })}

          {/* Active breakpoint coordinate callout */}
          <text
            x={Math.min(Math.max(selectedX, PLOT_LEFT + 34), PLOT_RIGHT - 34)}
            y={Math.max(PLOT_TOP - 7, selectedY - 12)}
            textAnchor="middle"
            fill={inkText}
            className="font-mono font-medium"
            fontSize="9.5"
          >
            {formatBp(selectedEvent.bp)}
          </text>

          {/* Plot Baseline Axis & Ticks */}
          <line
            x1={PLOT_LEFT}
            y1={PLOT_BOTTOM}
            x2={PLOT_RIGHT}
            y2={PLOT_BOTTOM}
            stroke={axisStroke}
            strokeWidth="1"
          />
          {X_TICKS.map((bp) => {
            const x = projectBpToX(bp, PLOT_LEFT, PLOT_WIDTH);
            return (
              <g key={bp}>
                <line
                  x1={x}
                  y1={PLOT_BOTTOM}
                  x2={x}
                  y2={PLOT_BOTTOM + 4}
                  stroke={mutedText}
                  strokeWidth="1"
                />
                <text
                  x={x}
                  y={PLOT_BOTTOM + 15}
                  textAnchor={bp === 0 ? 'start' : bp === GENOME_LENGTH_BP ? 'end' : 'middle'}
                  fill={mutedText}
                  className="font-mono"
                  fontSize="9"
                >
                  {bp === 0 ? '0 bp' : `${bp / 1000} kb`}
                </text>
              </g>
            );
          })}

          {/* Mosaic Tracks Label */}
          <text
            x={PLOT_LEFT}
            y={MOSAIC_TOP - 7}
            fill={mosaicPrimary ? inkText : mutedText}
            fontWeight={mosaicPrimary ? '600' : '400'}
            className="font-mono transition-colors duration-200"
            fontSize="8.5"
            letterSpacing="0.06em"
          >
            RECOMBINANT PARENTAL MOSAIC TRACTS
          </text>

          {/* Color-Coded Parental Mosaic Tracts */}
          {MOSAIC_TRACKS.map((track, rowIdx) => {
            const y = MOSAIC_TOP + rowIdx * (MOSAIC_ROW_HEIGHT + MOSAIC_ROW_GAP);
            const isTrackActive = track.eventIds.includes(selectedEvent.id) || mosaicPrimary;
            return (
              <g key={track.id}>
                <text
                  x={PLOT_LEFT - 8}
                  y={y + 11.5}
                  textAnchor="end"
                  fill={isTrackActive ? inkText : mutedText}
                  fontWeight={isTrackActive ? '600' : '400'}
                  className="font-mono transition-colors duration-200"
                  fontSize="8.5"
                >
                  {track.label}
                </text>

                {track.segments.map((seg, segIdx) => {
                  const x1 = projectBpToX(seg.startBp, PLOT_LEFT, PLOT_WIDTH);
                  const x2 = projectBpToX(seg.endBp, PLOT_LEFT, PLOT_WIDTH);
                  const width = Math.max(1.5, x2 - x1);
                  const baseFill = CLADE_COLORS[seg.clade].fill;
                  const cladeColor =
                    isDark && seg.clade === 'CladeA'
                      ? '#64748B'
                      : isDark && seg.clade === 'CladeB'
                        ? '#2DD4A4'
                        : isDark && seg.clade === 'CladeD'
                          ? '#9B95E5'
                          : baseFill;
                  return (
                    <rect
                      key={`${track.id}-${segIdx}`}
                      x={x1}
                      y={y}
                      width={width}
                      height={MOSAIC_ROW_HEIGHT}
                      rx={3}
                      fill={cladeColor}
                      fillOpacity={isTrackActive ? 0.94 : 0.6}
                      stroke={plotSurface}
                      strokeWidth="1"
                      className="transition-opacity duration-300"
                    />
                  );
                })}
              </g>
            );
          })}

          {/* Animated Scan Line Sweep on First View / Manual Replay */}
          {sweepKey > 0 && (
            <g clipPath="url(#scanPlotClip)" className="pointer-events-none">
              <g key={sweepKey} className="scan-beam-active">
                <rect
                  x={PLOT_LEFT - 56}
                  y={PLOT_TOP}
                  width={56}
                  height={MOSAIC_BOTTOM - PLOT_TOP + 4}
                  fill="url(#scanSweepTrail)"
                />
                <line
                  x1={PLOT_LEFT}
                  y1={PLOT_TOP}
                  x2={PLOT_LEFT}
                  y2={MOSAIC_BOTTOM + 4}
                  stroke={tealStroke}
                  strokeWidth="1.6"
                />
              </g>
            </g>
          )}
        </svg>
      </div>

      {/* Compact Peak Selector Strip */}
      <div className="px-5 py-2.5 border-t border-[#ECEBE6] dark:border-[#2B2E36] bg-[#FAF9F7] dark:bg-[#16181D] flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] font-mono text-[#5A5A62] dark:text-[#A8ADB8]">
          Inspect breakpoint:
        </span>
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Detected breakpoints">
          {STRESS_TEST_BREAKPOINTS.map((ev) => {
            const active = ev.id === selectedEvent.id;
            return (
              <button
                key={ev.id}
                type="button"
                onClick={() => onSelectEvent(ev.id)}
                aria-pressed={active}
                aria-label={`Breakpoint ${ev.id} at ${formatBp(ev.bp)}`}
                className={`px-2 py-0.5 text-[11px] font-mono tabular-nums rounded-full transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[#4F6BF6] ${
                  active
                    ? 'bg-[#141416] dark:bg-[#2DD4A4] text-white dark:text-[#0E1013] font-semibold'
                    : 'bg-white dark:bg-[#1E2128] border border-[#E5E4DE] dark:border-[#323640] text-[#5A5A62] dark:text-[#A8ADB8] hover:border-[#141416] dark:hover:border-[#2DD4A4] hover:text-[#141416] dark:hover:text-[#F3F4F6]'
                }`}
              >
                #{ev.id}
              </button>
            );
          })}
        </div>
      </div>

      {/* Monospace Breakpoint Inspector Readout */}
      <div
        key={selectedEvent.id}
        data-testid="breakpoint-readout"
        className="panel-settle px-5 py-3.5 border-t border-[#ECEBE6] dark:border-[#2B2E36] bg-white dark:bg-[#1A1C21] grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11.5px]"
      >
        <div>
          <span className="block text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
            Breakpoint (95% CI)
          </span>
          <span className="font-mono font-semibold text-[#141416] dark:text-[#F3F4F6] tabular-nums">
            {formatBp(selectedEvent.bp)}
          </span>
          <span className="block font-mono text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8] tabular-nums">
            {selectedEvent.spanStart}..{selectedEvent.spanEnd}
          </span>
        </div>
        <div>
          <span className="block text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
            Hybrid Signal
          </span>
          <span className="font-mono font-semibold text-[#1B9E77] dark:text-[#2DD4A4] tabular-nums">
            z = {selectedEvent.zScore.toFixed(1)}
          </span>
          <span className="block font-mono text-[10.5px] text-[#4F6BF6] dark:text-[#7C93FF] tabular-nums">
            TF logit {selectedEvent.tfLogit.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="block text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
            Carrier Clade
          </span>
          <span className="font-mono font-semibold text-[#141416] dark:text-[#F3F4F6]">
            {selectedEvent.carriers}
          </span>
          <span className="block text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8] truncate">
            {selectedEvent.eventType}
          </span>
        </div>
        <div>
          <span className="block text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
            External Parents
          </span>
          <span className="font-mono font-semibold text-[#141416] dark:text-[#F3F4F6]">
            {selectedEvent.parents}
          </span>
          <span className="block text-[10.5px] text-[#5A5A62] dark:text-[#A8ADB8]">
            Co-descendants excluded
          </span>
        </div>
      </div>
    </div>
  );
};
