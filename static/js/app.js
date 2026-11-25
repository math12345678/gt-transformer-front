/**
 * Frontend controller for gt-transformer (RDP5 / OpenRDP-style Web Workbench).
 * Plain, dependency-free JS driving the 2x2 desktop panes:
 *   1. Top overview strip + Top-Left FASTA nucleotide alignment viewer
 *   2. Top-Right tabbed event/diagnostic/timing/10k-validation tables
 *   3. Bottom-Left RDP5-style boundary scan plot + method controls
 *   4. Bottom-Right mosaic parental-tract map
 */

(function () {
  "use strict";

  const PALETTE = [
    "#2e8b57", // sea green (RDP5 parent tract green)
    "#3a6ea5", // steel blue
    "#b07830", // ochre brown
    "#b2473e", // brick red
    "#705898", // muted violet
    "#1e8080", // teal
    "#6b7a40", // olive
    "#8c4a6b", // plum
    "#556677", // slate
  ];

  const EMPTY_STATE_MSG = "Import or drop a FASTA alignment (.fasta, .fas, .fa), then press Run.";

  const state = {
    sourceMode: "upload",
    uploadedFastaFile: null,
    uploadedPastedFasta: "",
    uploadedGtFile: null,
    uploadedPastedGt: "",
    presets: [],
    checkpoints: [],
    scoringModes: [],
    currentRun: null,
    selectedEventId: null,
    selectedCarrierSeqIdx: null,
    alnStart: 0,
    alnWidth: 105,
    alnShowCarriersOnly: false,
    alnColorBases: true,
    lastAlnWindow: null,
    timingHistory: [],
    annotations: [],
    canvasMarkers: [],
    mosaicRowMarkers: [],
    scatterMarkers: [],
    confBarMarkers: [],
    hoveredScatterEventId: null,
    hoveredConfEventId: null,
    hoveredMosaicRowIdx: null,
    genomeHoverBp: null,
    selectedRegion: null,
    plotViewRange: null,
    lastRegionStats: null,
    eventFilterText: "",
    eventFilterMinZ: 0,
    simplotData: null,
    simplotBpMarkers: [],
    simplotHoverX: null,
    simplotHiddenSeqs: new Set(),
    simplotUserPickedQuery: false,
    recentRuns: [],
    batchResults: [],
    isDraggingPlot: false,
    dragStartBp: null,
    dragCurBp: null,
    chatHistory: [],
    currentUser: null,
    activeJobId: null,
    jobTimerInterval: null,
    jobStartTimeMs: 0,
  };

  function setupHiDpiCanvas(canvas, fallbackW, logicalH) {
    // Always measure the container width (never a stale hardcoded canvas.style.width)
    canvas.style.width = "100%";
    const parent = canvas.parentElement;
    const tabBody = canvas.closest(".pane-tab-body");
    const modalBody = canvas.closest(".modal-content");
    const parentStyle = parent ? window.getComputedStyle(parent) : null;
    const padX = parentStyle
      ? (parseFloat(parentStyle.paddingLeft) || 0) + (parseFloat(parentStyle.paddingRight) || 0)
      : 0;
    const parentW = parent ? Math.floor(parent.clientWidth - padX - 2) : 0;
    const tabBodyW = tabBody ? Math.floor(tabBody.clientWidth - 24) : 0;
    const modalW = modalBody ? Math.floor(modalBody.clientWidth - 32) : 0;
    const measured = parentW > 180 ? parentW : (tabBodyW > 180 ? tabBodyW : (modalW > 180 ? modalW : fallbackW));
    const w = Math.max(460, measured);
    const h = logicalH;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    canvas._logicalW = w;
    canvas._logicalH = h;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: w, height: h, dpr };
  }

  const el = (id) => document.getElementById(id);

  function fmtInt(n) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
    return Number(n).toLocaleString("en-US");
  }

  function fmtFloat(n, digits = 4) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
    return Number(n).toFixed(digits);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderSafeMarkdown(rawText) {
    const formatInline = (s) =>
      escapeHtml(s)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    const lines = String(rawText ?? "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let inList = false;
    lines.forEach((line) => {
      const m = line.match(/^\s*[-*]\s+(.*)$/);
      if (m) {
        if (!inList) {
          out.push("<ul>");
          inList = true;
        }
        out.push(`<li>${formatInline(m[1])}</li>`);
      } else {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        if (line.trim() !== "") out.push(`<div>${formatInline(line)}</div>`);
        else if (out.length && !out[out.length - 1].endsWith("</ul>")) out.push('<div style="height:4px;"></div>');
      }
    });
    if (inList) out.push("</ul>");
    return out.join("");
  }

  function getFullHeader(name) {
    if (!name) return "";
    const map =
      state.currentRun &&
      state.currentRun.alignment &&
      state.currentRun.alignment.full_headers
        ? state.currentRun.alignment.full_headers
        : null;
    return (map && map[name]) || name;
  }

  function showGlobalError(message) {
    const banner = el("globalErrorBanner");
    const msg = el("globalErrorMessage");
    if (banner && msg) {
      msg.textContent = message;
      banner.classList.remove("hidden");
    }
  }

  function hideGlobalError() {
    const banner = el("globalErrorBanner");
    if (banner) banner.classList.add("hidden");
  }

  function showUploadError(message) {
    const banner = el("uploadModalErrorBanner");
    const msg = el("uploadModalErrorMessage");
    if (banner && msg) {
      msg.textContent = message;
      banner.classList.remove("hidden");
    }
  }

  function hideUploadError() {
    const banner = el("uploadModalErrorBanner");
    if (banner) banner.classList.add("hidden");
  }

  function clearResultsArea(emptyMessage = EMPTY_STATE_MSG) {
    state.currentRun = null;
    state.selectedEventId = null;
    state.selectedCarrierSeqIdx = null;
    state.lastAlnWindow = null;
    state.canvasMarkers = [];
    state.mosaicRowMarkers = [];
    state.scatterMarkers = [];
    state.confBarMarkers = [];
    state.hoveredScatterEventId = null;
    state.hoveredConfEventId = null;
    state.hoveredMosaicRowIdx = null;
    state.genomeHoverBp = null;
    state.selectedRegion = null;
    state.plotViewRange = null;
    state.lastRegionStats = null;
    state.simplotData = null;
    state.simplotBpMarkers = [];
    state.simplotHoverX = null;
    if (el("regionAnalysisBar")) el("regionAnalysisBar").classList.add("hidden");
    if (el("btnResetPlotZoom")) el("btnResetPlotZoom").classList.add("hidden");
    const alnBadge = el("activeAlignmentBadge");
    if (alnBadge && !state.uploadedFastaFile && !state.uploadedPastedFasta && emptyMessage === EMPTY_STATE_MSG) {
      alnBadge.textContent = "No alignment loaded (Click Open or Drop FASTA)";
    }

    el("sumDimensions").textContent = "-";
    el("sumCols").textContent = "-";
    el("sumEvents").textContent = "-";
    el("sumRecall").textContent = "-";
    el("sumPrecision").textContent = "-";
    el("sumMedErr").textContent = "-";
    el("sumTime").textContent = "-";
    const wBadge = el("sumWarningsTag");
    if (wBadge) wBadge.classList.add("hidden");

    const stageLine = el("stageTimingSummaryLine");
    if (stageLine) {
      stageLine.textContent = "";
      stageLine.classList.add("hidden");
    }

    el("presetMetaLine").textContent = emptyMessage;
    el("alnPaneTitle").textContent = "1. Multiple Sequence Alignment & Polymorphic Site Viewer";
    el("alnMatrixContainer").innerHTML =
      `<div class="mono-muted" style="padding:12px;">${escapeHtml(emptyMessage)}</div>`;
    el("alnSeqList").innerHTML = "";
    el("level1EventsBody").innerHTML =
      `<tr><td colspan="10" class="mono-muted">${escapeHtml(emptyMessage)} <button type="button" class="mini-btn" style="margin-left:8px;" onclick="document.getElementById('btnOpenFasta').click()">Import FASTA Alignment...</button></td></tr>`;
    el("level2CarriersBody").innerHTML = `<tr><td colspan="7">No event selected.</td></tr>`;
    el("level3ParentsContainer").innerHTML = `<div class="mono-muted">No carrier selected.</div>`;
    el("toleranceMetricsBody").innerHTML = "";
    el("gtDiagnosticBody").innerHTML = "";
    el("predDiagnosticBody").innerHTML = "";
    el("currentTimingBody").innerHTML = "";
    const wBox = el("diagnosticsWarningsBox");
    if (wBox) wBox.classList.add("hidden");
    if (el("simplotLegend")) el("simplotLegend").innerHTML = "";

    // Clear all 6 canvases cleanly
    [
      "overviewStripCanvas",
      "genomeTrackCanvas",
      "mosaicMapCanvas",
      "evidenceScatterCanvas",
      "eventConfidenceCanvas",
      "simplotCanvas",
    ].forEach((cid) => {
      const c = el(cid);
      if (!c) return;
      const ctx = c.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.restore();
      }
    });
  }

  async function init() {
    bindUIEvents();
    await loadPresetsAndConfig();
    await Promise.all([loadTimingHistory(), loadAnnotations(), loadBenchmarkSummaries()]);
    clearResultsArea(EMPTY_STATE_MSG);
  }

  function bindUIEvents() {
    el("presetSelect").addEventListener("change", () => {
      const upOpt = el("presetSelect").querySelector('option[value="__upload__"]');
      if (upOpt && el("presetSelect").value !== "__upload__") {
        upOpt.remove();
      }
      state.sourceMode = "preset";
    });

    el("btnRunTop").addEventListener("click", submitInference);
    el("btnRunSide").addEventListener("click", submitInference);

    if (el("btnCancelJobTop")) el("btnCancelJobTop").addEventListener("click", cancelActiveJob);
    if (el("btnCancelJobSide")) el("btnCancelJobSide").addEventListener("click", cancelActiveJob);
    if (el("btnCancelJobBanner")) el("btnCancelJobBanner").addEventListener("click", cancelActiveJob);

    if (el("btnToggleHowToUse")) {
      el("btnToggleHowToUse").addEventListener("click", () => {
        const p = el("howToUsePanel");
        if (p) p.classList.toggle("hidden");
      });
    }
    if (el("btnCloseHowToUse")) {
      el("btnCloseHowToUse").addEventListener("click", () => {
        const p = el("howToUsePanel");
        if (p) p.classList.add("hidden");
      });
    }
    if (el("btnDownloadCsvGuide")) {
      el("btnDownloadCsvGuide").addEventListener("click", exportBreakpointsCsv);
    }
    if (el("btnDownloadJsonGuide")) {
      el("btnDownloadJsonGuide").addEventListener("click", exportFullJson);
    }

    el("scoringModeSelect").addEventListener("change", () => {
      const m = el("scoringModeSelect").value;
      if (m === "global_kl") el("zThreshInput").value = "3.0";
      else if (m === "hybrid_flank" || m === "flank_kl") el("zThreshInput").value = "6.0";
      updateIniPreview();
    });

    // Error banners
    el("btnDismissGlobalError").addEventListener("click", hideGlobalError);
    el("btnDismissUploadError").addEventListener("click", hideUploadError);

    // Tab switching
    document.querySelectorAll(".rdp-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
    });

    // Exports
    if (el("btnExportCompleteZip")) el("btnExportCompleteZip").addEventListener("click", exportCompleteBundleZip);
    if (el("btnExportRdpProject")) el("btnExportRdpProject").addEventListener("click", exportRdpProject);
    if (el("btnExportGff3")) el("btnExportGff3").addEventListener("click", exportGff3Track);
    if (el("btnExportBed")) el("btnExportBed").addEventListener("click", exportBedTrack);
    if (el("btnExportRecFasta")) el("btnExportRecFasta").addEventListener("click", exportRecombinantFasta);
    el("btnExportOpenRdpTsv").addEventListener("click", exportOpenRdpTsv);
    el("btnExportBpCsv").addEventListener("click", exportBreakpointsCsv);
    el("btnExportSeqCsv").addEventListener("click", exportSequenceCallsCsv);
    el("btnExportJson").addEventListener("click", exportFullJson);
    if (el("btnExportHtmlReport")) el("btnExportHtmlReport").addEventListener("click", exportHtmlReport);

    // Open / Upload Modal
    const openUploadModal = () => {
      hideUploadError();
      el("openModal").classList.remove("hidden");
    };
    const closeUploadModal = () => {
      hideUploadError();
      el("openModal").classList.add("hidden");
    };
    el("btnOpenFasta").addEventListener("click", openUploadModal);
    el("btnOpenSide").addEventListener("click", openUploadModal);
    el("btnCloseOpenModal").addEventListener("click", closeUploadModal);
    el("btnCancelOpenModal").addEventListener("click", closeUploadModal);
    el("btnClearUpload").addEventListener("click", () => {
      const upOpt = el("presetSelect").querySelector('option[value="__upload__"]');
      if (upOpt) upOpt.remove();
      state.sourceMode = "upload";
      state.uploadedFastaFile = null;
      state.uploadedPastedFasta = "";
      state.uploadedGtFile = null;
      state.uploadedPastedGt = "";
      el("fastaFileInput").value = "";
      el("pastedFastaInput").value = "";
      el("gtFileInput").value = "";
      el("pastedGtInput").value = "";
      const alnBadge = el("activeAlignmentBadge");
      if (alnBadge) alnBadge.textContent = "No alignment loaded (Click Open or Drop FASTA)";
    });
    el("uploadForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      hideUploadError();
      const files = Array.from(el("fastaFileInput").files || []);
      if (files.length > 1) {
        closeUploadModal();
        await runBatchFastaFiles(files);
        return;
      }
      state.uploadedFastaFile = files[0] || null;
      state.uploadedPastedFasta = el("pastedFastaInput").value.trim();
      state.uploadedGtFile = el("gtFileInput").files[0] || null;
      state.uploadedPastedGt = el("pastedGtInput").value.trim();
      if (!state.uploadedFastaFile && !state.uploadedPastedFasta) {
        showUploadError("Choose a FASTA file or paste FASTA sequence text.");
        return;
      }
      const uploadName = state.uploadedFastaFile
        ? state.uploadedFastaFile.name
        : "pasted_alignment.fasta";
      const alnBadge = el("activeAlignmentBadge");
      if (alnBadge) alnBadge.textContent = `Loaded: ${uploadName}`;
      let upOpt = el("presetSelect").querySelector('option[value="__upload__"]');
      if (!upOpt) {
        upOpt = document.createElement("option");
        upOpt.value = "__upload__";
        el("presetSelect").prepend(upOpt);
      }
      upOpt.textContent = `Uploaded: ${uploadName}`;
      el("presetSelect").value = "__upload__";
      state.sourceMode = "upload";
      closeUploadModal();
      await submitInference();
    });

    // Options Modal
    const openOpts = () => {
      updateIniPreview();
      el("optionsModal").classList.remove("hidden");
    };
    const closeOpts = () => el("optionsModal").classList.add("hidden");
    el("btnOptionsTop").addEventListener("click", openOpts);
    el("btnOptionsSide").addEventListener("click", openOpts);
    el("btnCloseOptionsModal").addEventListener("click", closeOpts);
    el("btnCancelOptions").addEventListener("click", closeOpts);
    el("btnApplyOptions").addEventListener("click", async () => {
      closeOpts();
      await submitInference();
    });
    ["zThreshInput", "tfLogitThreshInput", "flankWindowInput", "mergeRadiusInput", "betaInput", "nullPolicyInput"].forEach(
      (id) => el(id).addEventListener("input", updateIniPreview)
    );
    ["checkpointSelect", "optMirrorTta"].forEach((id) => {
      if (el(id)) el(id).addEventListener("change", updateIniPreview);
    });

    // Alignment Viewer navigation
    el("btnAlnPrevFast").addEventListener("click", () => moveAlignmentWindow(-500));
    el("btnAlnPrev").addEventListener("click", () => moveAlignmentWindow(-80));
    el("btnAlnNext").addEventListener("click", () => moveAlignmentWindow(80));
    el("btnAlnNextFast").addEventListener("click", () => moveAlignmentWindow(500));
    el("btnAlnGo").addEventListener("click", () => {
      const val = parseInt(el("alnJumpInput").value, 10) || 0;
      loadAlignmentWindow(val);
    });
    el("btnAlnToEvent").addEventListener("click", jumpAlignmentToSelectedEvent);
    el("btnToggleCarriersOnly").addEventListener("click", () => {
      state.alnShowCarriersOnly = !state.alnShowCarriersOnly;
      el("btnToggleCarriersOnly").textContent = state.alnShowCarriersOnly
        ? "Show All Sequences"
        : "Carriers + Parents Only";
      if (state.lastAlnWindow) renderAlignmentWindow(state.lastAlnWindow);
    });
    if (el("btnToggleBaseColors")) {
      el("btnToggleBaseColors").addEventListener("click", () => {
        state.alnColorBases = !state.alnColorBases;
        el("btnToggleBaseColors").textContent = `Color Bases: ${state.alnColorBases ? "ON" : "OFF"}`;
        if (state.lastAlnWindow) renderAlignmentWindow(state.lastAlnWindow);
      });
    }
    if (el("eventSearchInput")) {
      el("eventSearchInput").addEventListener("input", () => {
        state.eventFilterText = el("eventSearchInput").value || "";
        if (state.currentRun) renderOverviewAndParents(state.currentRun);
      });
    }
    if (el("eventMinZInput")) {
      el("eventMinZInput").addEventListener("input", () => {
        state.eventFilterMinZ = parseFloat(el("eventMinZInput").value) || 0;
        if (state.currentRun) renderOverviewAndParents(state.currentRun);
      });
    }
    if (el("simplotQuerySelect")) {
      el("simplotQuerySelect").addEventListener("change", () => {
        state.simplotUserPickedQuery = true;
        loadSimplotProfile(Number(el("simplotQuerySelect").value));
      });
    }
    if (el("simplotWindowSelect")) {
      el("simplotWindowSelect").addEventListener("change", () => {
        const q = el("simplotQuerySelect") ? Number(el("simplotQuerySelect").value) : null;
        loadSimplotProfile(q);
      });
    }
    if (el("simplotAutoScaleY")) {
      el("simplotAutoScaleY").addEventListener("change", () => {
        if (state.simplotData && state.currentRun) {
          drawSimplotCanvas(state.simplotData, state.currentRun);
        }
      });
    }
    if (el("btnDownloadSimplot")) {
      el("btnDownloadSimplot").addEventListener("click", () => {
        if (!state.currentRun) return;
        downloadCanvasPng(el("simplotCanvas"), `${state.currentRun.alignment.dataset_name}_simplot_similarity.png`);
      });
    }
    if (el("simplotCanvas")) {
      const spc = el("simplotCanvas");
      spc.addEventListener("mousemove", (e) => {
        const d = state.simplotData;
        if (!d || !d.positions_bp || !d.positions_bp.length) return hidePlotFloatTip();
        const rect = spc.getBoundingClientRect();
        const w = spc._logicalW || Math.max(1, rect.width);
        const lx = e.clientX - rect.left;
        const padL = 52, padR = 18;
        const pw = w - padL - padR;
        if (lx < padL || lx > padL + pw) {
          if (state.simplotHoverX !== null) {
            state.simplotHoverX = null;
            drawSimplotCanvas(d, state.currentRun);
          }
          return hidePlotFloatTip();
        }
        state.simplotHoverX = lx;
        drawSimplotCanvas(d, state.currentRun);

        const L = Math.max(1, d.genome_length_bp);
        const targetBp = ((lx - padL) / Math.max(1, pw)) * L;
        let bestI = 0;
        let bestDist = Infinity;
        d.positions_bp.forEach((bp, idx) => {
          const dist = Math.abs(bp - targetBp);
          if (dist < bestDist) { bestDist = dist; bestI = idx; }
        });
        const atBp = d.positions_bp[bestI];
        const parentRoles = getSimplotParentRoles(d, state.currentRun);
        const lines = (d.series || [])
          .filter((s) => !state.simplotHiddenSeqs.has(s.seq_idx))
          .map((s) => {
            const role = parentRoles.leftNames.has(s.seq_name) || parentRoles.leftIdxs.has(s.seq_idx)
              ? " <span style='color:#1d4ed8;font-weight:700;'>[Left Parent]</span>"
              : (parentRoles.rightNames.has(s.seq_name) || parentRoles.rightIdxs.has(s.seq_idx)
                ? " <span style='color:#dc2626;font-weight:700;'>[Right Parent]</span>"
                : "");
            return { name: s.seq_name, role, val: s.identity_pct[bestI] };
          })
          .filter((x) => x.val !== null && x.val !== undefined)
          .sort((a, b) => b.val - a.val)
          .slice(0, 8)
          .map((x) => `${escapeHtml(x.name)}${x.role}: <strong>${x.val.toFixed(1)}%</strong>`);
        showPlotFloatTip(e, `<strong>Window @ ${fmtInt(atBp)} bp (Query: ${escapeHtml(d.query_seq_name)})</strong><br/>${lines.join("<br/>")}`);
      });
      spc.addEventListener("mouseleave", () => {
        if (state.simplotHoverX !== null && state.simplotData && state.currentRun) {
          state.simplotHoverX = null;
          drawSimplotCanvas(state.simplotData, state.currentRun);
        }
        hidePlotFloatTip();
      });
      spc.addEventListener("click", (e) => {
        if (!state.currentRun || !state.simplotBpMarkers.length) return;
        const rect = spc.getBoundingClientRect();
        const lx = e.clientX - rect.left;
        let best = null;
        let bestD = 16;
        state.simplotBpMarkers.forEach((m) => {
          const d = Math.abs(m.x - lx);
          if (d < bestD) { bestD = d; best = m; }
        });
        if (best) selectEvent(best.eventId, true);
      });
    }

    // Mosaic filter
    el("chkMosaicSelectedOnly").addEventListener("change", () => {
      if (state.currentRun) drawMosaicMap(state.currentRun);
    });

    // Annotation modal
    el("btnOpenAnnotationModal").addEventListener("click", () => openAnnotationModal("general", ""));
    el("btnCloseModal").addEventListener("click", closeAnnotationModal);
    el("btnCancelModal").addEventListener("click", closeAnnotationModal);
    el("annotationForm").addEventListener("submit", handleSaveAnnotation);

    // Canvas interactions (drag-select region + zoom + hover + click)
    const plotCanvas = el("genomeTrackCanvas");
    const canvasXToBp = (clientX) => {
      if (!state.currentRun) return 0;
      const rect = plotCanvas.getBoundingClientRect();
      const padL = 50;
      const padR = 46;
      const plotW = Math.max(1, rect.width - padL - padR);
      const { start, span, L } = getPlotViewBounds(state.currentRun);
      return Math.max(1, Math.min(L, Math.round(start + ((clientX - rect.left - padL) / plotW) * span)));
    };
    plotCanvas.addEventListener("mousedown", (e) => {
      if (!state.currentRun || e.button !== 0) return;
      const bp = canvasXToBp(e.clientX);
      state.isDraggingPlot = true;
      state.dragStartBp = bp;
      state.dragCurBp = bp;
    });
    window.addEventListener("mouseup", (e) => {
      if (!state.isDraggingPlot) return;
      state.isDraggingPlot = false;
      const endBp = canvasXToBp(e.clientX);
      const a = Math.min(state.dragStartBp, endBp);
      const b = Math.max(state.dragStartBp, endBp);
      if (b - a >= 10) {
        state.selectedRegion = { startBp: a, endBp: b };
        renderRegionAnalysis();
        if (state.currentRun) drawGenomeTrack(state.currentRun);
      }
      state.dragStartBp = null;
      state.dragCurBp = null;
    });
    plotCanvas.addEventListener("mousemove", (e) => {
      if (!state.currentRun) return;
      state.genomeHoverBp = canvasXToBp(e.clientX);
      if (state.isDraggingPlot) {
        state.dragCurBp = state.genomeHoverBp;
      }
      drawGenomeTrack(state.currentRun);
      handleCanvasHover(e);
    });
    plotCanvas.addEventListener("mouseleave", () => {
      el("trackTooltip").classList.add("hidden");
      if (state.genomeHoverBp !== null && state.currentRun) {
        state.genomeHoverBp = null;
        drawGenomeTrack(state.currentRun);
      }
    });
    plotCanvas.addEventListener("click", handleCanvasClick);
    const resetPlotZoom = () => {
      if (!state.plotViewRange) return;
      state.plotViewRange = null;
      renderRegionAnalysis();
      if (state.currentRun) {
        drawOverviewStrip(state.currentRun);
        drawGenomeTrack(state.currentRun);
      }
    };
    plotCanvas.addEventListener("dblclick", resetPlotZoom);

    if (el("btnZoomRegionPlot")) {
      el("btnZoomRegionPlot").addEventListener("click", () => {
        if (!state.currentRun || !state.selectedRegion) return;
        const { startBp, endBp } = state.selectedRegion;
        if (endBp <= startBp) return;
        state.plotViewRange = { startBp, endBp };
        renderRegionAnalysis();
        drawOverviewStrip(state.currentRun);
        drawGenomeTrack(state.currentRun);
      });
    }

    if (el("mosaicMapCanvas")) {
      const mc = el("mosaicMapCanvas");
      mc.addEventListener("click", (e) => {
        if (!state.currentRun || !state.mosaicRowMarkers.length) return;
        const rect = mc.getBoundingClientRect();
        const my = e.clientY - rect.top;
        const hit = state.mosaicRowMarkers.find((m) => my >= m.y1 && my <= m.y2);
        if (hit) selectEvent(hit.eventId, true);
      });
      mc.addEventListener("mousemove", (e) => {
        if (!state.currentRun || !state.mosaicRowMarkers.length) return hidePlotFloatTip();
        const rect = mc.getBoundingClientRect();
        const my = e.clientY - rect.top;
        const hitIdx = state.mosaicRowMarkers.findIndex((m) => my >= m.y1 && my <= m.y2);
        const hit = hitIdx >= 0 ? state.mosaicRowMarkers[hitIdx] : null;
        if (state.hoveredMosaicRowIdx !== (hitIdx >= 0 ? hitIdx : null)) {
          state.hoveredMosaicRowIdx = hitIdx >= 0 ? hitIdx : null;
          drawMosaicMap(state.currentRun);
        }
        if (hit && hit.tipHtml) showPlotFloatTip(e, hit.tipHtml);
        else hidePlotFloatTip();
      });
      mc.addEventListener("mouseleave", () => {
        if (state.hoveredMosaicRowIdx !== null && state.currentRun) {
          state.hoveredMosaicRowIdx = null;
          drawMosaicMap(state.currentRun);
        }
        hidePlotFloatTip();
      });
    }
    if (el("evidenceScatterCanvas")) {
      const sc = el("evidenceScatterCanvas");
      const findScatterHit = (e) => {
        const rect = sc.getBoundingClientRect();
        const lx = e.clientX - rect.left;
        const ly = e.clientY - rect.top;
        let best = null;
        let bestD = 18;
        state.scatterMarkers.forEach((m) => {
          const d = Math.hypot(m.x - lx, m.y - ly);
          if (d < bestD) { bestD = d; best = m; }
        });
        return best;
      };
      sc.addEventListener("click", (e) => {
        if (!state.currentRun || !state.scatterMarkers.length) return;
        const best = findScatterHit(e);
        if (best) selectEvent(best.eventId, true);
      });
      sc.addEventListener("mousemove", (e) => {
        if (!state.currentRun || !state.scatterMarkers.length) return hidePlotFloatTip();
        const best = findScatterHit(e);
        const nextHover = best ? best.eventId : null;
        if (state.hoveredScatterEventId !== nextHover) {
          state.hoveredScatterEventId = nextHover;
          drawEvidenceScatter(state.currentRun);
        }
        if (best && best.tipHtml) showPlotFloatTip(e, best.tipHtml);
        else hidePlotFloatTip();
      });
      sc.addEventListener("mouseleave", () => {
        if (state.hoveredScatterEventId !== null && state.currentRun) {
          state.hoveredScatterEventId = null;
          drawEvidenceScatter(state.currentRun);
        }
        hidePlotFloatTip();
      });
    }
    if (el("eventConfidenceCanvas")) {
      const cc = el("eventConfidenceCanvas");
      const findConfHit = (e) => {
        const rect = cc.getBoundingClientRect();
        const lx = e.clientX - rect.left;
        return state.confBarMarkers.find((m) => lx >= m.x1 && lx <= m.x2) || null;
      };
      cc.addEventListener("click", (e) => {
        if (!state.currentRun || !state.confBarMarkers.length) return;
        const hit = findConfHit(e);
        if (hit) selectEvent(hit.eventId, true);
      });
      cc.addEventListener("mousemove", (e) => {
        if (!state.currentRun || !state.confBarMarkers.length) return hidePlotFloatTip();
        const hit = findConfHit(e);
        const nextHover = hit ? hit.eventId : null;
        if (state.hoveredConfEventId !== nextHover) {
          state.hoveredConfEventId = nextHover;
          drawEventConfidenceBars(state.currentRun);
        }
        if (hit && hit.tipHtml) showPlotFloatTip(e, hit.tipHtml);
        else hidePlotFloatTip();
      });
      cc.addEventListener("mouseleave", () => {
        if (state.hoveredConfEventId !== null && state.currentRun) {
          state.hoveredConfEventId = null;
          drawEventConfidenceBars(state.currentRun);
        }
        hidePlotFloatTip();
      });
    }
    if (el("btnInspectRegion")) el("btnInspectRegion").addEventListener("click", openRegionInspectorModal);
    if (el("btnCloseRegionModal")) el("btnCloseRegionModal").addEventListener("click", () => el("regionModal").classList.add("hidden"));
    if (el("regionRefSeqSelect")) {
      el("regionRefSeqSelect").addEventListener("change", () => {
        if (!state.selectedRegion || !state.currentRun) return;
        loadRegionStats(Number(el("regionRefSeqSelect").value));
      });
    }
    if (el("btnDownloadRegionPlot")) {
      el("btnDownloadRegionPlot").addEventListener("click", () => {
        if (!state.currentRun || !state.lastRegionStats) return;
        const ds = state.currentRun.alignment.dataset_name;
        const { start_bp, end_bp } = state.lastRegionStats;
        downloadCanvasPng(el("regionIdentityCanvas"), `${ds}_region_${start_bp}_${end_bp}.png`);
      });
    }
    if (el("btnRegionAskAi")) {
      el("btnRegionAskAi").addEventListener("click", () => {
        if (!state.selectedRegion) return;
        const { startBp, endBp } = state.selectedRegion;
        el("regionModal").classList.add("hidden");
        el("aiChatDrawer").classList.remove("hidden");
        refreshAiChatStatus();
        sendAiChatMessage(`Analyze pairwise similarity, recombination signals, and diversity in the selected region from ${startBp} bp to ${endBp} bp.`);
      });
    }
    if (el("btnCloseBatchModal")) el("btnCloseBatchModal").addEventListener("click", () => el("batchModal").classList.add("hidden"));
    if (el("btnExportBatchCsv")) el("btnExportBatchCsv").addEventListener("click", exportBatchCsv);
    if (el("btnExportBatchJson")) el("btnExportBatchJson").addEventListener("click", exportBatchJson);

    if (el("btnResetPlotZoom")) {
      el("btnResetPlotZoom").addEventListener("click", resetPlotZoom);
    }
    if (el("btnClearRegion")) {
      el("btnClearRegion").addEventListener("click", () => {
        state.selectedRegion = null;
        renderRegionAnalysis();
        if (state.currentRun) drawGenomeTrack(state.currentRun);
      });
    }
    if (el("btnZoomRegionAln")) {
      el("btnZoomRegionAln").addEventListener("click", () => {
        if (state.selectedRegion) loadAlignmentWindow(Math.max(0, state.selectedRegion.startBp - 1));
      });
    }
    if (el("btnAskAiRegion")) {
      el("btnAskAiRegion").addEventListener("click", () => {
        if (!state.selectedRegion) return;
        const { startBp, endBp } = state.selectedRegion;
        el("aiChatDrawer").classList.remove("hidden");
        refreshAiChatStatus();
        sendAiChatMessage(`Analyze and explain the selected alignment region from ${startBp} bp to ${endBp} bp.`);
      });
    }

    // Multi-plot exports
    if (el("btnDownloadScanPlot")) {
      el("btnDownloadScanPlot").addEventListener("click", () => {
        if (!state.currentRun) return;
        downloadCanvasPng(el("genomeTrackCanvas"), `${state.currentRun.alignment.dataset_name}_scan_profile.png`);
      });
    }
    if (el("btnDownloadMosaicPlot")) {
      el("btnDownloadMosaicPlot").addEventListener("click", () => {
        if (!state.currentRun) return;
        downloadCanvasPng(el("mosaicMapCanvas"), `${state.currentRun.alignment.dataset_name}_mosaic_map.png`);
      });
    }
    if (el("btnDownloadScatterPlot")) {
      el("btnDownloadScatterPlot").addEventListener("click", () => {
        if (!state.currentRun) return;
        drawEvidenceScatter(state.currentRun);
        downloadCanvasPng(el("evidenceScatterCanvas"), `${state.currentRun.alignment.dataset_name}_evidence_scatter.png`);
      });
    }
    if (el("btnDownloadConfPlot")) {
      el("btnDownloadConfPlot").addEventListener("click", () => {
        if (!state.currentRun) return;
        drawEventConfidenceBars(state.currentRun);
        downloadCanvasPng(el("eventConfidenceCanvas"), `${state.currentRun.alignment.dataset_name}_event_confidence.png`);
      });
    }
    if (el("btnExportAllPlots")) {
      el("btnExportAllPlots").addEventListener("click", exportAllFourPlots);
    }

    // AI Assistant Chat Widget
    if (el("btnToggleAiChat")) {
      el("btnToggleAiChat").addEventListener("click", () => {
        el("aiChatDrawer").classList.toggle("hidden");
        if (!el("aiChatDrawer").classList.contains("hidden")) refreshAiChatStatus();
      });
    }
    if (el("btnCloseAiChat")) {
      el("btnCloseAiChat").addEventListener("click", () => el("aiChatDrawer").classList.add("hidden"));
    }
    if (el("btnClearAiChat")) {
      el("btnClearAiChat").addEventListener("click", () => {
        state.chatHistory = [];
        const box = el("aiChatMessages");
        if (box) {
          box.innerHTML =
            '<div class="ai-msg ai-msg-bot">Chat cleared. Ask anything about your breakpoints, Z-scores, Transformer probabilities, parents, or a selected region on the graph.</div>';
        }
      });
    }
    if (el("aiChatForm")) {
      el("aiChatForm").addEventListener("submit", (e) => {
        e.preventDefault();
        const q = el("aiChatInput").value.trim();
        if (!q) return;
        el("aiChatInput").value = "";
        sendAiChatMessage(q);
      });
    }
    document.querySelectorAll(".ai-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-q");
        if (q) sendAiChatMessage(q);
      });
    });

    // Private Accounts, History & Admin Modal (Username & Password only)
    refreshCurrentUser();
    if (el("btnAuthAccount")) {
      el("btnAuthAccount").addEventListener("click", () => {
        el("authErrorMsg").classList.add("hidden");
        el("authModal").classList.remove("hidden");
      });
    }
    if (el("btnCloseAuthModal")) el("btnCloseAuthModal").addEventListener("click", () => el("authModal").classList.add("hidden"));
    if (el("btnLoginSubmit")) el("btnLoginSubmit").addEventListener("click", () => submitAuth("/api/auth/login"));
    if (el("btnRegisterSubmit")) el("btnRegisterSubmit").addEventListener("click", () => submitAuth("/api/auth/register"));
    if (el("btnLogoutSubmit")) {
      el("btnLogoutSubmit").addEventListener("click", async () => {
        await fetch("/api/auth/logout", { method: "POST" });
        state.currentUser = null;
        updateAuthUi();
        el("authModal").classList.add("hidden");
      });
    }
    if (el("btnCompareRuns")) el("btnCompareRuns").addEventListener("click", openCompareModal);
    if (el("btnCloseCompareModal")) el("btnCloseCompareModal").addEventListener("click", () => el("compareModal").classList.add("hidden"));
    if (el("btnMyHistory")) el("btnMyHistory").addEventListener("click", openMyHistoryModal);
    if (el("btnCloseHistoryModal")) el("btnCloseHistoryModal").addEventListener("click", () => el("historyModal").classList.add("hidden"));
    if (el("btnAdminPanel")) el("btnAdminPanel").addEventListener("click", openAdminModal);
    if (el("btnCloseAdminModal")) el("btnCloseAdminModal").addEventListener("click", () => el("adminModal").classList.add("hidden"));

    // Export dropdown toggle
    if (el("btnExportMenu") && el("exportMenuPanel")) {
      el("btnExportMenu").addEventListener("click", (e) => {
        e.stopPropagation();
        el("exportMenuPanel").classList.toggle("hidden");
      });
      document.addEventListener("click", () => el("exportMenuPanel").classList.add("hidden"));
    }

    // Plain-English Summary AI button
    if (el("btnExplainSummaryAi")) {
      el("btnExplainSummaryAi").addEventListener("click", () => {
        el("aiChatDrawer").classList.remove("hidden");
        refreshAiChatStatus();
        sendAiChatMessage("Give me a complete interpretation of all detected recombination events, carriers, and Left->Right parental lineages in this run.");
      });
    }

    // Drag-and-drop FASTA anywhere on the window
    const dropOverlay = el("dropOverlay");
    let dragDepth = 0;
    document.addEventListener("dragenter", (e) => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) {
        dragDepth += 1;
        if (dropOverlay) dropOverlay.classList.remove("hidden");
      }
    });
    document.addEventListener("dragleave", () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0 && dropOverlay) dropOverlay.classList.add("hidden");
    });
    document.addEventListener("dragover", (e) => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) {
        e.preventDefault();
      }
    });
    document.addEventListener("drop", async (e) => {
      if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
      e.preventDefault();
      dragDepth = 0;
      if (dropOverlay) dropOverlay.classList.add("hidden");
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 1) {
        await runBatchFastaFiles(files);
        return;
      }
      const file = files[0];
      state.uploadedFastaFile = file;
      state.uploadedPastedFasta = "";
      state.uploadedGtFile = null;
      state.uploadedPastedGt = "";
      const alnBadge = el("activeAlignmentBadge");
      if (alnBadge) alnBadge.textContent = `Loaded: ${file.name}`;
      let upOpt = el("presetSelect").querySelector('option[value="__upload__"]');
      if (!upOpt) {
        upOpt = document.createElement("option");
        upOpt.value = "__upload__";
        el("presetSelect").prepend(upOpt);
      }
      upOpt.textContent = `Uploaded: ${file.name}`;
      el("presetSelect").value = "__upload__";
      state.sourceMode = "upload";
      await submitInference();
    });

    // Keyboard shortcuts: Escape closes modals/chat drawer; Ctrl/Cmd+Enter runs inference
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        ["openModal", "optionsModal", "annotationModal", "authModal", "historyModal", "adminModal", "compareModal", "regionModal", "batchModal", "aiChatDrawer"].forEach((id) => {
          const node = el(id);
          if (node) node.classList.add("hidden");
        });
      } else if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        submitInference();
      }
    });

    const ovCanvas = el("overviewStripCanvas");
    ovCanvas.addEventListener("click", (e) => {
      if (!state.currentRun) return;
      const rect = e.target.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
      const bp = Math.round(frac * state.currentRun.alignment.genome_length_bp);
      loadAlignmentWindow(Math.max(0, bp - Math.floor(state.alnWidth / 2)));
    });
    ovCanvas.addEventListener("mousemove", (e) => {
      if (!state.currentRun) return;
      const rect = ovCanvas.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
      const bp = Math.round(frac * state.currentRun.alignment.genome_length_bp);
      const nearEv = (state.currentRun.consolidated_events || []).find(
        (ev) => bp >= ev.span_orig_start - 15 && bp <= ev.span_orig_end + 15
      );
      const msg = nearEv
        ? `<strong>Pos ${fmtInt(bp)} bp</strong> · Event #${nearEv.event_id} (${fmtInt(nearEv.most_plausible_bp_orig)} bp, z=${nearEv.score_z.toFixed(2)}) (click to jump)`
        : `<strong>Pos ${fmtInt(bp)} bp</strong> (click to jump) alignment here`;
      showPlotFloatTip(e, msg);
    });
    ovCanvas.addEventListener("mouseleave", hidePlotFloatTip);

    window.addEventListener("resize", () => {
      if (state.currentRun) {
        drawOverviewStrip(state.currentRun);
        drawGenomeTrack(state.currentRun);
        drawMosaicMap(state.currentRun);
        if (!el("tab-evidence").classList.contains("hidden")) {
          drawEvidenceScatter(state.currentRun);
          drawEventConfidenceBars(state.currentRun);
        }
        if (!el("tab-simplot").classList.contains("hidden") && state.simplotData) {
          drawSimplotCanvas(state.simplotData, state.currentRun);
        }
      }
    });
  }

  function switchTab(tabId) {
    document.querySelectorAll(".rdp-tab").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-content").forEach((p) => {
      p.classList.toggle("hidden", p.id !== tabId);
    });
    if (tabId === "tab-evidence" && state.currentRun) {
      requestAnimationFrame(() => {
        drawEvidenceScatter(state.currentRun);
        drawEventConfidenceBars(state.currentRun);
      });
    }
    if (tabId === "tab-simplot" && state.currentRun) {
      requestAnimationFrame(() => {
        if (!state.simplotData) loadSimplotProfile();
        else drawSimplotCanvas(state.simplotData, state.currentRun);
      });
    }
  }

  function updateIniPreview() {
    const ini = [
      `[GTTransformer]`,
      `checkpoint = ${el("checkpointSelect").value || "ckpt_N10000"}`,
      `scoring_mode = ${el("scoringModeSelect").value || "hybrid_flank"}`,
      `use_mirror_tta = ${el("optMirrorTta")?.checked ? "true" : "false"}`,
      `null_policy = ${el("nullPolicyInput").value}`,
      `tf_logit_thresh = ${el("tfLogitThreshInput").value}`,
      ``,
      `[FlankKL]`,
      `flank_window = ${el("flankWindowInput").value}`,
      `z_thresh = ${el("zThreshInput").value}`,
      `beta = ${el("betaInput").value}`,
      `merge_radius_bp = ${el("mergeRadiusInput").value}`,
      `eval_tolerance_bp = ${el("evalToleranceInput").value}`,
    ].join("\n");
    el("iniPreviewBox").textContent = ini;
  }

  async function loadPresetsAndConfig() {
    const res = await fetch("/api/presets");
    const data = await res.json();
    state.presets = data.presets || [];
    state.checkpoints = data.checkpoints || [];
    state.scoringModes = data.scoring_modes || [];

    el("presetSelect").innerHTML = state.presets
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
      .join("");
    el("checkpointSelect").innerHTML = state.checkpoints
      .map((c) => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`)
      .join("");
    el("scoringModeSelect").innerHTML = state.scoringModes
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`)
      .join("");
    updateIniPreview();
  }

  function setJobRunningUI(isRunning, statusLabel = "Running", stageLabel = "parsing") {
    const btnTop = el("btnRunTop");
    const btnSide = el("btnRunSide");
    const cancelTop = el("btnCancelJobTop");
    const cancelSide = el("btnCancelJobSide");
    const statusBar = el("jobStatusBar");
    const statusBadge = el("jobStatusBadge");
    const stageText = el("jobStageText");

    btnTop.disabled = isRunning;
    btnSide.disabled = isRunning;
    btnTop.textContent = isRunning ? statusLabel : "Run";
    btnSide.textContent = isRunning ? statusLabel : "Run";

    if (cancelTop) cancelTop.classList.toggle("hidden", !isRunning);
    if (cancelSide) cancelSide.classList.toggle("hidden", !isRunning);
    if (statusBar) statusBar.classList.toggle("hidden", !isRunning);
    if (statusBadge) statusBadge.textContent = statusLabel;
    if (stageText) stageText.textContent = `Stage: ${stageLabel}`;
  }

  function startClientElapsedTimer() {
    stopClientElapsedTimer();
    state.jobStartTimeMs = Date.now();
    const elapsedEl = el("jobElapsedText");
    if (elapsedEl) elapsedEl.textContent = "(0.0 s)";
    state.jobTimerInterval = setInterval(() => {
      const sec = (Date.now() - state.jobStartTimeMs) / 1000;
      if (elapsedEl) elapsedEl.textContent = `(${sec.toFixed(1)} s)`;
    }, 200);
  }

  function stopClientElapsedTimer() {
    if (state.jobTimerInterval) {
      clearInterval(state.jobTimerInterval);
      state.jobTimerInterval = null;
    }
  }

  async function cancelActiveJob() {
    if (!state.activeJobId) return;
    try {
      await fetch(`/api/jobs/${encodeURIComponent(state.activeJobId)}/cancel`, {
        method: "POST",
      });
    } catch (err) {
      console.error("Cancel request failed:", err);
    }
  }

  function sleepMs(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function submitInference() {
    if (state.sourceMode === "upload" && !state.uploadedFastaFile && !state.uploadedPastedFasta) {
      hideGlobalError();
      showUploadError("Import a FASTA file (.fasta, .fas, .fa) or paste aligned sequences to run detection.");
      if (el("openModal")) el("openModal").classList.remove("hidden");
      return;
    }
    // Phase 1: Clear old results and old error banner immediately on Run
    hideGlobalError();
    clearResultsArea("Running recombination analysis...");
    setJobRunningUI(true, "Queued", "parsing");
    startClientElapsedTimer();

    try {
      const fd = new FormData();
      fd.append("checkpoint", el("checkpointSelect").value);
      fd.append("scoring_mode", el("scoringModeSelect").value);
      fd.append("use_mirror_tta", el("optMirrorTta")?.checked ? "true" : "false");
      fd.append("null_policy", el("nullPolicyInput").value);
      fd.append("flank_window", el("flankWindowInput").value);
      fd.append("beta", el("betaInput").value);
      fd.append("z_thresh", el("zThreshInput").value);
      fd.append("tf_logit_thresh", el("tfLogitThreshInput").value);
      fd.append("merge_radius_bp", el("mergeRadiusInput").value);
      fd.append("eval_tolerance_bp", el("evalToleranceInput").value);

      if (state.sourceMode === "preset") {
        fd.append("preset_id", el("presetSelect").value);
      } else {
        if (state.uploadedFastaFile) {
          fd.append("fasta_file", state.uploadedFastaFile);
        } else if (state.uploadedPastedFasta) {
          fd.append("pasted_fasta", state.uploadedPastedFasta);
        }
        if (state.uploadedGtFile) {
          fd.append("gt_file", state.uploadedGtFile);
        } else if (state.uploadedPastedGt) {
          fd.append("pasted_gt", state.uploadedPastedGt);
        }
      }

      let res;
      try {
        res = await fetch("/api/jobs", { method: "POST", body: fd });
      } catch (netErr) {
        clearResultsArea(EMPTY_STATE_MSG);
        showGlobalError("Network error: Unable to connect to inference server (" + (netErr.message || "offline") + ")");
        return;
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}: ${res.statusText}` }));
        clearResultsArea(EMPTY_STATE_MSG);
        showGlobalError(err.detail || `Validation failed with status ${res.status}`);
        return;
      }

      const jobInit = await res.json();
      const jobId = jobInit.job_id;
      state.activeJobId = jobId;

      // Poll GET /api/jobs/<id> every 1 s
      while (state.activeJobId === jobId) {
        const pollRes = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
        if (!pollRes.ok) {
          const pollErr = await pollRes.json().catch(() => ({ detail: `HTTP ${pollRes.status}` }));
          clearResultsArea(EMPTY_STATE_MSG);
          showGlobalError(pollErr.detail || `Failed to poll job ${jobId}`);
          return;
        }
        const job = await pollRes.json();
        const statusCap = job.status === "queued" ? "Queued" : "Running";
        setJobRunningUI(true, statusCap, job.stage || "running");

        if (job.status === "done" && job.result) {
          const data = job.result;
          displayRunPayload(data);
          const evs = data.consolidated_events || [];
          const startBp =
            evs.length > 0 ? Math.max(0, evs[0].most_plausible_bp_orig - Math.floor(state.alnWidth / 2)) : 0;
          await Promise.all([loadAlignmentWindow(startBp), loadTimingHistory()]);
          return;
        }

        if (job.status === "error") {
          clearResultsArea(EMPTY_STATE_MSG);
          showGlobalError(job.error || "Inference job failed.");
          return;
        }

        if (job.status === "cancelled") {
          clearResultsArea(EMPTY_STATE_MSG);
          showGlobalError(job.error || "Job cancelled by user.");
          return;
        }

        await sleepMs(1000);
      }
    } finally {
      state.activeJobId = null;
      stopClientElapsedTimer();
      setJobRunningUI(false);
    }
  }

  function rememberRun(run) {
    if (!run || !run.timings || !run.timings.run_id) return;
    const rid = run.timings.run_id;
    state.recentRuns = [run, ...state.recentRuns.filter((r) => r && r.timings && r.timings.run_id !== rid)].slice(0, 20);
  }

  function displayRunPayload(data) {
    if (!data) return;
    state.currentRun = data;
    const evs = data.consolidated_events || [];
    state.selectedEventId = evs.length > 0 ? evs[0].event_id : null;
    state.selectedCarrierSeqIdx =
      evs.length > 0 && evs[0].recombinants && evs[0].recombinants.length > 0
        ? evs[0].recombinants[0].seq_idx
        : null;
    renderRun(data);
  }

  function renderRun(run) {
    rememberRun(run);
    const aln = run.alignment;
    const tim = run.timings;
    const diag = run.diagnostics;
    const evs = run.consolidated_events || [];
    const rawPeaks = run.raw_peaks || [];

    el("engineStatusBadge").textContent =
      `live · ${fmtInt(run.model_param_count)} params · ${tim.run_id}`;
    const alnBadge = el("activeAlignmentBadge");
    if (alnBadge) {
      alnBadge.textContent = `${aln.dataset_name} (${aln.n_sequences} × ${fmtInt(aln.genome_length_bp)} bp)`;
    }

    el("sumDimensions").textContent = `${aln.n_sequences} × ${fmtInt(aln.genome_length_bp)} bp`;
    el("sumCols").textContent = `${fmtInt(aln.n_informative_cols)}→${fmtInt(aln.n_transformed_cols)} (W=${aln.window_size})`;
    el("sumEvents").textContent = `${evs.length} cons / ${rawPeaks.length} raw`;

    if (diag.has_ground_truth && diag.tolerance_metrics.length > 0) {
      const m =
        diag.tolerance_metrics.find((x) => x.tolerance_bp === diag.active_tolerance_bp) ||
        diag.tolerance_metrics[diag.tolerance_metrics.length - 1];
      if (diag.n_ground_truth_breakpoints === 0) {
        el("sumRecall").textContent = "N/A (0 GT)";
        el("sumPrecision").textContent = m.fp === 0 ? "0 FP (100% clean)" : `${m.fp} FP`;
        el("sumMedErr").textContent = "-";
      } else {
        el("sumRecall").textContent = `${(m.recall * 100).toFixed(1)}% (${m.tp}/${diag.n_ground_truth_breakpoints})`;
        el("sumPrecision").textContent = `P=${(m.precision * 100).toFixed(1)}% F1=${m.f1.toFixed(3)}`;
        el("sumMedErr").textContent = m.median_error_bp !== null ? `${m.median_error_bp.toFixed(1)} bp` : "-";
      }
    } else {
      el("sumRecall").textContent = "Unlabelled";
      el("sumPrecision").textContent = "Unlabelled";
      el("sumMedErr").textContent = "-";
    }
    el("sumTime").textContent = `${tim.total_ms.toFixed(0)} ms (${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s)`;

    const stageLine = el("stageTimingSummaryLine");
    if (stageLine) {
      if (tim.stage_summary_line) {
        stageLine.textContent = tim.stage_summary_line;
        stageLine.classList.remove("hidden");
      } else {
        stageLine.classList.add("hidden");
      }
    }

    const warnings = run.warnings || [];
    const wBadge = el("sumWarningsTag");
    if (wBadge) {
      if (warnings.length > 0) {
        wBadge.textContent = `⚠️ ${warnings.length} Warning${warnings.length > 1 ? "s" : ""}`;
        wBadge.title = warnings.join("\n") + "\n(Click to view in Diagnostics tab)";
        wBadge.classList.remove("hidden");
        wBadge.onclick = () => switchTab("tab-diagnostics");
      } else {
        wBadge.classList.add("hidden");
      }
    }

    const ttaTag = run.config.use_mirror_tta ? " | TTA=ON" : "";
    el("presetMetaLine").textContent =
      `${aln.dataset_name} (${aln.source_type}) | mode=${run.config.scoring_mode} | ckpt=${run.config.checkpoint}${ttaTag} | GC=${aln.gc_content_pct}% | Edge dead-zone: ±${aln.edge_dead_zone_bp || 30} bp`;

    const plainCard = el("plainEnglishSummary");
    const plainTxt = el("plainEnglishSummaryText");
    if (plainCard && plainTxt) {
      const evs = run.consolidated_events || [];
      if (!evs.length) {
        plainTxt.textContent = `Summary: No recombination breakpoints were detected across ${aln.n_sequences} sequences (${fmtInt(aln.genome_length_bp)} bp).`;
      } else {
        const preview = evs.slice(0, 3).map((e) => {
          const rec = (e.recombinant_sequences && e.recombinant_sequences[0]) || "unknown";
          const r0 = (e.recombinants && e.recombinants[0]) || null;
          const par = r0 ? `${r0.ext_parent_left} → ${r0.ext_parent_right}` : "";
          return `${rec} @ ${fmtInt(e.most_plausible_bp_orig)} bp${par ? ` (${par})` : ""}`;
        }).join("; ");
        const more = evs.length > 3 ? ` (+${evs.length - 3} more)` : "";
        plainTxt.textContent = `Summary: Detected ${evs.length} breakpoint(s) across ${aln.n_sequences} sequences (${fmtInt(aln.genome_length_bp)} bp): ${preview}${more}.`;
      }
      plainCard.classList.remove("hidden");
    }

    state.selectedRegion = null;
    state.plotViewRange = null;
    state.lastRegionStats = null;
    state.simplotData = null;
    state.simplotBpMarkers = [];
    state.simplotHoverX = null;
    state.simplotHiddenSeqs = new Set();
    state.simplotUserPickedQuery = false;
    state.hoveredScatterEventId = null;
    state.hoveredConfEventId = null;
    state.hoveredMosaicRowIdx = null;
    state.genomeHoverBp = null;
    if (el("regionAnalysisBar")) el("regionAnalysisBar").classList.add("hidden");
    if (el("btnResetPlotZoom")) el("btnResetPlotZoom").classList.add("hidden");
    drawOverviewStrip(run);
    drawGenomeTrack(run);
    drawMosaicMap(run);
    drawEvidenceScatter(run);
    drawEventConfidenceBars(run);
    renderOverviewAndParents(run);
    renderDiagnosticsTab(run);
    renderCurrentTimings(tim);
    loadSimplotProfile().catch(() => {});
  }

  // -------------------------------------------------------------------------
  // Top-Left: Real FASTA Alignment Viewer
  // -------------------------------------------------------------------------
  async function loadAlignmentWindow(startBp) {
    if (!state.currentRun) return;
    const runId = state.currentRun.timings.run_id;
    const L = state.currentRun.alignment.genome_length_bp;
    const clampedStart = Math.max(0, Math.min(Math.max(0, L - state.alnWidth), Math.round(startBp)));
    state.alnStart = clampedStart;
    el("alnJumpInput").value = String(clampedStart);

    const res = await fetch(
      `/api/alignment/window?run_id=${encodeURIComponent(runId)}&start=${clampedStart}&width=${state.alnWidth}`
    );
    if (!res.ok) return;
    const data = await res.json();
    state.lastAlnWindow = data;
    renderAlignmentWindow(data);
  }

  function moveAlignmentWindow(delta) {
    loadAlignmentWindow(state.alnStart + delta);
  }

  function jumpAlignmentToSelectedEvent() {
    if (!state.currentRun) return;
    const ev = (state.currentRun.consolidated_events || []).find((x) => x.event_id === state.selectedEventId);
    const bp = ev ? ev.most_plausible_bp_orig : 0;
    loadAlignmentWindow(Math.max(0, bp - Math.floor(state.alnWidth / 2)));
  }

  function renderAlignmentWindow(win) {
    const start = win.start_bp;
    const end = win.end_bp;
    const infoSet = new Set(win.informative_cols || []);
    el("alnPaneTitle").textContent =
      `Sequence Alignment (${fmtInt(start + 1)}..${fmtInt(end)} of ${fmtInt(win.genome_length_bp)} bp)`;

    // Identify carriers & parents of selected event for highlighting
    const selectedEv = state.currentRun
      ? (state.currentRun.consolidated_events || []).find((x) => x.event_id === state.selectedEventId)
      : null;
    const carrierSet = new Set(selectedEv ? selectedEv.recombinant_sequences : []);
    const parentSet = new Set();
    if (selectedEv && selectedEv.recombinants) {
      selectedEv.recombinants.forEach((r) => {
        parentSet.add(r.ext_parent_left);
        parentSet.add(r.ext_parent_right);
      });
    }
    const evBp = selectedEv ? selectedEv.most_plausible_bp_orig : -1;

    let seqRows = win.sequences || [];
    if (state.alnShowCarriersOnly && (carrierSet.size > 0 || parentSet.size > 0)) {
      seqRows = seqRows.filter((s) => carrierSet.has(s.seq_idx) || parentSet.has(s.seq_idx));
    }

    // Build ruler header
    let rulerCells = `<th style="padding:1px 5px;">Seq</th><th>`;
    for (let pos = start; pos < end; pos++) {
      const pos1 = pos + 1;
      const mark = pos1 % 10 === 0 ? "|" : pos1 % 5 === 0 ? ":" : ".";
      rulerCells += `<span class="nt-cell" title="pos ${pos1}">${mark}</span>`;
    }
    rulerCells += `</th>`;

    // Build nucleotide rows
    const baseClassOf = (ch) => {
      if (!state.alnColorBases) return "";
      const u = ch.toUpperCase();
      if (u === "A") return "nt-A";
      if (u === "C") return "nt-C";
      if (u === "G") return "nt-G";
      if (u === "T" || u === "U") return "nt-T";
      return "nt-gap";
    };
    const bodyHtml = seqRows
      .map((row) => {
        const isC = carrierSet.has(row.seq_idx);
        const isP = !isC && parentSet.has(row.seq_idx);
        const cls = isC ? "aln-seq-row aln-row-carrier" : isP ? "aln-seq-row aln-row-parent" : "aln-seq-row";
        let seqHtml = "";
        const txt = row.slice || "";
        for (let k = 0; k < txt.length; k++) {
          const absPos = start + k;
          const clsList = ["nt-cell"];
          const bCls = baseClassOf(txt[k]);
          if (bCls) clsList.push(bCls);
          if (infoSet.has(absPos)) clsList.push("nt-info");
          if (absPos === evBp) clsList.push("nt-bp");
          seqHtml += `<span class="${clsList.join(" ")}">${escapeHtml(txt[k])}</span>`;
        }
        const fullHdr = getFullHeader(row.name);
        return `<tr class="${cls}"><td class="aln-row-idx" title="${escapeHtml(fullHdr)}">${row.seq_idx}: ${escapeHtml(row.name)}</td><td>${seqHtml}</td></tr>`;
      })
      .join("");

    const tblCls = state.alnColorBases ? "aln-matrix-table aln-colored" : "aln-matrix-table";
    el("alnMatrixContainer").innerHTML =
      `<table class="${tblCls}"><thead><tr class="aln-ruler-row">${rulerCells}</tr></thead><tbody>${bodyHtml}</tbody></table>`;

    // Sidebar sequence list (like RDP5 "Show All Sequences")
    el("alnSeqList").innerHTML = (win.sequences || [])
      .map((row) => {
        const isC = carrierSet.has(row.seq_idx);
        const isP = !isC && parentSet.has(row.seq_idx);
        const cls = isC ? "seq-list-item is-carrier" : isP ? "seq-list-item is-parent" : "seq-list-item";
        const tag = isC ? " [REC]" : isP ? " [PAR]" : "";
        const fullHdr = getFullHeader(row.name);
        return `<div class="${cls}" title="${escapeHtml(fullHdr)}">${row.seq_idx}: ${escapeHtml(row.name)}${tag}</div>`;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // Top-Right: Overview, Parents, Diagnostics, Timings, 10k Validation
  // -------------------------------------------------------------------------
  function statusTagHtml(status) {
    if (status === "TP") return `<span class="tag tag-tp">TP</span>`;
    if (status === "FP") return `<span class="tag tag-fp">FP</span>`;
    if (status === "FN") return `<span class="tag tag-fn">FN</span>`;
    return `<span class="tag tag-unlabelled">UNLABELLED</span>`;
  }

  function selectEvent(eventId, jumpAlign = false) {
    if (!state.currentRun) return;
    state.selectedEventId = eventId;
    const ev = (state.currentRun.consolidated_events || []).find((x) => x.event_id === eventId);
    if (ev && ev.recombinants && ev.recombinants.length > 0) {
      state.selectedCarrierSeqIdx = ev.recombinants[0].seq_idx;
    } else {
      state.selectedCarrierSeqIdx = null;
    }
    renderOverviewAndParents(state.currentRun);
    drawOverviewStrip(state.currentRun);
    drawGenomeTrack(state.currentRun);
    drawMosaicMap(state.currentRun);
    drawEvidenceScatter(state.currentRun);
    drawEventConfidenceBars(state.currentRun);
    if (
      !state.simplotUserPickedQuery &&
      state.selectedCarrierSeqIdx !== null &&
      (!state.simplotData || state.simplotData.query_seq_idx !== state.selectedCarrierSeqIdx)
    ) {
      loadSimplotProfile(state.selectedCarrierSeqIdx).catch(() => {});
    } else if (state.simplotData) {
      drawSimplotCanvas(state.simplotData, state.currentRun);
    }
    if (jumpAlign) {
      jumpAlignmentToSelectedEvent();
    } else if (state.lastAlnWindow) {
      renderAlignmentWindow(state.lastAlnWindow);
    }
  }

  function renderOverviewAndParents(run) {
    const evs = run.consolidated_events || [];
    const l1Body = el("level1EventsBody");
    const countSpan = el("eventFilterCount");

    if (evs.length === 0) {
      if (countSpan) countSpan.textContent = "";
      l1Body.innerHTML = `<tr><td colspan="10">No consolidated recombination events detected above threshold.</td></tr>`;
      el("level2CarriersBody").innerHTML = `<tr><td colspan="7">No event selected.</td></tr>`;
      el("level3ParentsContainer").innerHTML = `<div class="mono-muted">No carrier selected.</div>`;
      return;
    }

    const qText = (state.eventFilterText || "").trim().toLowerCase();
    const minZ = Number(state.eventFilterMinZ) || 0;
    const filteredEvs = evs.filter((ev) => {
      if (ev.score_z < minZ) return false;
      if (!qText) return true;
      const ancId = ev.ancestral_event_id || ev.event_id;
      const hay = [
        ev.event_id,
        `a${ancId}`,
        ev.most_plausible_bp_orig,
        ev.signal_attribution,
        ...(ev.recombinants || []).flatMap((r) => [
          r.seq_name,
          r.ext_parent_left_name || r.ext_parent_left,
          r.ext_parent_right_name || r.ext_parent_right,
        ]),
      ]
        .join(" ")
        .toLowerCase();
      return hay.includes(qText);
    });
    if (countSpan) {
      countSpan.textContent = (qText || minZ > 0) ? `Showing ${filteredEvs.length} / ${evs.length}` : "";
    }

    if (filteredEvs.length === 0) {
      l1Body.innerHTML = `<tr><td colspan="10" class="mono-muted">No breakpoints match current filter (${evs.length} total).</td></tr>`;
    } else {
      l1Body.innerHTML = filteredEvs
        .map((ev) => {
        const sel = ev.event_id === state.selectedEventId ? ' class="selected-row"' : "";
        const lead = ev.recombinants && ev.recombinants[0] ? ev.recombinants[0] : null;
        const parentsStr = lead
          ? `${lead.ext_parent_left_name || lead.ext_parent_left} → ${lead.ext_parent_right_name || lead.ext_parent_right}`
          : "-";
        const carrierNames = (ev.recombinants || []).map((r) => r.seq_name || String(r.seq_idx));
        const carrierDisplay =
          carrierNames.length > 0
            ? carrierNames.slice(0, 6).join(", ")
            : ev.recombinant_sequences.slice(0, 6).join(", ");
        const carrierTooltips = (ev.recombinants || [])
          .map((r) => getFullHeader(r.seq_name))
          .join("\n");
        const attrLabels = {
          "hybrid_flank": "Hybrid",
          "flank_kl_heuristic_only": "flank_kl (heuristic)",
          "transformer_corroborated": "Corroborated",
          "standalone_tf": "Transformer",
        };
        const attrText = attrLabels[ev.signal_attribution] || ev.signal_attribution;
        const attrTag = `<span class="tag-attribution tag-${escapeHtml(ev.signal_attribution)}" title="${escapeHtml(ev.attribution_note || '')}">${escapeHtml(attrText)}</span>`;
        const ancId = ev.ancestral_event_id || ev.event_id;
        return (
          `<tr${sel} data-row-event="${ev.event_id}">` +
          `<td class="num"><strong>${ev.event_id}</strong><span class="ancestral-pill" title="Ancestral co-descendant group A${ancId}">A${ancId}</span></td>` +
          `<td class="mono">${attrTag}</td>` +
          `<td class="num"><strong>${fmtInt(ev.most_plausible_bp_orig)}</strong></td>` +
          `<td class="mono">${fmtInt(ev.span_orig_start)}..${fmtInt(ev.span_orig_end)}</td>` +
          `<td class="num">${ev.n_peaks_consolidated}</td>` +
          `<td class="num">${ev.score_z.toFixed(2)}</td>` +
          `<td class="mono">${ev.tf_logit.toFixed(2)} (${ev.tf_prob.toFixed(2)})</td>` +
          `<td class="mono" title="${escapeHtml(carrierTooltips)}">${escapeHtml(carrierDisplay)}</td>` +
          `<td class="mono">${escapeHtml(parentsStr)}</td>` +
          `<td>${statusTagHtml(ev.match_status)}</td>` +
          `</tr>`
        );
      })
      .join("");
    }

    l1Body.querySelectorAll("tr[data-row-event]").forEach((tr) => {
      tr.addEventListener("click", () => {
        selectEvent(Number(tr.getAttribute("data-row-event")), true);
      });
    });

    const selectedEv = evs.find((x) => x.event_id === state.selectedEventId) || evs[0];
    el("level2HeaderTitle").textContent =
      `Level 2 Recombinant Sequences - Event #${selectedEv.event_id} (${fmtInt(selectedEv.most_plausible_bp_orig)} bp)`;

    const recombs = selectedEv.recombinants || [];
    const l2Body = el("level2CarriersBody");
    if (recombs.length === 0) {
      l2Body.innerHTML = `<tr><td colspan="7">No carrier sequences recorded.</td></tr>`;
      el("level3ParentsContainer").innerHTML = "";
      return;
    }

    l2Body.innerHTML = recombs
      .map((r) => {
        const sel = r.seq_idx === state.selectedCarrierSeqIdx ? ' class="selected-row"' : "";
        const fullHdr = getFullHeader(r.seq_name);
        const pLeftStr = `${r.ext_parent_left_name || r.parent_left_name} (#${r.ext_parent_left})`;
        const pRightStr = `${r.ext_parent_right_name || r.parent_right_name} (#${r.ext_parent_right})`;
        return (
          `<tr${sel} data-row-carrier="${r.seq_idx}">` +
          `<td class="num"><strong>${r.seq_idx}</strong></td>` +
          `<td class="mono" title="${escapeHtml(fullHdr)}">${escapeHtml(r.seq_name)}</td>` +
          `<td class="num">${fmtFloat(r.p_switch, 4)}</td>` +
          `<td class="mono">${escapeHtml(pLeftStr)}</td>` +
          `<td class="mono">${escapeHtml(pRightStr)}</td>` +
          `<td class="num">${fmtFloat(r.left_sim, 4)}</td>` +
          `<td class="num">${fmtFloat(r.right_sim, 4)}</td>` +
          `</tr>`
        );
      })
      .join("");

    l2Body.querySelectorAll("tr[data-row-carrier]").forEach((tr) => {
      tr.addEventListener("click", () => {
        state.selectedCarrierSeqIdx = Number(tr.getAttribute("data-row-carrier"));
        renderOverviewAndParents(state.currentRun);
      });
    });

    const selectedCarrier =
      recombs.find((r) => r.seq_idx === state.selectedCarrierSeqIdx) || recombs[0];
    renderLevel3Parents(selectedEv, selectedCarrier);
  }

  function renderLevel3Parents(ev, carrier) {
    el("level3HeaderTitle").textContent =
      `Level 3 Ranked Parental Candidates - Carrier Seq ${carrier.seq_idx} (${carrier.seq_name})`;
    const mkTable = (title, cands) => {
      const rows = (cands || [])
        .map(
          (c, i) =>
            `<tr>` +
            `<td class="num">#${i + 1}</td>` +
            `<td class="num"><strong>${c.seq_idx}</strong></td>` +
            `<td class="mono" title="${escapeHtml(getFullHeader(c.seq_name))}">${escapeHtml(c.seq_name)}</td>` +
            `<td class="num">${fmtFloat(c.similarity, 4)}</td>` +
            `<td class="mono">${c.is_external_to_event ? "external" : "co-carrier"}</td>` +
            `</tr>`
        )
        .join("");
      return (
        `<div class="pane-subhead mt-6">${escapeHtml(title)}</div>` +
        `<div class="table-scroll"><table class="rdp-table">` +
        `<thead><tr><th>Rank</th><th>seq_idx</th><th>Name</th><th>Similarity</th><th>Scope</th></tr></thead>` +
        `<tbody>${rows || '<tr><td colspan="5">None</td></tr>'}</tbody></table></div>`
      );
    };
    el("level3ParentsContainer").innerHTML =
      mkTable(`Left Flank (< ${fmtInt(ev.most_plausible_bp_orig)} bp)`, carrier.left_candidates) +
      mkTable(`Right Flank (> ${fmtInt(ev.most_plausible_bp_orig)} bp)`, carrier.right_candidates);
  }

  function renderDiagnosticsTab(run) {
    const diag = run.diagnostics;
    const evs = run.consolidated_events || [];

    el("diagVerdictBanner").textContent = (diag.summary_notes || []).join(" ");

    const warnings = run.warnings || [];
    const wBox = el("diagnosticsWarningsBox");
    const wList = el("diagWarningsList");
    const wCount = el("diagWarningsCount");
    if (wBox && wList) {
      if (warnings.length > 0) {
        if (wCount) wCount.textContent = warnings.length;
        wList.innerHTML = warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
        wBox.classList.remove("hidden");
      } else {
        wList.innerHTML = "";
        wBox.classList.add("hidden");
      }
    }

    const tolBody = el("toleranceMetricsBody");
    if (!diag.has_ground_truth || diag.tolerance_metrics.length === 0) {
      tolBody.innerHTML = `<tr><td colspan="10">No ground-truth log loaded. Use Open FASTA... to attach a SantaSim .txt, .json, or .csv ground-truth log.</td></tr>`;
    } else {
      tolBody.innerHTML = diag.tolerance_metrics
        .map((m) => {
          const isAct = m.tolerance_bp === diag.active_tolerance_bp;
          const style = isAct ? ' style="background-color:#e6eef8;font-weight:700;"' : "";
          const sign = m.lift >= 0 ? "+" : "";
          return (
            `<tr${style}>` +
            `<td class="mono">±${m.tolerance_bp} bp</td>` +
            `<td class="num">${m.tp}</td>` +
            `<td class="num">${m.fp}</td>` +
            `<td class="num">${m.fn}</td>` +
            `<td class="num">${(m.recall * 100).toFixed(1)}%</td>` +
            `<td class="num">${(m.precision * 100).toFixed(1)}%</td>` +
            `<td class="num">${m.f1.toFixed(3)}</td>` +
            `<td class="num">${m.control_f1.toFixed(3)}</td>` +
            `<td class="num">${sign}${m.lift.toFixed(3)}</td>` +
            `<td class="num">${m.median_error_bp !== null ? m.median_error_bp.toFixed(1) + " bp" : "-"}</td>` +
            `</tr>`
          );
        })
        .join("");
    }

    const gtBody = el("gtDiagnosticBody");
    if (!diag.has_ground_truth) {
      gtBody.innerHTML = `<tr><td colspan="9">No ground-truth breakpoints loaded.</td></tr>`;
    } else if (diag.ground_truth_items.length === 0) {
      gtBody.innerHTML = `<tr><td colspan="9">Clonal negative control (0 true breakpoints). Any calls below are false positives.</td></tr>`;
    } else {
      gtBody.innerHTML = diag.ground_truth_items
        .map((g) => {
          const st = g.detected ? statusTagHtml("TP") : statusTagHtml("FN");
          const matchedStr = g.detected ? `#${g.matched_pred_event_id} (${fmtInt(g.matched_pred_bp_orig)})` : "-";
          const errStr = g.detected && g.bp_error !== null ? `${g.bp_error}` : "-";
          const localZ =
            g.nearest_subthreshold_score_z !== null
              ? `${g.nearest_subthreshold_score_z.toFixed(2)} @ ${fmtInt(g.nearest_subthreshold_peak_bp)}`
              : "-";
          const targetId = `GT ${g.gt_bp_orig} bp (Event ${g.gt_event_id ?? "?"})`;
          return (
            `<tr>` +
            `<td>${st}</td>` +
            `<td class="num"><strong>${fmtInt(g.gt_bp_orig)}</strong></td>` +
            `<td class="mono">${escapeHtml(g.gt_event_id ?? "-")}</td>` +
            `<td>${escapeHtml(g.event_type)}</td>` +
            `<td class="mono">${matchedStr}</td>` +
            `<td class="num">${errStr}</td>` +
            `<td class="mono">${localZ}</td>` +
            `<td>${escapeHtml(g.failure_hypothesis)}</td>` +
            `<td><button type="button" class="btn-link" data-ann-type="${g.detected ? "predicted_event" : "missed_gt_breakpoint"}" data-ann-id="${escapeHtml(targetId)}">Note</button></td>` +
            `</tr>`
          );
        })
        .join("");
    }

    const predBody = el("predDiagnosticBody");
    if (evs.length === 0) {
      predBody.innerHTML = `<tr><td colspan="8">No predicted calls above threshold.</td></tr>`;
    } else {
      predBody.innerHTML = evs
        .map((ev) => {
          const st = statusTagHtml(ev.match_status);
          const gtStr = ev.matched_gt_bp !== null ? `${fmtInt(ev.matched_gt_bp)}` : "-";
          const errStr = ev.bp_error !== null ? `${ev.bp_error}` : "-";
          const targetId = `Event #${ev.event_id} (${ev.most_plausible_bp_orig} bp)`;
          const explStr = ev.attribution_note
            ? `${ev.diagnostic_explanation ? ev.diagnostic_explanation + ' | ' : ''}${ev.attribution_note}`
            : ev.diagnostic_explanation;
          return (
            `<tr>` +
            `<td>${st}</td>` +
            `<td class="num">#${ev.event_id}</td>` +
            `<td class="num"><strong>${fmtInt(ev.most_plausible_bp_orig)}</strong></td>` +
            `<td class="mono">${ev.score_z.toFixed(2)} / ${ev.tf_logit.toFixed(2)}</td>` +
            `<td class="mono">${gtStr}</td>` +
            `<td class="num">${errStr}</td>` +
            `<td>${escapeHtml(explStr)}</td>` +
            `<td><button type="button" class="btn-link" data-ann-type="predicted_event" data-ann-id="${escapeHtml(targetId)}">Note</button></td>` +
            `</tr>`
          );
        })
        .join("");
    }

    document.querySelectorAll("[data-ann-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        openAnnotationModal(btn.getAttribute("data-ann-type"), btn.getAttribute("data-ann-id"));
      });
    });
  }

  function renderCurrentTimings(tim) {
    el("timingRunSubtitle").textContent =
      `run_id: ${tim.run_id} | total: ${tim.total_ms.toFixed(2)} ms | ${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s`;
    const rowsHtml = (tim.stages || [])
      .map((s) => {
        const itemsStr = s.items_processed !== null ? `${fmtInt(s.items_processed)} ${s.items_unit || ""}` : "-";
        const tpStr = s.throughput !== null ? `${fmtInt(Math.round(s.throughput))} ${s.throughput_unit || ""}` : "-";
        return (
          `<tr>` +
          `<td><strong>${escapeHtml(s.stage_name)}</strong></td>` +
          `<td class="num">${s.duration_ms.toFixed(2)}</td>` +
          `<td class="num">${s.pct_of_total.toFixed(1)}%</td>` +
          `<td class="mono">${escapeHtml(itemsStr)}</td>` +
          `<td class="mono">${escapeHtml(tpStr)}</td>` +
          `<td>${escapeHtml(s.notes)}</td>` +
          `</tr>`
        );
      })
      .join("");
    el("currentTimingBody").innerHTML =
      rowsHtml +
      `<tr style="background-color:#ece9d8;font-weight:700;">` +
      `<td>Total End-to-End</td><td class="num">${tim.total_ms.toFixed(2)}</td><td class="num">100.0%</td>` +
      `<td class="mono">${tim.n_sequences} × ${fmtInt(tim.genome_length_bp)} bp</td>` +
      `<td class="mono">${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s</td>` +
      `<td>W=${tim.window_size}, M_tx=${fmtInt(tim.n_transformed_cols)}</td></tr>`;
  }

  async function loadTimingHistory() {
    const res = await fetch("/api/timings");
    const data = await res.json();
    state.timingHistory = data.records || [];
    const tbody = el("historyTimingBody");
    if (state.timingHistory.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12">No live timing runs recorded yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = state.timingHistory
      .map((r) => {
        const byId = {};
        (r.stages || []).forEach((s) => {
          byId[s.stage_id] = s.duration_ms;
        });
        return (
          `<tr>` +
          `<td class="mono">${escapeHtml(r.run_id)}</td>` +
          `<td class="mono">${escapeHtml(r.dataset_name)}</td>` +
          `<td class="mono">${escapeHtml(r.checkpoint)}</td>` +
          `<td class="mono">${escapeHtml(r.scoring_mode)}</td>` +
          `<td class="mono">${r.n_sequences}×${fmtInt(r.genome_length_bp)}</td>` +
          `<td class="num">${fmtInt(r.n_transformed_cols)}</td>` +
          `<td class="num">${(byId.parse_filter ?? 0).toFixed(1)}</td>` +
          `<td class="num">${(byId.pairwise_embed ?? 0).toFixed(1)}</td>` +
          `<td class="num">${(byId.transformer_fwd ?? 0).toFixed(1)}</td>` +
          `<td class="num">${(byId.flank_kl_null ?? 0).toFixed(1)}</td>` +
          `<td class="num"><strong>${r.total_ms.toFixed(1)}</strong></td>` +
          `<td class="num">${fmtInt(Math.round(r.throughput_bp_per_sec))}</td>` +
          `</tr>`
        );
      })
      .join("");
  }

  async function loadBenchmarkSummaries() {
    try {
      const res = await fetch("/api/benchmarks");
      const data = await res.json();
      renderSummaryCsvTable("scalingBenchmarkHead", "scalingBenchmarkBody", data.scaling_table);
      renderSummaryCsvTable("extremeBenchmarkHead", "extremeBenchmarkBody", data.out_of_range_table);
    } catch (err) {
      console.error("Failed to load benchmarks:", err);
    }
  }

  function renderSummaryCsvTable(headId, bodyId, tableObj) {
    const thead = el(headId);
    const tbody = el(bodyId);
    if (!tableObj || !tableObj.headers || tableObj.headers.length === 0) return;
    thead.innerHTML = `<tr>${tableObj.headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
    tbody.innerHTML = tableObj.rows
      .map((row) => {
        if (row[0] && row[0].startsWith("===")) {
          return `<tr style="background-color:#ece9d8;font-weight:700;"><td colspan="${tableObj.headers.length}">${escapeHtml(row[0])}</td></tr>`;
        }
        return `<tr>${row.map((cell, i) => `<td${i > 0 ? ' class="mono"' : ""}>${escapeHtml(cell)}</td>`).join("")}</tr>`;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // Canvas 1: Graph 1: Top Genomic Overview Strip
  // -------------------------------------------------------------------------
  function drawOverviewStrip(run) {
    const canvas = el("overviewStripCanvas");
    const { ctx, width: w, height: h } = setupHiDpiCanvas(canvas, 760, 30);

    ctx.fillStyle = "#fdf2f8";
    ctx.fillRect(0, 0, w, h);

    const prof = run.track_profile;
    const L = Math.max(1, run.alignment.genome_length_bp);
    const dens = prof && prof.informative_density ? prof.informative_density : [];
    const pos = prof && prof.positions_orig_bp ? prof.positions_orig_bp : [];

    for (let i = 0; i < dens.length; i++) {
      const x = (pos[i] / L) * w;
      const bw = Math.max(2, w / Math.max(1, dens.length));
      const d = Math.min(1, Math.max(0, dens[i]));
      const alpha = 0.16 + 0.82 * d;
      ctx.fillStyle = `rgba(180, 0, 140, ${alpha.toFixed(2)})`;
      ctx.fillRect(x, 0, bw, h - 10);
    }

    // Bottom horizontal bracket bar for predicted events
    ctx.fillStyle = "#f4f3ee";
    ctx.fillRect(0, h - 10, w, 10);
    ctx.strokeStyle = "#808080";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h - 10);
    ctx.lineTo(w, h - 10);
    ctx.stroke();

    (run.consolidated_events || []).forEach((ev) => {
      const x1 = (ev.span_orig_start / L) * w;
      const x2 = Math.max(x1 + 4, (ev.span_orig_end / L) * w);
      const isSel = ev.event_id === state.selectedEventId;
      ctx.fillStyle = isSel ? "#0f172a" : "#1d4ed8";
      ctx.fillRect(x1, h - 8, x2 - x1, 6);
    });

    if (state.plotViewRange && state.plotViewRange.endBp > state.plotViewRange.startBp) {
      const zx1 = Math.max(0, (state.plotViewRange.startBp / L) * w);
      const zx2 = Math.min(w, (state.plotViewRange.endBp / L) * w);
      if (zx2 > zx1) {
        ctx.fillStyle = "rgba(29, 78, 137, 0.18)";
        ctx.fillRect(zx1, 1, Math.max(4, zx2 - zx1), h - 2);
        ctx.strokeStyle = "#1d4e89";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(zx1, 1, Math.max(4, zx2 - zx1), h - 2);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Canvas 2: Graph 2: Bottom-Left Breakpoint Scan Plot
  // -------------------------------------------------------------------------
  function getPlotViewBounds(run) {
    const L = Math.max(1, run.alignment.genome_length_bp);
    if (state.plotViewRange && state.plotViewRange.endBp > state.plotViewRange.startBp) {
      const start = Math.max(0, Math.min(L - 1, state.plotViewRange.startBp));
      const end = Math.max(start + 1, Math.min(L, state.plotViewRange.endBp));
      return { start, end, span: end - start, L };
    }
    return { start: 0, end: L, span: L, L };
  }

  function drawGenomeTrack(run) {
    const canvas = el("genomeTrackCanvas");
    const parentRect = canvas.parentElement ? canvas.parentElement.getBoundingClientRect() : null;
    const targetH = parentRect && parentRect.height > 180 ? Math.floor(parentRect.height) : 235;
    const { ctx, width, height } = setupHiDpiCanvas(canvas, 600, targetH);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const { start, end, span, L } = getPlotViewBounds(run);
    const padL = 50;
    const padR = 46;
    const plotTop = 36;
    const plotBot = height - 28;
    const plotW = Math.max(100, width - padL - padR);
    const plotH = Math.max(80, plotBot - plotTop);
    const xOfBp = (bp) => padL + ((bp - start) / span) * plotW;

    // Plot background & frame
    ctx.fillStyle = "#fafbfc";
    ctx.fillRect(padL, plotTop, plotW, plotH);

    // Selected or dragging region highlight
    const activeSel = state.isDraggingPlot && state.dragStartBp != null && state.dragCurBp != null
      ? { startBp: Math.min(state.dragStartBp, state.dragCurBp), endBp: Math.max(state.dragStartBp, state.dragCurBp) }
      : state.selectedRegion;
    if (activeSel && activeSel.endBp > activeSel.startBp) {
      const sx1 = Math.max(padL, xOfBp(activeSel.startBp));
      const sx2 = Math.min(padL + plotW, xOfBp(activeSel.endBp));
      if (sx2 > sx1) {
        ctx.fillStyle = "rgba(37, 99, 235, 0.14)";
        ctx.fillRect(sx1, plotTop, Math.max(2, sx2 - sx1), plotH);
        ctx.strokeStyle = "rgba(37, 99, 235, 0.8)";
        ctx.lineWidth = 1;
        ctx.strokeRect(sx1, plotTop, Math.max(2, sx2 - sx1), plotH);
      }
    }

    // Top GT / Pred tick lane
    ctx.fillStyle = "#334155";
    ctx.font = "bold 9.5px Consolas, monospace";
    ctx.textAlign = "right";
    ctx.fillText("GT", padL - 6, 12);
    ctx.fillText("PRED", padL - 6, 25);

    state.canvasMarkers = [];
    const gtItems = (run.diagnostics && run.diagnostics.ground_truth_items) || [];
    gtItems.forEach((g) => {
      if (g.gt_bp_orig < start || g.gt_bp_orig > end) return;
      const gx = xOfBp(g.gt_bp_orig);
      ctx.strokeStyle = g.detected ? "#16a34a" : "#dc2626";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(gx, 3);
      ctx.lineTo(gx, 14);
      ctx.stroke();
      state.canvasMarkers.push({
        kind: "gt",
        x: gx,
        y: 9,
        bp: g.gt_bp_orig,
        label: `GT ${fmtInt(g.gt_bp_orig)} bp (${g.detected ? "TP" : "FN"})`,
        detail: g.failure_hypothesis,
      });
    });

    const prof = run.track_profile;
    const maxZ = prof && prof.flank_kl_z && prof.flank_kl_z.length
      ? Math.max(10.0, (prof.threshold_z || 6.0) * 1.35, ...prof.flank_kl_z)
      : 10.0;
    const yOfZ = (z) => plotBot - Math.min(1.0, Math.max(0.0, z / maxZ)) * (plotH - 6);

    // Left axis (Flanking KL Z-Score) & Right axis (Transformer P(bp)) gridlines
    ctx.font = "9.5px Consolas, monospace";
    [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {
      const y = plotBot - f * (plotH - 6);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();

      ctx.fillStyle = "#0f172a";
      ctx.textAlign = "right";
      ctx.fillText((f * maxZ).toFixed(1), padL - 5, y + 3);

      ctx.fillStyle = "#7e22ce";
      ctx.textAlign = "left";
      ctx.fillText(f.toFixed(2), padL + plotW + 5, y + 3);
    });

    // Axis titles on left & right
    ctx.save();
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 9.5px sans-serif";
    ctx.translate(12, plotTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Flanking KL Z", 0, 0);
    ctx.restore();

    ctx.save();
    ctx.fillStyle = "#7e22ce";
    ctx.font = "bold 9.5px sans-serif";
    ctx.translate(width - 10, plotTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Transformer P(bp)", 0, 0);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, plotTop, plotW, plotH);
    ctx.clip();

    if (prof && prof.positions_orig_bp && prof.positions_orig_bp.length > 0) {
      // Shaded permutation-null band [0 .. threshold_z]
      const yBandTop = yOfZ(Math.min(maxZ, prof.threshold_z));
      ctx.fillStyle = "rgba(148, 163, 184, 0.16)";
      ctx.fillRect(padL, yBandTop, plotW, plotBot - yBandTop);

      // Dashed threshold line
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 1.1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, yBandTop);
      ctx.lineTo(padL + plotW, yBandTop);
      ctx.stroke();
      ctx.setLineDash([]);

      // Filled area + sharp black curve for flank_kl z-score
      ctx.fillStyle = "rgba(30, 41, 59, 0.18)";
      ctx.beginPath();
      ctx.moveTo(xOfBp(prof.positions_orig_bp[0]), plotBot);
      prof.positions_orig_bp.forEach((pos, idx) => {
        ctx.lineTo(xOfBp(pos), yOfZ(prof.flank_kl_z[idx]));
      });
      ctx.lineTo(xOfBp(prof.positions_orig_bp[prof.positions_orig_bp.length - 1]), plotBot);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      prof.positions_orig_bp.forEach((pos, idx) => {
        const x = xOfBp(pos);
        const y = yOfZ(prof.flank_kl_z[idx]);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Secondary Transformer P(bp) trace
      ctx.strokeStyle = "#7e22ce";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      prof.positions_orig_bp.forEach((pos, idx) => {
        const x = xOfBp(pos);
        const y = plotBot - Math.min(1.0, Math.max(0.0, prof.tf_prob[idx])) * (plotH - 6);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Event vertical guide lines inside plot
    const evs = run.consolidated_events || [];
    evs.forEach((ev) => {
      if (ev.most_plausible_bp_orig < start || ev.most_plausible_bp_orig > end) return;
      const px = xOfBp(ev.most_plausible_bp_orig);
      const isSel = ev.event_id === state.selectedEventId;
      ctx.save();
      ctx.strokeStyle = isSel ? "rgba(29, 78, 216, 0.75)" : "rgba(59, 130, 246, 0.30)";
      ctx.lineWidth = isSel ? 1.6 : 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, plotTop);
      ctx.lineTo(px, plotBot);
      ctx.stroke();
      ctx.restore();
    });

    // Live vertical hover crosshair
    if (state.genomeHoverBp !== null && state.genomeHoverBp >= start && state.genomeHoverBp <= end) {
      const hx = xOfBp(state.genomeHoverBp);
      ctx.strokeStyle = "rgba(15, 23, 42, 0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hx, plotTop);
      ctx.lineTo(hx, plotBot);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    // Plot outer border
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, plotTop, plotW, plotH);

    // Event flags + labels in top lane
    let lastTagRight = -999;
    [...evs].sort((a, b) => a.most_plausible_bp_orig - b.most_plausible_bp_orig).forEach((ev) => {
      if (ev.most_plausible_bp_orig < start || ev.most_plausible_bp_orig > end) return;
      const px = xOfBp(ev.most_plausible_bp_orig);
      const isTp = ev.match_status === "TP";
      const isSel = ev.event_id === state.selectedEventId;
      ctx.fillStyle = isTp ? "#1d4ed8" : ev.match_status === "FP" ? "#dc2626" : "#1e3a8a";
      ctx.fillRect(px - 2.5, 16, 5, 13);
      if (isSel) {
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 1.4;
        ctx.strokeRect(px - 4.5, 15, 9, 15);
      }
      const tag = `E${ev.event_id}`;
      ctx.font = isSel ? "bold 9px Consolas, monospace" : "9px Consolas, monospace";
      const tw = ctx.measureText(tag).width + 6;
      if (px - tw / 2 > lastTagRight + 2) {
        ctx.fillStyle = isSel ? "#1d4ed8" : "#475569";
        ctx.textAlign = "center";
        ctx.fillText(tag, Math.max(padL + 10, Math.min(padL + plotW - 10, px)), 33);
        lastTagRight = px + tw / 2;
      }
      state.canvasMarkers.push({
        kind: "pred",
        eventId: ev.event_id,
        x: px,
        y: 22,
        bp: ev.most_plausible_bp_orig,
        label: `Event #${ev.event_id} @ ${fmtInt(ev.most_plausible_bp_orig)} bp (${ev.match_status})`,
        detail: `KL z=${ev.score_z.toFixed(2)}, TF P=${ev.tf_prob.toFixed(3)}`,
      });
    });

    // X-axis ticks (Position in alignment)
    ctx.fillStyle = "#111827";
    ctx.strokeStyle = "#334155";
    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "center";
    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {
      const bp = Math.round(start + (span * i) / nTicks);
      const tx = xOfBp(bp);
      ctx.beginPath();
      ctx.moveTo(tx, plotBot);
      ctx.lineTo(tx, plotBot + 4);
      ctx.stroke();
      ctx.fillText(fmtInt(bp), tx, plotBot + 14);
    }
    ctx.font = "bold 10px sans-serif";
    const axisLbl = state.plotViewRange
      ? `Position in Alignment (bp) [Zoomed: ${fmtInt(start)}–${fmtInt(end)} bp · Double-click to reset]`
      : "Position in Alignment (bp)";
    ctx.fillText(axisLbl, padL + plotW / 2, height - 3);
  }

  function handleCanvasHover(e) {
    if (!state.currentRun) return;
    const canvas = el("genomeTrackCanvas");
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Update cursor readout box (bp : z)
    const padL = 50;
    const padR = 46;
    const plotW = Math.max(1, rect.width - padL - padR);
    const { start, span, L } = getPlotViewBounds(state.currentRun);
    const bpAtCursor = Math.max(0, Math.min(L, Math.round(start + ((mx - padL) / plotW) * span)));
    const prof = state.currentRun.track_profile;
    let zAtCursor = 0;
    let pAtCursor = 0;
    if (prof && prof.positions_orig_bp && prof.positions_orig_bp.length > 0) {
      let bestI = 0;
      let bestD = Infinity;
      prof.positions_orig_bp.forEach((p, idx) => {
        const d = Math.abs(p - bpAtCursor);
        if (d < bestD) {
          bestD = d;
          bestI = idx;
        }
      });
      zAtCursor = prof.flank_kl_z[bestI] || 0;
      pAtCursor = prof.tf_prob ? (prof.tf_prob[bestI] || 0) : 0;
    }
    el("cursorReadoutBox").textContent = `${bpAtCursor} bp : z=${zAtCursor.toFixed(2)} | P=${pAtCursor.toFixed(2)}`;

    const hit = state.canvasMarkers.find((m) => Math.abs(m.x - mx) <= 10 && Math.abs(m.y - my) <= 16);
    const tip = el("trackTooltip");
    if (hit) {
      tip.innerHTML = `<strong>${escapeHtml(hit.label)}</strong><br/>${escapeHtml(hit.detail)}`;
      tip.style.left = `${Math.min(rect.width - 240, Math.max(6, mx + 10))}px`;
      tip.style.top = `${Math.max(6, my + 12)}px`;
      tip.classList.remove("hidden");
    } else {
      tip.classList.add("hidden");
    }
  }

  function handleCanvasClick(e) {
    if (!state.currentRun) return;
    const canvas = el("genomeTrackCanvas");
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const hit = state.canvasMarkers.find(
      (m) => m.kind === "pred" && Math.abs(m.x - mx) <= 12 && Math.abs(m.y - my) <= 18
    );
    if (hit && hit.eventId) {
      selectEvent(hit.eventId, true);
    }
  }

  // -------------------------------------------------------------------------
  // Canvas 3: Graph 3: Bottom-Right Recombinant Mosaic Tract Map
  // -------------------------------------------------------------------------
  function colorForParent(idx) {
    return PALETTE[Math.abs(Number(idx)) % PALETTE.length];
  }

  function drawMosaicMap(run) {
    const canvas = el("mosaicMapCanvas");
    const L = Math.max(1, run.alignment.genome_length_bp);

    const selectedOnly = el("chkMosaicSelectedOnly").checked;
    let evs = run.consolidated_events || [];
    if (selectedOnly && state.selectedEventId !== null) {
      evs = evs.filter((x) => x.event_id === state.selectedEventId);
    }

    const bars = [];
    evs.forEach((ev) => {
      (ev.recombinants || []).slice(0, 4).forEach((r) => {
        bars.push({ ev, r });
      });
    });

    el("mosaicLegendSummary").textContent = `${bars.length} carrier tract(s) shown`;
    const rowH = 30;
    const headerH = 26;
    const targetHeight = Math.max(205, headerH + bars.length * rowH + 14);
    const { ctx, width, height } = setupHiDpiCanvas(canvas, 560, targetHeight);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 14;
    const padR = 16;
    const plotW = Math.max(100, width - padL - padR);
    const xOfBp = (bp) => padL + (Math.max(0, Math.min(L, bp)) / L) * plotW;

    // Top genomic coordinate ruler
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(padL, 2, plotW, headerH - 6);
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, headerH - 4);
    ctx.lineTo(padL + plotW, headerH - 4);
    ctx.stroke();

    ctx.font = "9.5px Consolas, monospace";
    ctx.fillStyle = "#475569";
    for (let i = 0; i <= 4; i++) {
      const bp = Math.round((L * i) / 4);
      const tx = xOfBp(bp);
      ctx.strokeStyle = "#e2e8f0";
      ctx.beginPath();
      ctx.moveTo(tx, headerH - 4);
      ctx.lineTo(tx, height - 6);
      ctx.stroke();
      ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
      ctx.fillText(`${fmtInt(bp)} bp`, tx, 15);
    }

    if (bars.length === 0) {
      state.mosaicRowMarkers = [];
      ctx.fillStyle = "#64748b";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No recombinant carrier tracts detected in this run.", width / 2, height / 2);
      return;
    }

    state.mosaicRowMarkers = [];
    bars.forEach(({ ev, r }, idx) => {
      const y = headerH + idx * rowH;
      const bp = ev.most_plausible_bp_orig;
      const xBp = xOfBp(bp);
      const xSpanL = xOfBp(ev.span_orig_start);
      const xSpanR = xOfBp(ev.span_orig_end);
      const colL = colorForParent(r.ext_parent_left);
      const colR = colorForParent(r.ext_parent_right);
      const pLName = r.ext_parent_left_name || r.ext_parent_left;
      const pRName = r.ext_parent_right_name || r.ext_parent_right;
      const isSel = ev.event_id === state.selectedEventId;
      const isHover = idx === state.hoveredMosaicRowIdx;

      state.mosaicRowMarkers.push({
        eventId: ev.event_id,
        y1: y - 2,
        y2: y + rowH - 2,
        tipHtml: formatEventHoverHtml(ev, r),
      });
      if (isSel || isHover) {
        ctx.fillStyle = isSel ? "rgba(29, 78, 216, 0.10)" : "rgba(100, 116, 139, 0.08)";
        ctx.fillRect(padL - 4, y - 2, plotW + 8, rowH - 2);
        ctx.strokeStyle = isSel ? "#1d4ed8" : "#94a3b8";
        ctx.lineWidth = isSel ? 1.3 : 1;
        ctx.strokeRect(padL - 4, y - 2, plotW + 8, rowH - 2);
      }

      // Label above bar
      ctx.fillStyle = isSel ? "#0f172a" : "#334155";
      ctx.font = isSel ? "bold 10px Consolas, monospace" : "10px Consolas, monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        `E${ev.event_id} · ${r.seq_name.slice(0, 20)} | L: ${pLName} (${(r.left_sim * 100).toFixed(1)}%) → R: ${pRName} (${(r.right_sim * 100).toFixed(1)}%) @ ${fmtInt(bp)} bp`,
        padL,
        y + 9
      );

      // Left parental tract bar [0 .. bp]
      ctx.fillStyle = colL;
      ctx.fillRect(padL, y + 13, Math.max(2, xBp - padL), 11);

      // Right parental tract bar [bp .. L]
      ctx.fillStyle = colR;
      ctx.fillRect(xBp, y + 13, Math.max(2, padL + plotW - xBp), 11);

      // Uncertainty span box
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1;
      ctx.strokeRect(xSpanL, y + 12, Math.max(3, xSpanR - xSpanL), 13);

      // Breakpoint diamond & divider tick
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(xBp, y + 11);
      ctx.lineTo(xBp, y + 26);
      ctx.stroke();
    });
  }

  // -------------------------------------------------------------------------
  // Annotations & Exports
  // -------------------------------------------------------------------------
  async function loadAnnotations() {
    const res = await fetch("/api/annotations");
    const data = await res.json();
    state.annotations = data.annotations || [];
    el("annotationCountBadge").textContent = String(state.annotations.length);
    const tbody = el("annotationsBody");
    if (state.annotations.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6">No notes saved yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = state.annotations
      .map(
        (a) =>
          `<tr>` +
          `<td class="mono">${escapeHtml((a.created_at_iso || "").slice(0, 19).replace("T", " "))}</td>` +
          `<td class="mono">${escapeHtml(a.dataset_name)}</td>` +
          `<td class="mono">${escapeHtml(a.target_id)}</td>` +
          `<td class="mono">${escapeHtml(a.verdict)}</td>` +
          `<td>${escapeHtml(a.reviewer_name)}</td>` +
          `<td>${escapeHtml(a.comment)}</td>` +
          `</tr>`
      )
      .join("");
  }

  function openAnnotationModal(targetType, targetId) {
    el("annTargetType").value = targetType || "predicted_event";
    el("annTargetId").value = targetId || "";
    el("annComment").value = "";
    el("annotationModal").classList.remove("hidden");
  }

  function closeAnnotationModal() {
    el("annotationModal").classList.add("hidden");
  }

  async function handleSaveAnnotation(e) {
    e.preventDefault();
    const run = state.currentRun;
    const payload = {
      dataset_name: run ? run.alignment.dataset_name : "unknown",
      run_id: run ? run.timings.run_id : "manual",
      target_type: el("annTargetType").value,
      target_id: el("annTargetId").value.trim(),
      verdict: el("annVerdict").value,
      reviewer_name: "Darren Martin",
      comment: el("annComment").value.trim(),
    };
    const res = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      closeAnnotationModal();
      await loadAnnotations();
      switchTab("tab-notes");
    }
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function buildRdpProjectText(run) {
    const aln = run.alignment || {};
    const cfg = run.config || {};
    const tim = run.timings || {};
    const prof = run.track_profile || {};
    const projectObj = {
      format: "gt-transformer-rdp-project",
      version: "2.0",
      exported_at: new Date().toISOString(),
      software: "gt-transformer",
      run_id: tim.run_id || null,
      alignment: {
        dataset_name: aln.dataset_name,
        n_sequences: aln.n_sequences,
        genome_length_bp: aln.genome_length_bp,
        n_informative_cols: aln.n_informative_cols,
        n_transformed_cols: aln.n_transformed_cols,
        gc_content_pct: aln.gc_content_pct,
        edge_dead_zone_bp: aln.edge_dead_zone_bp,
        sequence_names: aln.sequence_names || [],
      },
      parameters: cfg,
      timings: tim,
      warnings: run.warnings || [],
      threshold_z: prof.threshold_z ?? cfg.z_thresh ?? 6.0,
      events: (run.consolidated_events || []).map((ev) => ({
        event_id: ev.event_id,
        ancestral_event_id: ev.ancestral_event_id || ev.event_id,
        breakpoint_bp: ev.most_plausible_bp_orig,
        confidence_span_bp: [ev.span_orig_start, ev.span_orig_end],
        n_peaks_consolidated: ev.n_peaks_consolidated,
        flank_kl_z: ev.score_z,
        transformer_logit: ev.tf_logit,
        transformer_prob: ev.tf_prob,
        signal_attribution: ev.signal_attribution,
        match_status: ev.match_status,
        recombinants: (ev.recombinants || []).map((r) => ({
          seq_idx: r.seq_idx,
          seq_name: r.seq_name,
          p_switch: r.p_switch,
          left_parent_name: r.ext_parent_left_name || String(r.ext_parent_left),
          right_parent_name: r.ext_parent_right_name || String(r.ext_parent_right),
          left_similarity: r.left_sim,
          right_similarity: r.right_sim,
        })),
        parent_candidates: (ev.parent_candidates || []).slice(0, 10),
      })),
      scan_track: {
        positions_orig_bp: prof.positions_orig_bp || [],
        flank_kl_z: prof.flank_kl_z || [],
        tf_prob: prof.tf_prob || [],
      },
    };
    return JSON.stringify(projectObj, null, 2);
  }

  function exportRdpProject() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name || "alignment";
    downloadBlob(`${ds}.rdp`, buildRdpProjectText(state.currentRun), "application/octet-stream");
  }

  function buildGff3Text(run) {
    const aln = run.alignment || {};
    const seqId = (aln.dataset_name || "alignment").replace(/\s+/g, "_");
    const L = Math.max(1, aln.genome_length_bp || 1);
    const lines = [
      "##gff-version 3",
      `##sequence-region ${seqId} 1 ${L}`,
    ];
    (run.consolidated_events || []).forEach((ev) => {
      const s = Math.max(1, Math.min(L, Math.min(ev.span_orig_start || ev.most_plausible_bp_orig, ev.most_plausible_bp_orig)));
      const e = Math.max(s, Math.min(L, Math.max(ev.span_orig_end || ev.most_plausible_bp_orig, ev.most_plausible_bp_orig)));
      const r0 = ev.recombinants && ev.recombinants[0];
      const recStr = (ev.recombinants || []).map((r) => r.seq_name).join(",") || "unknown";
      const parStr = r0 ? `${r0.ext_parent_left_name || r0.ext_parent_left}->${r0.ext_parent_right_name || r0.ext_parent_right}` : "unknown";
      const attrs = [
        `ID=E${ev.event_id}`,
        `Name=Breakpoint_E${ev.event_id}_at_${ev.most_plausible_bp_orig}bp`,
        `ancestral_group=A${ev.ancestral_event_id || ev.event_id}`,
        `peak_bp=${ev.most_plausible_bp_orig}`,
        `kl_z=${ev.score_z.toFixed(4)}`,
        `tf_prob=${ev.tf_prob.toFixed(4)}`,
        `attribution=${encodeURIComponent(ev.signal_attribution || "hybrid")}`,
        `carriers=${encodeURIComponent(recStr)}`,
        `parents=${encodeURIComponent(parStr)}`,
      ].join(";");
      lines.push([seqId, "gt-transformer", "recombination_breakpoint", s, e, ev.score_z.toFixed(2), ".", ".", attrs].join("\t"));
    });
    return lines.join("\n") + "\n";
  }

  function exportGff3Track() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name || "alignment";
    downloadBlob(`${ds}_breakpoints.gff3`, buildGff3Text(state.currentRun), "text/plain");
  }

  function buildBedText(run) {
    const aln = run.alignment || {};
    const seqId = (aln.dataset_name || "alignment").replace(/\s+/g, "_");
    const L = Math.max(1, aln.genome_length_bp || 1);
    const lines = [
      `track name="gt-transformer_${seqId}" description="Detected recombination breakpoint spans (${run.config.checkpoint})" useScore=1`,
    ];
    (run.consolidated_events || []).forEach((ev) => {
      const start0 = Math.max(0, Math.min(L - 1, (ev.span_orig_start || ev.most_plausible_bp_orig) - 1));
      const end1 = Math.max(start0 + 1, Math.min(L, ev.span_orig_end || ev.most_plausible_bp_orig));
      const r0 = ev.recombinants && ev.recombinants[0];
      const name = `E${ev.event_id}_bp${ev.most_plausible_bp_orig}_${r0 ? r0.seq_name : "rec"}`.replace(/\s+/g, "_");
      const score = Math.min(1000, Math.max(100, Math.round(ev.tf_prob * 600 + Math.min(20, ev.score_z) * 20)));
      lines.push([seqId, start0, end1, name, score, "."].join("\t"));
    });
    return lines.join("\n") + "\n";
  }

  function exportBedTrack() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name || "alignment";
    downloadBlob(`${ds}_breakpoints.bed`, buildBedText(state.currentRun), "text/plain");
  }

  function parseFastaTextClient(raw) {
    const seqs = new Map();
    let curName = null;
    let buf = [];
    String(raw || "").split(/\r?\n/).forEach((line) => {
      const t = line.trim();
      if (!t) return;
      if (t.startsWith(">")) {
        if (curName !== null) seqs.set(curName, buf.join(""));
        curName = t.slice(1).trim().split(/\s+/)[0];
        buf = [];
      } else {
        buf.push(t.replace(/\s+/g, ""));
      }
    });
    if (curName !== null) seqs.set(curName, buf.join(""));
    return seqs;
  }

  async function fetchFullSequencesForRun(run) {
    if (state.uploadedFastaFile) {
      const txt = await state.uploadedFastaFile.text();
      const map = parseFastaTextClient(txt);
      if (map.size > 0) return map;
    } else if (state.uploadedPastedFasta) {
      const map = parseFastaTextClient(state.uploadedPastedFasta);
      if (map.size > 0) return map;
    }
    const L = Math.max(1, run.alignment.genome_length_bp || 1);
    const runId = run.timings.run_id;
    const chunks = new Map();
    for (let s = 0; s < L; s += 300) {
      const w = Math.min(300, L - s);
      const res = await fetch(`/api/alignment/window?run_id=${encodeURIComponent(runId)}&start=${s}&width=${w}`);
      if (!res.ok) break;
      const win = await res.json();
      (win.sequences || []).forEach((row) => {
        const key = String(row.name);
        if (!chunks.has(key)) chunks.set(key, "");
        chunks.set(key, chunks.get(key) + (row.slice || ""));
      });
    }
    return chunks;
  }

  async function exportRecombinantFasta() {
    const run = state.currentRun;
    if (!run) return;
    const ds = run.alignment.dataset_name || "alignment";
    const evs = run.consolidated_events || [];
    if (!evs.length) {
      showGlobalError("No recombinant events detected in this run to export as recombinant FASTA tracts.");
      return;
    }
    const seqMap = await fetchFullSequencesForRun(run);
    const L = run.alignment.genome_length_bp || 1;
    const out = [];
    const seen = new Set();
    evs.forEach((ev) => {
      const bp = Math.max(1, Math.min(L - 1, ev.most_plausible_bp_orig));
      (ev.recombinants || []).slice(0, 3).forEach((r) => {
        const seq = seqMap.get(r.seq_name) || "";
        if (!seq) return;
        const pLName = String(r.ext_parent_left_name || r.ext_parent_left).split(" or ")[0];
        const pRName = String(r.ext_parent_right_name || r.ext_parent_right).split(" or ")[0];
        const key = `${ev.event_id}:${r.seq_name}`;
        if (seen.has(key)) return;
        seen.add(key);
        out.push(`>E${ev.event_id}_${r.seq_name}_FULL|bp=${bp}|parents=${pLName}->${pRName}`);
        out.push(seq);
        out.push(`>E${ev.event_id}_${r.seq_name}_LEFT_1_to_${bp}|parent=${pLName}|sim=${(r.left_sim * 100).toFixed(1)}%`);
        out.push(seq.slice(0, bp));
        out.push(`>E${ev.event_id}_${r.seq_name}_RIGHT_${bp + 1}_to_${seq.length}|parent=${pRName}|sim=${(r.right_sim * 100).toFixed(1)}%`);
        out.push(seq.slice(bp));
        const seqL = seqMap.get(pLName);
        if (seqL) {
          out.push(`>E${ev.event_id}_PARENT_LEFT_${pLName}_1_to_${bp}`);
          out.push(seqL.slice(0, bp));
        }
        const seqR = seqMap.get(pRName);
        if (seqR) {
          out.push(`>E${ev.event_id}_PARENT_RIGHT_${pRName}_${bp + 1}_to_${seqR.length}`);
          out.push(seqR.slice(bp));
        }
      });
    });
    if (!out.length) {
      showGlobalError("Could not extract recombinant FASTA sequences for this run.");
      return;
    }
    downloadBlob(`${ds}_recombinant_tracts.fasta`, out.join("\n") + "\n", "text/plain");
  }

  function buildOpenRdpTsvText(run) {
    const mode = run.config.scoring_mode;
    const lines = ["Method\tEventID\tAncestralID\tStart\tEnd\tBreakpoint\tRecombinant\tParent1\tParent2\tFlankKL_Z\tTF_Logit"];
    (run.consolidated_events || []).forEach((ev) => {
      const r = ev.recombinants && ev.recombinants[0];
      lines.push(
        [
          mode,
          `E${ev.event_id}`,
          `A${ev.ancestral_event_id || ev.event_id}`,
          ev.span_orig_start,
          ev.span_orig_end,
          ev.most_plausible_bp_orig,
          r ? r.seq_name : "-",
          r ? r.ext_parent_left_name : "-",
          r ? r.ext_parent_right_name : "-",
          ev.score_z.toFixed(4),
          ev.tf_logit.toFixed(4),
        ].join("\t")
      );
    });
    return lines.join("\n");
  }

  function exportOpenRdpTsv() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    downloadBlob(`${ds}_summary.tsv`, buildOpenRdpTsvText(state.currentRun), "text/tab-separated-values");
  }

  function buildBreakpointsCsvText(run) {
    const header = [
      "event_id",
      "ancestral_event_id",
      "most_plausible_bp_orig",
      "span_orig_start",
      "span_orig_end",
      "n_peaks_consolidated",
      "score_z",
      "tf_logit",
      "tf_prob",
      "boundary_tx",
      "n_switch_recombinants",
      "recombinant_sequences",
      "match_status",
      "matched_gt_bp",
      "bp_error",
    ];
    const rows = (run.consolidated_events || []).map((ev) => [
      ev.event_id,
      ev.ancestral_event_id || ev.event_id,
      ev.most_plausible_bp_orig,
      ev.span_orig_start,
      ev.span_orig_end,
      ev.n_peaks_consolidated,
      ev.score_z.toFixed(4),
      ev.tf_logit.toFixed(4),
      ev.tf_prob.toFixed(4),
      ev.boundary_tx,
      ev.n_switch_recombinants,
      ev.recombinant_sequences.join(";"),
      ev.match_status,
      ev.matched_gt_bp ?? "",
      ev.bp_error ?? "",
    ]);
    return [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
  }

  function exportBreakpointsCsv() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    downloadBlob(`${ds}_breakpoints_consolidated.csv`, buildBreakpointsCsvText(state.currentRun), "text/csv");
  }

  function buildSequenceCallsCsvText(run) {
    const header = [
      "event_id",
      "ancestral_event_id",
      "most_plausible_bp_orig",
      "span_orig_start",
      "span_orig_end",
      "seq_idx",
      "seq_name",
      "p_switch",
      "is_recombinant",
      "parent_left",
      "parent_right",
      "ext_parent_left",
      "ext_parent_right",
      "left_sim",
      "right_sim",
    ];
    const rows = [];
    (run.consolidated_events || []).forEach((ev) => {
      (ev.recombinants || []).forEach((r) => {
        rows.push([
          ev.event_id,
          ev.ancestral_event_id || ev.event_id,
          ev.most_plausible_bp_orig,
          ev.span_orig_start,
          ev.span_orig_end,
          r.seq_idx,
          `"${String(r.seq_name).replace(/"/g, '""')}"`,
          r.p_switch.toFixed(4),
          r.is_recombinant,
          r.parent_left,
          r.parent_right,
          r.ext_parent_left,
          r.ext_parent_right,
          r.left_sim.toFixed(4),
          r.right_sim.toFixed(4),
        ]);
      });
    });
    return [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
  }

  function exportSequenceCallsCsv() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    downloadBlob(`${ds}_sequence_calls_consolidated.csv`, buildSequenceCallsCsvText(state.currentRun), "text/csv");
  }

  function exportFullJson() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    downloadBlob(`${ds}_diagnostic_report.json`, JSON.stringify(state.currentRun, null, 2), "application/json");
  }

  function renderRegionAnalysis() {
    const bar = el("regionAnalysisBar");
    const txt = el("regionAnalysisText");
    if (!bar || !txt || !state.currentRun) return;
    const inspectBtn = el("btnInspectRegion");
    const zoomBtn = el("btnZoomRegionPlot");
    const resetBtn = el("btnResetPlotZoom");
    const alnBtn = el("btnZoomRegionAln");
    const askBtn = el("btnAskAiRegion");
    const clearBtn = el("btnClearRegion");

    const hasSel = Boolean(state.selectedRegion && state.selectedRegion.endBp > state.selectedRegion.startBp);
    const hasZoom = Boolean(state.plotViewRange && state.plotViewRange.endBp > state.plotViewRange.startBp);
    if (inspectBtn) inspectBtn.classList.toggle("hidden", !hasSel);
    if (zoomBtn) zoomBtn.classList.toggle("hidden", !hasSel);
    if (resetBtn) resetBtn.classList.toggle("hidden", !hasZoom);
    if (alnBtn) alnBtn.classList.toggle("hidden", !hasSel);
    if (askBtn) askBtn.classList.toggle("hidden", !hasSel);
    if (clearBtn) clearBtn.classList.toggle("hidden", !hasSel);

    if (!hasSel && !hasZoom) {
      bar.classList.add("hidden");
      return;
    }
    if (!hasSel && hasZoom) {
      txt.textContent = `Zoomed graph view: ${fmtInt(state.plotViewRange.startBp)}–${fmtInt(state.plotViewRange.endBp)} bp (drag to select or Reset Zoom)`;
      bar.classList.remove("hidden");
      return;
    }

    const { startBp, endBp } = state.selectedRegion;
    const prof = state.currentRun.track_profile;
    let maxZ = 0;
    let maxZBp = startBp;
    let maxP = 0;
    let maxPBp = startBp;
    if (prof && prof.positions_orig_bp) {
      prof.positions_orig_bp.forEach((bp, idx) => {
        if (bp >= startBp && bp <= endBp) {
          const z = prof.flank_kl_z[idx] || 0;
          const p = prof.tf_prob[idx] || 0;
          if (z > maxZ) { maxZ = z; maxZBp = bp; }
          if (p > maxP) { maxP = p; maxPBp = bp; }
        }
      });
    }
    const evsIn = (state.currentRun.consolidated_events || []).filter(
      (ev) => ev.most_plausible_bp_orig >= startBp && ev.most_plausible_bp_orig <= endBp
    );
    const evText = evsIn.length
      ? `${evsIn.length} event(s) [${evsIn.map((e) => `#${e.event_id}@${e.most_plausible_bp_orig}`).join(", ")}]`
      : "0 events";
    const zoomTag = hasZoom ? ` | Zoomed: ${fmtInt(state.plotViewRange.startBp)}–${fmtInt(state.plotViewRange.endBp)} bp` : "";
    txt.textContent = `Region ${fmtInt(startBp)}–${fmtInt(endBp)} bp (${fmtInt(endBp - startBp + 1)} bp) | ${evText} | Peak KL z=${maxZ.toFixed(2)} @ ${maxZBp} bp | Peak TF P=${maxP.toFixed(3)} @ ${maxPBp} bp${zoomTag}`;
    bar.classList.remove("hidden");
  }

  function drawEvidenceScatter(run) {
    const canvas = el("evidenceScatterCanvas");
    if (!canvas || !run) return;
    const { ctx, width, height } = setupHiDpiCanvas(canvas, 720, 275);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 56, padR = 24, padT = 22, padB = 40;
    const pw = Math.max(180, width - padL - padR);
    const ph = Math.max(120, height - padT - padB);
    const evs = run.consolidated_events || [];
    const thrZ = run.track_profile?.threshold_z || 6.0;
    const maxZ = Math.max(10, thrZ * 1.6, ...evs.map((e) => (e.score_z || 0) * 1.12));
    const yHalf = padT + ph * 0.5;
    const xThr = padL + Math.min(1, thrZ / maxZ) * pw;

    // Quadrant fills
    ctx.fillStyle = "rgba(22, 163, 74, 0.055)";
    ctx.fillRect(xThr, padT, padL + pw - xThr, yHalf - padT);
    ctx.fillStyle = "rgba(217, 119, 6, 0.055)";
    ctx.fillRect(xThr, yHalf, padL + pw - xThr, padT + ph - yHalf);
    ctx.fillStyle = "rgba(126, 34, 206, 0.045)";
    ctx.fillRect(padL, padT, xThr - padL, yHalf - padT);
    ctx.fillStyle = "rgba(148, 163, 184, 0.045)";
    ctx.fillRect(padL, yHalf, xThr - padL, padT + ph - yHalf);

    // Light gridlines
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((f) => {
      const yg = padT + f * ph;
      ctx.beginPath(); ctx.moveTo(padL, yg); ctx.lineTo(padL + pw, yg); ctx.stroke();
      const xg = padL + f * pw;
      ctx.beginPath(); ctx.moveTo(xg, padT); ctx.lineTo(xg, padT + ph); ctx.stroke();
    });

    // Quadrant labels inside corners (registered as label obstacles so points never overlap them)
    const placedBoxes = [];
    ctx.font = "bold 9.5px sans-serif";
    const qTopText = "Dual-Supported Consensus (KL + Transformer)";
    const qTopW = ctx.measureText(qTopText).width + 8;
    ctx.fillStyle = "rgba(21, 128, 61, 0.85)";
    ctx.textAlign = "right";
    ctx.fillText(qTopText, padL + pw - 6, padT + 13);
    placedBoxes.push({ x1: padL + pw - 6 - qTopW, y1: padT + 2, x2: padL + pw - 2, y2: padT + 18 });

    const qBotText = "KL-Driven / Short-Tract Regime";
    const qBotW = ctx.measureText(qBotText).width + 8;
    ctx.fillStyle = "rgba(180, 83, 9, 0.85)";
    ctx.fillText(qBotText, padL + pw - 6, padT + ph - 6);
    placedBoxes.push({ x1: padL + pw - 6 - qBotW, y1: padT + ph - 19, x2: padL + pw - 2, y2: padT + ph - 2 });

    if (xThr - padL > 105) {
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(107, 33, 168, 0.85)";
      ctx.fillText("Transformer Only", padL + 6, padT + 13);
      placedBoxes.push({ x1: padL + 4, y1: padT + 2, x2: padL + 100, y2: padT + 18 });
    }

    // Plot border
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, pw, ph);

    // Threshold dashed lines & clean top-margin badge (never collides with quadrant text)
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#64748b";
    ctx.beginPath(); ctx.moveTo(padL, yHalf); ctx.lineTo(padL + pw, yHalf); ctx.stroke();
    ctx.strokeStyle = "#dc2626";
    ctx.beginPath(); ctx.moveTo(xThr, padT); ctx.lineTo(xThr, padT + ph); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#dc2626";
    ctx.font = "bold 9px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`z=${thrZ.toFixed(1)}`, Math.max(padL + 22, Math.min(padL + pw - 22, xThr)), padT - 6);

    // Compact right-aligned summary in top margin
    ctx.fillStyle = "#475569";
    ctx.font = "9.5px Consolas, monospace";
    ctx.textAlign = "right";
    const dsShort = String(run.alignment.dataset_name || "").slice(0, 34);
    ctx.fillText(`${dsShort} | ${evs.length} event(s)`, padL + pw, padT - 6);

    // Axis ticks & labels
    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "right";
    [0, 0.25, 0.5, 0.75, 1.0].forEach((p) => {
      const y = padT + (1 - p) * ph;
      ctx.fillStyle = "#1e293b";
      ctx.fillText(p.toFixed(2), padL - 6, y + 3);
    });
    ctx.textAlign = "center";
    [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {
      const val = (f * maxZ).toFixed(0);
      ctx.fillStyle = "#1e293b";
      ctx.fillText(val, padL + f * pw, padT + ph + 15);
    });
    ctx.font = "bold 10.5px sans-serif";
    ctx.fillText("Flanking KL Z-Score (Permutation-Calibrated)", padL + pw / 2, height - 6);

    ctx.save();
    ctx.translate(15, padT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Transformer P(bp)", 0, 0);
    ctx.restore();

    state.scatterMarkers = [];
    if (!evs.length) {
      ctx.fillStyle = "#64748b";
      ctx.font = "11px sans-serif";
      ctx.fillText("No recombination events detected above threshold.", padL + pw / 2, padT + ph / 2);
      return;
    }

    const pts = evs.map((ev) => ({
      ev,
      x: padL + Math.min(1, Math.max(0, ev.score_z / maxZ)) * pw,
      y: padT + (1 - Math.min(1, Math.max(0, ev.tf_prob))) * ph,
    }));

    // Draw hover axis crosshair if hovered
    const hoverPt = pts.find((p) => p.ev.event_id === state.hoveredScatterEventId);
    if (hoverPt) {
      ctx.save();
      ctx.strokeStyle = "rgba(15, 23, 42, 0.45)";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverPt.x, padT); ctx.lineTo(hoverPt.x, padT + ph);
      ctx.moveTo(padL, hoverPt.y); ctx.lineTo(padL + pw, hoverPt.y);
      ctx.stroke();
      ctx.restore();
    }

    // Collision-avoiding label placement across a 24-slot radial search grid
    const pointObstacles = pts.map((p) => ({ x1: p.x - 8, y1: p.y - 8, x2: p.x + 8, y2: p.y + 8 }));
    const overlaps = (b1, b2) => !(b1.x2 < b2.x1 || b1.x1 > b2.x2 || b1.y2 < b2.y1 || b1.y1 > b2.y2);

    ctx.font = "bold 9.5px Consolas, monospace";
    const candidateOffsets = [
      [12, 0], [-12, 0], [12, -16], [12, 16], [-12, -16], [-12, 16],
      [0, -22], [0, 22], [14, -32], [14, 32], [-14, -32], [-14, 32],
      [28, -16], [28, 16], [-28, -16], [-28, 16],
      [0, -40], [0, 40], [36, -34], [36, 34], [-36, -34], [-36, 34],
      [50, 0], [-50, 0],
    ];

    // Draw points first so leader lines & badges layer cleanly
    pts.forEach((p) => {
      const { ev, x, y } = p;
      const isSel = ev.event_id === state.selectedEventId;
      const isHover = ev.event_id === state.hoveredScatterEventId;
      state.scatterMarkers.push({ eventId: ev.event_id, x, y, tipHtml: formatEventHoverHtml(ev) });

      if (isSel || isHover) {
        ctx.strokeStyle = isSel ? "#0f172a" : "#2563eb";
        ctx.lineWidth = isSel ? 2.2 : 1.8;
        ctx.beginPath();
        ctx.arc(x, y, 9.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = ev.tf_prob >= 0.5 && ev.score_z >= thrZ ? "#16a34a" : (ev.tf_prob >= 0.5 ? "#7e22ce" : "#d97706");
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    });

    // Place non-overlapping labels (try full "E1 (251bp)" first, then compact "E1" if dense)
    pts.forEach((p) => {
      const { ev, x, y } = p;
      const isSel = ev.event_id === state.selectedEventId;
      const fullLbl = `E${ev.event_id} (${ev.most_plausible_bp_orig}bp)`;
      const shortLbl = `E${ev.event_id}`;
      let chosen = null;
      let chosenLbl = fullLbl;
      let chosenW = ctx.measureText(fullLbl).width + 8;
      const lh = 15;

      for (const tryLbl of [fullLbl, shortLbl]) {
        const lw = ctx.measureText(tryLbl).width + 8;
        for (const [dx, dy] of candidateOffsets) {
          let bx = dx < 0 ? x + dx - lw : (dx === 0 ? x - lw / 2 : x + dx);
          let by = y + dy - lh / 2;
          bx = Math.max(padL + 3, Math.min(padL + pw - lw - 3, bx));
          by = Math.max(padT + 3, Math.min(padT + ph - lh - 3, by));
          const box = { x1: bx - 1, y1: by - 1, x2: bx + lw + 1, y2: by + lh + 1 };
          const hitsLabel = placedBoxes.some((b) => overlaps(box, b));
          const hitsPt = pointObstacles.some((b) => overlaps(box, b));
          if (!hitsLabel && !hitsPt) {
            chosen = { x1: bx, y1: by, x2: bx + lw, y2: by + lh };
            chosenLbl = tryLbl;
            chosenW = lw;
            break;
          }
        }
        if (chosen) break;
      }
      if (!chosen) {
        const lw = ctx.measureText(shortLbl).width + 8;
        const bx = Math.max(padL + 3, Math.min(padL + pw - lw - 3, x + 9));
        const by = Math.max(padT + 3, Math.min(padT + ph - lh - 3, y - lh - 3));
        chosen = { x1: bx, y1: by, x2: bx + lw, y2: by + lh };
        chosenLbl = shortLbl;
        chosenW = lw;
      }
      placedBoxes.push(chosen);

      const cxLbl = (chosen.x1 + chosen.x2) / 2;
      const cyLbl = (chosen.y1 + chosen.y2) / 2;
      if (Math.hypot(cxLbl - x, cyLbl - y) > 14) {
        ctx.strokeStyle = "rgba(100, 116, 139, 0.65)";
        ctx.lineWidth = 0.95;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(cxLbl, cyLbl);
        ctx.stroke();
      }
      ctx.fillStyle = isSel ? "#eff6ff" : "rgba(255, 255, 255, 0.94)";
      ctx.strokeStyle = isSel ? "#1d4ed8" : "#cbd5e1";
      ctx.lineWidth = 1;
      ctx.fillRect(chosen.x1, chosen.y1, chosenW, lh);
      ctx.strokeRect(chosen.x1, chosen.y1, chosenW, lh);
      ctx.fillStyle = "#0f172a";
      ctx.textAlign = "center";
      ctx.fillText(chosenLbl, cxLbl, chosen.y1 + 11);
    });
  }

    let plotFloatTipNode = null;
    function ensurePlotFloatTip() {
      if (!plotFloatTipNode) {
        plotFloatTipNode = document.createElement("div");
        plotFloatTipNode.className = "plot-float-tip hidden";
        document.body.appendChild(plotFloatTipNode);
      }
      return plotFloatTipNode;
    }

    function showPlotFloatTip(e, html) {
      const tip = ensurePlotFloatTip();
      tip.innerHTML = html;
      tip.style.left = `${Math.min(window.innerWidth - 295, e.clientX + 12)}px`;
      tip.style.top = `${Math.min(window.innerHeight - 95, e.clientY + 12)}px`;
      tip.classList.remove("hidden");
    }

    function hidePlotFloatTip() {
      if (plotFloatTipNode) plotFloatTipNode.classList.add("hidden");
    }

    function formatEventHoverHtml(ev, recOverride = null) {
      const ancId = ev.ancestral_event_id || ev.event_id;
      const r0 = recOverride || (ev.recombinants && ev.recombinants[0]) || null;
      const recNames = recOverride
        ? recOverride.seq_name
        : ((ev.recombinants || []).map((r) => r.seq_name).slice(0, 3).join(", ") || "-");
      const parents = r0
        ? `${r0.ext_parent_left_name || r0.ext_parent_left} → ${r0.ext_parent_right_name || r0.ext_parent_right}`
        : "-";
      return (
        `<strong>Event #${ev.event_id} (A${ancId}) @ ${fmtInt(ev.most_plausible_bp_orig)} bp</strong><br/>` +
        `KL z=${ev.score_z.toFixed(2)} | TF P=${ev.tf_prob.toFixed(3)}<br/>` +
        `Carriers: ${escapeHtml(recNames)}<br/>` +
        `Parents: ${escapeHtml(parents)}`
      );
    }

  function downloadCanvasPng(canvas, filename) {
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  function drawEventConfidenceBars(run) {
    const canvas = el("eventConfidenceCanvas");
    if (!canvas || !run) return;
    const { ctx, width, height } = setupHiDpiCanvas(canvas, 720, 210);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 52, padR = 24, padT = 18, padB = 34;
    const pw = Math.max(180, width - padL - padR);
    const ph = Math.max(90, height - padT - padB);

    ctx.fillStyle = "#fafbfc";
    ctx.fillRect(padL, padT, pw, ph);

    // Y-axis gridlines & labels
    ctx.font = "9.5px Consolas, monospace";
    [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {
      const y = padT + (1 - f) * ph;
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
      ctx.stroke();
      ctx.fillStyle = "#334155";
      ctx.textAlign = "right";
      ctx.fillText(`${Math.round(f * 100)}%`, padL - 6, y + 3);
    });

    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, pw, ph);

    const evs = run.consolidated_events || [];
    state.confBarMarkers = [];
    if (!evs.length) {
      ctx.font = "11px sans-serif";
      ctx.fillStyle = "#64748b";
      ctx.textAlign = "center";
      ctx.fillText("No breakpoints detected", padL + pw / 2, padT + ph / 2);
      return;
    }

    const thrZ = run.track_profile?.threshold_z || 6.0;
    const maxZ = Math.max(10, thrZ * 1.4, ...evs.map((e) => (e.score_z || 0) * 1.1));
    const slotW = pw / evs.length;
    const barW = Math.min(20, Math.max(5, slotW * 0.34));
    const showValLabels = slotW >= 44;

    evs.forEach((ev, idx) => {
      const cx = padL + (idx + 0.5) * slotW;
      const hZ = Math.min(1, Math.max(0, ev.score_z / maxZ)) * (ph - 12);
      const hP = Math.min(1, Math.max(0, ev.tf_prob)) * (ph - 12);
      const isSel = ev.event_id === state.selectedEventId;
      const isHover = ev.event_id === state.hoveredConfEventId;

      state.confBarMarkers.push({
        eventId: ev.event_id,
        x1: cx - barW - 6,
        x2: cx + barW + 6,
        tipHtml: formatEventHoverHtml(ev),
      });
      if (isSel || isHover) {
        ctx.fillStyle = isSel ? "rgba(29, 78, 216, 0.10)" : "rgba(100, 116, 139, 0.08)";
        ctx.fillRect(cx - barW - 5, padT + 2, barW * 2 + 10, ph - 3);
        ctx.strokeStyle = isSel ? "#1d4ed8" : "#94a3b8";
        ctx.lineWidth = isSel ? 1.4 : 1;
        ctx.strokeRect(cx - barW - 5, padT + 2, barW * 2 + 10, ph - 3);
      }

      const xZ = cx - barW - 1;
      const yZ = padT + ph - hZ;
      ctx.fillStyle = "#1d4ed8";
      ctx.fillRect(xZ, yZ, barW, hZ);

      const xP = cx + 1;
      const yP = padT + ph - hP;
      ctx.fillStyle = "#7e22ce";
      ctx.fillRect(xP, yP, barW, hP);

      if (showValLabels) {
        ctx.font = "8.5px Consolas, monospace";
        ctx.textAlign = "center";
        ctx.fillStyle = "#1e40af";
        ctx.fillText(ev.score_z.toFixed(1), xZ + barW / 2, Math.max(padT + 9, yZ - 2));
        ctx.fillStyle = "#6b21a8";
        ctx.fillText(`${Math.round(ev.tf_prob * 100)}%`, xP + barW / 2, Math.max(padT + 9, yP - 2));
      }

      ctx.fillStyle = isSel ? "#1d4ed8" : "#0f172a";
      ctx.font = isSel ? "bold 9.5px Consolas, monospace" : "9.5px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.fillText(`E${ev.event_id}`, cx, padT + ph + 13);
      if (slotW >= 54) {
        ctx.fillStyle = "#64748b";
        ctx.font = "8.5px Consolas, monospace";
        ctx.fillText(`${ev.most_plausible_bp_orig}bp`, cx, padT + ph + 24);
      }
    });
  }

  function buildCombinedPlotsCanvas(run) {
    const ds = run.alignment.dataset_name;
    drawEvidenceScatter(run);
    drawEventConfidenceBars(run);
    const c1 = el("genomeTrackCanvas");
    const c2 = el("mosaicMapCanvas");
    const c3 = el("evidenceScatterCanvas");
    const c4 = el("eventConfidenceCanvas");
    const combo = document.createElement("canvas");
    const w = Math.max(c1.width, c2.width, c3.width, c4.width, 900);
    const h = 48 + c1.height + c2.height + c3.height + c4.height + 52;
    combo.width = w;
    combo.height = h;
    const ctx = combo.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#111827";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(`Recombination Report Plots: ${ds} (${run.alignment.n_sequences} seqs × ${run.alignment.genome_length_bp} bp)`, 16, 28);
    let y = 44;
    ctx.drawImage(c1, 0, y);
    y += c1.height + 12;
    ctx.drawImage(c2, 0, y);
    y += c2.height + 12;
    ctx.drawImage(c3, 0, y);
    y += c3.height + 12;
    ctx.drawImage(c4, 0, y);
    return combo;
  }

  function exportAllFourPlots() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    const combo = buildCombinedPlotsCanvas(state.currentRun);
    const c1 = el("genomeTrackCanvas");
    const c2 = el("mosaicMapCanvas");
    const c3 = el("evidenceScatterCanvas");
    const c4 = el("eventConfidenceCanvas");
    downloadCanvasPng(c1, `${ds}_1_scan_profile.png`);
    setTimeout(() => downloadCanvasPng(c2, `${ds}_2_mosaic_map.png`), 150);
    setTimeout(() => downloadCanvasPng(c3, `${ds}_3_evidence_scatter.png`), 300);
    setTimeout(() => downloadCanvasPng(c4, `${ds}_4_event_confidence.png`), 450);
    setTimeout(() => downloadCanvasPng(combo, `${ds}_5_combined_report.png`), 600);
  }

  function buildHtmlReportText(run) {
    drawEvidenceScatter(run);
    drawEventConfidenceBars(run);
    const aln = run.alignment;
    const tim = run.timings;
    const diag = run.diagnostics || {};
    const evs = run.consolidated_events || [];
    const warnings = run.warnings || [];
    const ds = aln.dataset_name || "alignment";
    const summaryText = (el("plainEnglishSummaryText") && el("plainEnglishSummaryText").textContent) ||
      `Detected ${evs.length} breakpoint(s) across ${aln.n_sequences} sequences (${aln.genome_length_bp} bp).`;

    const imgScan = el("genomeTrackCanvas").toDataURL("image/png");
    const imgMosaic = el("mosaicMapCanvas").toDataURL("image/png");
    const imgScatter = el("evidenceScatterCanvas").toDataURL("image/png");
    const imgConf = el("eventConfidenceCanvas").toDataURL("image/png");
    let simplotPlotHtml = "";
    if (state.simplotData && el("simplotCanvas")) {
      drawSimplotCanvas(state.simplotData, run);
      const imgSim = el("simplotCanvas").toDataURL("image/png");
      simplotPlotHtml = `<div class="plot-card"><strong>5. Sliding-Window Sequence Similarity (SimPlot vs ${escapeHtml(state.simplotData.query_seq_name)})</strong><img src="${imgSim}" alt="Similarity plot" /></div>`;
    }

    const evRows = evs.length
      ? evs.map((ev) => {
          const recs = (ev.recombinants || []).map((r) => r.seq_name).join(", ") || (ev.recombinant_sequences || []).join(", ") || "-";
          const r0 = (ev.recombinants && ev.recombinants[0]) || null;
          const par = r0 ? `${r0.ext_parent_left_name || r0.ext_parent_left} → ${r0.ext_parent_right_name || r0.ext_parent_right}` : "-";
          const ancId = ev.ancestral_event_id || ev.event_id;
          return `<tr><td>#${ev.event_id}</td><td>A${ancId}</td><td>${fmtInt(ev.most_plausible_bp_orig)} bp</td><td>${fmtInt(ev.span_orig_start)}–${fmtInt(ev.span_orig_end)}</td><td>${ev.score_z.toFixed(2)}</td><td>${ev.tf_prob.toFixed(3)}</td><td>${escapeHtml(ev.signal_attribution)}</td><td>${escapeHtml(recs)}</td><td>${escapeHtml(par)}</td></tr>`;
        }).join("")
      : `<tr><td colspan="9">No breakpoints detected above threshold.</td></tr>`;

    let gtSection = "";
    if (diag.has_ground_truth && Array.isArray(diag.tolerance_metrics) && diag.tolerance_metrics.length) {
      const mRows = diag.tolerance_metrics.map((m) =>
        `<tr><td>±${fmtInt(m.tolerance_bp)} bp</td><td>${m.tp}</td><td>${m.fp}</td><td>${m.fn}</td><td>${(m.precision * 100).toFixed(1)}%</td><td>${(m.recall * 100).toFixed(1)}%</td><td>${m.f1.toFixed(3)}</td><td>${m.median_error_bp !== null ? `${m.median_error_bp.toFixed(1)} bp` : "-"}</td></tr>`
      ).join("");
      gtSection = `<h2>Ground Truth Accuracy</h2><table><thead><tr><th>Tolerance</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>Median Err</th></tr></thead><tbody>${mRows}</tbody></table>`;
    }

    const warnSection = warnings.length
      ? `<div class="warn-box"><strong>Biological / Signal Warnings:</strong><ul>${warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul></div>`
      : "";

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>gt-transformer Report: ${escapeHtml(ds)}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2937; margin: 24px auto; max-width: 1020px; padding: 0 16px; line-height: 1.45; }
    h1, h2 { color: #0f2f66; border-bottom: 1px solid #d1d5db; padding-bottom: 4px; }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; margin: 12px 0; font-size: 13px; }
    .summary { background: #eff6ff; border-left: 4px solid #1d4e89; padding: 10px 14px; margin: 12px 0; border-radius: 4px; }
    .warn-box { background: #fef2f2; border-left: 4px solid #b91c1c; padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 18px; font-size: 12px; }
    th, td { border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }
    th { background: #f1f5f9; }
    .plots { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .plot-card { border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; page-break-inside: avoid; }
    .plot-card img { max-width: 100%; height: auto; display: block; }
    .print-bar { display: flex; justify-content: flex-end; margin-bottom: 10px; }
    .print-btn { background: #1d4e89; color: #fff; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; }
    @media print { .no-print { display: none !important; } body { margin: 8mm; } }
  </style>
</head>
<body>
  <div class="print-bar no-print"><button class="print-btn" onclick="window.print()">Print / Save as PDF</button></div>
  <h1>gt-transformer Recombination Analysis Report</h1>
  <div class="meta">
    <div><strong>Dataset:</strong> ${escapeHtml(ds)}</div>
    <div><strong>Dimensions:</strong> ${aln.n_sequences} sequences × ${fmtInt(aln.genome_length_bp)} bp</div>
    <div><strong>Checkpoint:</strong> ${escapeHtml(run.config.checkpoint)}</div>
    <div><strong>Scoring Mode:</strong> ${escapeHtml(run.config.scoring_mode)}${run.config.use_mirror_tta ? " (Mirror TTA)" : ""}</div>
    <div><strong>Events / Peaks:</strong> ${evs.length} consolidated / ${(run.raw_peaks || []).length} raw</div>
    <div><strong>Runtime:</strong> ${tim.total_ms.toFixed(0)} ms (${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s)</div>
    <div><strong>GC Content:</strong> ${aln.gc_content_pct}%</div>
    <div><strong>Run ID:</strong> ${escapeHtml(tim.run_id)}</div>
  </div>
  <div class="summary">${escapeHtml(summaryText)}</div>
  ${warnSection}
  <h2>Consolidated Breakpoints (${evs.length})</h2>
  <table>
    <thead><tr><th>Event</th><th>Ancestral</th><th>Breakpoint</th><th>95% / Span</th><th>KL Z-Score</th><th>TF P(bp)</th><th>Attribution</th><th>Recombinant Sequences</th><th>Parent Pair</th></tr></thead>
    <tbody>${evRows}</tbody>
  </table>
  ${gtSection}
  <h2>Diagnostic Plots</h2>
  <div class="plots">
    <div class="plot-card"><strong>1. Breakpoint Scan Profile (KL Z-Score + Transformer P(bp))</strong><img src="${imgScan}" alt="Scan profile" /></div>
    <div class="plot-card"><strong>2. Recombinant Sequence Mosaic Map</strong><img src="${imgMosaic}" alt="Mosaic map" /></div>
    <div class="plot-card"><strong>3. Transformer vs. Flanking-KL Evidence Scatter</strong><img src="${imgScatter}" alt="Evidence scatter" /></div>
    <div class="plot-card"><strong>4. Per-Event Confidence Breakdown</strong><img src="${imgConf}" alt="Per-event confidence" /></div>
    ${simplotPlotHtml}
  </div>
</body>
</html>`;
  }

  function exportHtmlReport() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name || "alignment";
    downloadBlob(`${ds}_printable_report.html`, buildHtmlReportText(state.currentRun), "text/html");
  }

  let _crc32Table = null;
  function crc32Bytes(bytes) {
    if (!_crc32Table) {
      _crc32Table = new Uint32Array(256);
      for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        _crc32Table[i] = c >>> 0;
      }
    }
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
      crc = _crc32Table[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function buildZipUint8Array(entries) {
    const enc = new TextEncoder();
    const localChunks = [];
    const cdChunks = [];
    let offset = 0;

    entries.forEach((entry) => {
      const nameBytes = enc.encode(entry.name);
      const data = entry.bytes;
      const crc = crc32Bytes(data);

      const lh = new Uint8Array(30 + nameBytes.length);
      const lv = new DataView(lh.buffer);
      lv.setUint32(0, 0x04034b50, true);
      lv.setUint16(4, 20, true);
      lv.setUint16(6, 0x0800, true);
      lv.setUint16(8, 0, true);
      lv.setUint16(10, 0, true);
      lv.setUint16(12, 0x5421, true);
      lv.setUint32(14, crc, true);
      lv.setUint32(18, data.length, true);
      lv.setUint32(22, data.length, true);
      lv.setUint16(26, nameBytes.length, true);
      lv.setUint16(28, 0, true);
      lh.set(nameBytes, 30);

      const cd = new Uint8Array(46 + nameBytes.length);
      const cv = new DataView(cd.buffer);
      cv.setUint32(0, 0x02014b50, true);
      cv.setUint16(4, 20, true);
      cv.setUint16(6, 20, true);
      cv.setUint16(8, 0x0800, true);
      cv.setUint16(10, 0, true);
      cv.setUint16(12, 0, true);
      cv.setUint16(14, 0x5421, true);
      cv.setUint32(16, crc, true);
      cv.setUint32(20, data.length, true);
      cv.setUint32(24, data.length, true);
      cv.setUint16(28, nameBytes.length, true);
      cv.setUint16(30, 0, true);
      cv.setUint16(32, 0, true);
      cv.setUint16(34, 0, true);
      cv.setUint16(36, 0, true);
      cv.setUint32(38, 0, true);
      cv.setUint32(42, offset, true);
      cd.set(nameBytes, 46);

      localChunks.push(lh, data);
      cdChunks.push(cd);
      offset += lh.length + data.length;
    });

    const cdSize = cdChunks.reduce((acc, a) => acc + a.length, 0);
    const eocd = new Uint8Array(22);
    const ev = new DataView(eocd.buffer);
    ev.setUint32(0, 0x06054b50, true);
    ev.setUint16(4, 0, true);
    ev.setUint16(6, 0, true);
    ev.setUint16(8, entries.length, true);
    ev.setUint16(10, entries.length, true);
    ev.setUint32(12, cdSize, true);
    ev.setUint32(16, offset, true);
    ev.setUint16(20, 0, true);

    const totalLen = offset + cdSize + 22;
    const out = new Uint8Array(totalLen);
    let pos = 0;
    localChunks.forEach((c) => { out.set(c, pos); pos += c.length; });
    cdChunks.forEach((c) => { out.set(c, pos); pos += c.length; });
    out.set(eocd, pos);
    return out;
  }

  function canvasToUint8Array(canvas) {
    return new Promise((resolve) => {
      if (!canvas) return resolve(new Uint8Array(0));
      canvas.toBlob((blob) => {
        if (!blob) return resolve(new Uint8Array(0));
        blob.arrayBuffer().then((buf) => resolve(new Uint8Array(buf))).catch(() => resolve(new Uint8Array(0)));
      }, "image/png");
    });
  }

  async function exportCompleteBundleZip() {
    const run = state.currentRun;
    if (!run) return;
    const ds = run.alignment.dataset_name || "alignment";
    const combo = buildCombinedPlotsCanvas(run);
    const [p1, p2, p3, p4, p5] = await Promise.all([
      canvasToUint8Array(el("genomeTrackCanvas")),
      canvasToUint8Array(el("mosaicMapCanvas")),
      canvasToUint8Array(el("evidenceScatterCanvas")),
      canvasToUint8Array(el("eventConfidenceCanvas")),
      canvasToUint8Array(combo),
    ]);
    const enc = new TextEncoder();
    const entries = [
      { name: `${ds}_1_scan_profile.png`, bytes: p1 },
      { name: `${ds}_2_mosaic_map.png`, bytes: p2 },
      { name: `${ds}_3_evidence_scatter.png`, bytes: p3 },
      { name: `${ds}_4_event_confidence.png`, bytes: p4 },
      { name: `${ds}_5_combined_report.png`, bytes: p5 },
    ];
    if (state.simplotData && el("simplotCanvas")) {
      drawSimplotCanvas(state.simplotData, run);
      const p6 = await canvasToUint8Array(el("simplotCanvas"));
      if (p6.length > 0) entries.push({ name: `${ds}_6_simplot_similarity.png`, bytes: p6 });
    }
    entries.push(
      { name: `${ds}.rdp`, bytes: enc.encode(buildRdpProjectText(run)) },
      { name: `${ds}_breakpoints.gff3`, bytes: enc.encode(buildGff3Text(run)) },
      { name: `${ds}_breakpoints.bed`, bytes: enc.encode(buildBedText(run)) },
      { name: `${ds}_printable_report.html`, bytes: enc.encode(buildHtmlReportText(run)) },
      { name: `${ds}_breakpoints_consolidated.csv`, bytes: enc.encode(buildBreakpointsCsvText(run)) },
      { name: `${ds}_sequence_calls_consolidated.csv`, bytes: enc.encode(buildSequenceCallsCsvText(run)) },
      { name: `${ds}_summary.tsv`, bytes: enc.encode(buildOpenRdpTsvText(run)) },
      { name: `${ds}_diagnostic_report.json`, bytes: enc.encode(JSON.stringify(run, null, 2)) }
    );
    const zipBytes = buildZipUint8Array(entries);
    downloadBlob(`${ds}_complete_bundle.zip`, zipBytes, "application/zip");
  }

  async function refreshAiChatStatus() {
    const st = el("aiChatStatus");
    if (!st) return;
    try {
      const res = await fetch("/api/chat/status");
      const data = await res.json();
      st.textContent = data.configured
        ? `Connected (${data.provider}${data.model ? ` · ${data.model}` : ""})`
        : "Not configured (add API key in web/.env)";
    } catch {
      st.textContent = "Status unavailable";
    }
  }

  function appendChatBubble(role, text, opts = {}) {
    const box = el("aiChatMessages");
    if (!box) return null;
    const div = document.createElement("div");
    div.className = `ai-msg ${role === "user" ? "ai-msg-user" : "ai-msg-bot"}${opts.thinking ? " ai-msg-thinking" : ""}`;
    if (role === "assistant" && !opts.thinking) {
      const body = document.createElement("div");
      body.innerHTML = renderSafeMarkdown(text);
      div.appendChild(body);
      const actions = document.createElement("div");
      actions.className = "ai-msg-actions";
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "ai-copy-btn";
      copyBtn.textContent = "Copy";
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = "Copied";
          setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
        } catch {}
      });
      actions.appendChild(copyBtn);
      div.appendChild(actions);
    } else {
      div.textContent = text;
    }
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
  }

  async function sendAiChatMessage(question) {
    appendChatBubble("user", question);
    const thinkingNode = appendChatBubble("assistant", "Thinking...", { thinking: true });
    let trackSummary = null;
    if (state.currentRun && state.currentRun.track_profile) {
      const tp = state.currentRun.track_profile;
      const pos = tp.positions_orig_bp || [];
      const kl = tp.flank_kl_z || [];
      const tfp = tp.tf_prob || [];
      let maxKl = 0, maxKlBp = 0, maxTf = 0, maxTfBp = 0;
      for (let i = 0; i < pos.length; i++) {
        if ((kl[i] || 0) > maxKl) { maxKl = kl[i]; maxKlBp = pos[i]; }
        if ((tfp[i] || 0) > maxTf) { maxTf = tfp[i]; maxTfBp = pos[i]; }
      }
      trackSummary = {
        threshold_z: tp.threshold_z,
        max_flank_kl_z: Number(maxKl.toFixed(3)),
        max_flank_kl_bp: maxKlBp,
        max_tf_prob: Number(maxTf.toFixed(4)),
        max_tf_prob_bp: maxTfBp,
      };
    }

    const runContext = state.currentRun
      ? {
          alignment: {
            dataset_name: state.currentRun.alignment.dataset_name,
            source_type: state.currentRun.alignment.source_type,
            n_sequences: state.currentRun.alignment.n_sequences,
            genome_length_bp: state.currentRun.alignment.genome_length_bp,
            n_informative_cols: state.currentRun.alignment.n_informative_cols,
            n_transformed_cols: state.currentRun.alignment.n_transformed_cols,
            gc_content_pct: state.currentRun.alignment.gc_content_pct,
            edge_dead_zone_bp: state.currentRun.alignment.edge_dead_zone_bp,
          },
          config: state.currentRun.config,
          timings: state.currentRun.timings,
          warnings: state.currentRun.warnings,
          track_summary: trackSummary,
          diagnostics: state.currentRun.diagnostics,
          consolidated_events: state.currentRun.consolidated_events || [],
          simplot_query: state.simplotData ? state.simplotData.query_seq_name : null,
          selected_event_id: state.selectedEventId,
          selected_region: state.selectedRegion,
          zoomed_region: state.plotViewRange,
          region_stats:
            state.lastRegionStats &&
            state.selectedRegion &&
            state.lastRegionStats.start_bp === state.selectedRegion.startBp &&
            state.lastRegionStats.end_bp === state.selectedRegion.endBp
              ? state.lastRegionStats
              : null,
        }
      : null;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: state.chatHistory,
          run_context: runContext,
        }),
      });
      const data = await resp.json();
      const ans = data.answer || data.detail || "No response.";
      if (thinkingNode) thinkingNode.remove();
      appendChatBubble("assistant", ans);
      state.chatHistory.push({ role: "user", content: question });
      state.chatHistory.push({ role: "assistant", content: ans });
    } catch (err) {
      if (thinkingNode) thinkingNode.remove();
      appendChatBubble("assistant", `Error: ${err.message}`);
    }
  }

  async function refreshCurrentUser() {
    try {
      const res = await fetch("/api/auth/me");
      const data = await res.json();
      state.currentUser = data.user || null;
      updateAuthUi();
    } catch {
      state.currentUser = null;
    }
  }

  function updateAuthUi() {
    const u = state.currentUser;
    const btn = el("btnAuthAccount");
    const histBtn = el("btnMyHistory");
    const admBtn = el("btnAdminPanel");
    if (u) {
      if (btn) btn.textContent = u.username;
      if (histBtn) histBtn.classList.remove("hidden");
      if (admBtn) admBtn.classList.toggle("hidden", u.role !== "admin");
      if (el("authLoggedOutView")) el("authLoggedOutView").classList.add("hidden");
      if (el("authLoggedInView")) el("authLoggedInView").classList.remove("hidden");
      if (el("authLoggedInLabel")) el("authLoggedInLabel").textContent = `Signed in as ${u.username} (${u.role})`;
    } else {
      if (btn) btn.textContent = "Sign In";
      if (histBtn) histBtn.classList.add("hidden");
      if (admBtn) admBtn.classList.add("hidden");
      if (el("authLoggedOutView")) el("authLoggedOutView").classList.remove("hidden");
      if (el("authLoggedInView")) el("authLoggedInView").classList.add("hidden");
    }
  }

  async function submitAuth(endpoint) {
    const errBox = el("authErrorMsg");
    errBox.classList.add("hidden");
    const username = el("authUsername").value.trim();
    const password = el("authPassword").value;
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        errBox.textContent = data.detail || "Authentication failed.";
        errBox.classList.remove("hidden");
        return;
      }
      state.currentUser = data.user;
      updateAuthUi();
      el("authModal").classList.add("hidden");
    } catch (err) {
      errBox.textContent = err.message;
      errBox.classList.remove("hidden");
    }
  }

  async function openMyHistoryModal() {
    el("historyModal").classList.remove("hidden");
    const body = el("myRunsTableBody");
    const search = el("myRunsSearchInput");
    body.innerHTML = '<tr><td colspan="6" class="mono-muted">Loading...</td></tr>';
    const res = await fetch("/api/runs");
    const data = await res.json();
    const allRuns = data.runs || [];
    const renderRows = () => {
      const q = (search ? search.value : "").trim().toLowerCase();
      const runs = q
        ? allRuns.filter((r) => `${r.dataset_name} ${r.checkpoint}`.toLowerCase().includes(q))
        : allRuns;
      if (!runs.length) {
        body.innerHTML = '<tr><td colspan="6" class="mono-muted">No matching saved runs.</td></tr>';
        return;
      }
      body.innerHTML = "";
      runs.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${escapeHtml(r.dataset_name)}</td><td>${r.n_sequences} × ${r.genome_length_bp} bp</td><td>${r.n_events}</td><td>${escapeHtml(r.checkpoint)}</td><td>${Math.round(r.total_ms)} ms</td><td style="display:flex;gap:4px;"><button type="button" class="mini-btn btn-load">Load</button><button type="button" class="mini-btn btn-json">JSON</button><button type="button" class="mini-btn btn-del">Delete</button></td>`;
        tr.querySelector(".btn-load").addEventListener("click", async () => {
          const dRes = await fetch(`/api/runs/${r.id}`);
          if (dRes.ok) {
            const full = await dRes.json();
            displayRunPayload(full.result);
            el("historyModal").classList.add("hidden");
            const evs = full.result.consolidated_events || [];
            const startBp = evs.length > 0 ? Math.max(0, evs[0].most_plausible_bp_orig - Math.floor(state.alnWidth / 2)) : 0;
            loadAlignmentWindow(startBp);
          }
        });
        tr.querySelector(".btn-json").addEventListener("click", async () => {
          const dRes = await fetch(`/api/runs/${r.id}`);
          if (dRes.ok) {
            const full = await dRes.json();
            downloadBlob(`${r.dataset_name}_${r.id}.json`, JSON.stringify(full.result, null, 2), "application/json");
          }
        });
        tr.querySelector(".btn-del").addEventListener("click", async () => {
          const delRes = await fetch(`/api/runs/${r.id}`, { method: "DELETE" });
          if (delRes.ok) {
            const idx = allRuns.findIndex((x) => x.id === r.id);
            if (idx >= 0) allRuns.splice(idx, 1);
            renderRows();
          }
        });
        body.appendChild(tr);
      });
    };
    if (search) search.oninput = renderRows;
    renderRows();
  }

  async function openAdminModal() {
    el("adminModal").classList.remove("hidden");
    const runsBody = el("adminRunsBody");
    const usersBody = el("adminUsersBody");
    const retBody = el("adminRetrainBody");
    const kpiStrip = el("adminKpiStrip");
    const search = el("adminSearchInput");
    runsBody.innerHTML = '<tr><td colspan="7" class="mono-muted">Loading...</td></tr>';
    retBody.innerHTML = '<tr><td colspan="6" class="mono-muted">Loading...</td></tr>';
    const res = await fetch("/api/admin/overview");
    if (!res.ok) {
      runsBody.innerHTML = '<tr><td colspan="7" class="mono-muted">Admin access required.</td></tr>';
      return;
    }
    const data = await res.json();
    const st = data.stats || {};
    if (kpiStrip) {
      kpiStrip.innerHTML = `<span><strong>Users:</strong> ${st.total_users ?? 0}</span><span><strong>Total Runs:</strong> ${st.total_runs ?? 0}</span><span><strong>User Uploads:</strong> ${st.total_uploads ?? 0}</span><span><strong>Adaptive Updates:</strong> ${st.retrain_updates ?? 0}</span><span><strong>Avg Runtime:</strong> ${st.avg_ms ?? 0} ms</span>`;
    }

    const adaptiveLine = el("adminAdaptiveStatus");
    const resetBtn = el("btnAdminResetAdaptive");
    if (adaptiveLine) {
      try {
        const aRes = await fetch("/api/admin/adaptive/status");
        if (aRes.ok) {
          const ast = await aRes.json();
          adaptiveLine.textContent = ast.exists
            ? `Adaptive model: ${ast.updates} update(s) · ${ast.size_kb} KB · last run=${ast.last_run_id || "-"}`
            : "Adaptive model: not initialized yet (uses base ckpt_N10000)";
        }
      } catch {}
    }
    if (resetBtn) {
      resetBtn.onclick = async () => {
        resetBtn.disabled = true;
        try {
          const rRes = await fetch("/api/admin/adaptive/reset", { method: "POST" });
          if (rRes.ok) {
            await loadConfigOptions();
            await openAdminModal();
          }
        } finally {
          resetBtn.disabled = false;
        }
      };
    }
    const allRuns = data.runs || [];
    const renderRuns = () => {
      const q = (search ? search.value : "").trim().toLowerCase();
      const filtered = q
        ? allRuns.filter((r) => `${r.username} ${r.dataset_name} ${r.checkpoint}`.toLowerCase().includes(q))
        : allRuns;
      runsBody.innerHTML = "";
      filtered.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${escapeHtml(r.username)}</td><td>${escapeHtml(r.dataset_name)}</td><td>${escapeHtml(r.source_type)}</td><td>${r.n_sequences}×${r.genome_length_bp}</td><td>${r.n_events}</td><td>${escapeHtml(r.checkpoint)}</td><td style="display:flex;gap:4px;"><a class="mini-btn" href="/api/admin/runs/${r.id}/fasta">FASTA</a><a class="mini-btn" href="/api/admin/runs/${r.id}/result" download="${escapeHtml(r.dataset_name)}_${r.id}.json">JSON</a><button type="button" class="mini-btn btn-view-admin">View</button><button type="button" class="mini-btn btn-del-admin">Delete</button></td>`;
        tr.querySelector(".btn-view-admin").addEventListener("click", async () => {
          const dRes = await fetch(`/api/admin/runs/${r.id}/result`);
          if (dRes.ok) {
            const full = await dRes.json();
            displayRunPayload(full.result);
            el("adminModal").classList.add("hidden");
            const evs = full.result.consolidated_events || [];
            const startBp = evs.length > 0 ? Math.max(0, evs[0].most_plausible_bp_orig - Math.floor(state.alnWidth / 2)) : 0;
            loadAlignmentWindow(startBp);
          }
        });
        tr.querySelector(".btn-del-admin").addEventListener("click", async () => {
          const delRes = await fetch(`/api/admin/runs/${r.id}`, { method: "DELETE" });
          if (delRes.ok) {
            const idx = allRuns.findIndex((x) => x.id === r.id);
            if (idx >= 0) allRuns.splice(idx, 1);
            renderRuns();
          }
        });
        runsBody.appendChild(tr);
      });
    };
    if (search) search.oninput = renderRuns;
    renderRuns();

    if (usersBody) {
      usersBody.innerHTML = "";
      (data.users || []).forEach((u) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${escapeHtml(u.username)}</td><td>${escapeHtml(u.email || "-")}</td><td>${escapeHtml(u.auth_provider || "local")}</td><td>${escapeHtml(u.role)}</td><td>${u.run_count ?? 0}</td>`;
        usersBody.appendChild(tr);
      });
    }
    retBody.innerHTML = "";
    (data.retrain_log || []).forEach((rl) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(rl.dataset_name)}</td><td>${escapeHtml(rl.status)}</td><td>${rl.loss_before ?? "-"}</td><td>${rl.loss_after ?? "-"}</td><td>${rl.steps}</td><td>${escapeHtml(rl.notes)}</td>`;
      retBody.appendChild(tr);
    });
  }

  async function openCompareModal() {
    el("compareModal").classList.remove("hidden");
    const selA = el("compareRunASelect");
    const selB = el("compareRunBSelect");
    const metricsBody = el("compareMetricsBody");
    const eventsDiff = el("compareEventsDiff");
    metricsBody.innerHTML = '<tr><td colspan="4" class="mono-muted">Loading runs...</td></tr>';
    eventsDiff.textContent = "";

    const candidates = new Map();
    state.recentRuns.forEach((r) => {
      const rid = r.timings.run_id;
      candidates.set(rid, {
        id: rid,
        label: `${r.alignment.dataset_name} · ${r.config.checkpoint} · ${rid} (session)`,
        result: r,
      });
    });

    if (state.currentUser) {
      try {
        const res = await fetch("/api/runs");
        if (res.ok) {
          const data = await res.json();
          (data.runs || []).forEach((r) => {
            if (!candidates.has(r.id)) {
              candidates.set(r.id, {
                id: r.id,
                label: `${r.dataset_name} · ${r.checkpoint} · ${r.id} (saved)`,
                result: null,
              });
            }
          });
        }
      } catch {}
    }

    const list = Array.from(candidates.values());
    if (!list.length) {
      selA.innerHTML = "<option value=\"\">No runs</option>";
      selB.innerHTML = "<option value=\"\">No runs</option>";
      metricsBody.innerHTML = '<tr><td colspan="4" class="mono-muted">Run at least one analysis (or sign in with saved runs) to compare side by side.</td></tr>';
      return;
    }

    const optsHtml = list.map((c) => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`).join("");
    selA.innerHTML = optsHtml;
    selB.innerHTML = optsHtml;
    selA.selectedIndex = 0;
    selB.selectedIndex = Math.min(1, list.length - 1);

    const resolveRun = async (id) => {
      const entry = candidates.get(id);
      if (!entry) return null;
      if (entry.result) return entry.result;
      const rRes = await fetch(`/api/runs/${encodeURIComponent(id)}`);
      if (rRes.ok) {
        const full = await rRes.json();
        entry.result = full.result;
        return full.result;
      }
      return null;
    };

    const refresh = async () => {
      const [runA, runB] = await Promise.all([resolveRun(selA.value), resolveRun(selB.value)]);
      renderRunComparison(runA, runB);
    };
    selA.onchange = refresh;
    selB.onchange = refresh;
    await refresh();
  }

  function renderRunComparison(runA, runB) {
    const body = el("compareMetricsBody");
    const diffBox = el("compareEventsDiff");
    if (!runA || !runB) {
      body.innerHTML = '<tr><td colspan="4" class="mono-muted">Select two valid runs to compare.</td></tr>';
      diffBox.textContent = "";
      return;
    }

    const evA = runA.consolidated_events || [];
    const evB = runB.consolidated_events || [];
    const maxZA = evA.length ? Math.max(...evA.map((e) => e.score_z)) : 0;
    const maxZB = evB.length ? Math.max(...evB.map((e) => e.score_z)) : 0;
    const gtStr = (r) => {
      const d = r.diagnostics;
      if (!d || !d.has_ground_truth || !d.tolerance_metrics || !d.tolerance_metrics.length) return "Unlabelled";
      const m = d.tolerance_metrics.find((x) => x.tolerance_bp === d.active_tolerance_bp) || d.tolerance_metrics[0];
      return `Recall=${(m.recall * 100).toFixed(1)}% · F1=${m.f1.toFixed(3)}`;
    };

    const rows = [
      ["Dataset", runA.alignment.dataset_name, runB.alignment.dataset_name, runA.alignment.dataset_name === runB.alignment.dataset_name ? "Same dataset" : "Different dataset"],
      ["Checkpoint / Mode", `${runA.config.checkpoint} (${runA.config.scoring_mode})`, `${runB.config.checkpoint} (${runB.config.scoring_mode})`, ""],
      ["Dimensions", `${runA.alignment.n_sequences} × ${fmtInt(runA.alignment.genome_length_bp)} bp`, `${runB.alignment.n_sequences} × ${fmtInt(runB.alignment.genome_length_bp)} bp`, ""],
      ["Consolidated Events", `${evA.length}`, `${evB.length}`, `${evB.length - evA.length >= 0 ? "+" : ""}${evB.length - evA.length}`],
      ["Raw Peaks", `${(runA.raw_peaks || []).length}`, `${(runB.raw_peaks || []).length}`, `${(runB.raw_peaks || []).length - (runA.raw_peaks || []).length >= 0 ? "+" : ""}${(runB.raw_peaks || []).length - (runA.raw_peaks || []).length}`],
      ["Max KL Z-Score", maxZA.toFixed(2), maxZB.toFixed(2), `${maxZB - maxZA >= 0 ? "+" : ""}${(maxZB - maxZA).toFixed(2)}`],
      ["Runtime (ms)", `${runA.timings.total_ms.toFixed(1)} ms`, `${runB.timings.total_ms.toFixed(1)} ms`, `${runB.timings.total_ms - runA.timings.total_ms >= 0 ? "+" : ""}${(runB.timings.total_ms - runA.timings.total_ms).toFixed(1)} ms`],
      ["Accuracy (vs GT)", gtStr(runA), gtStr(runB), ""],
    ];
    body.innerHTML = rows.map((r) =>
      `<tr><td><strong>${escapeHtml(r[0])}</strong></td><td class="mono">${escapeHtml(r[1])}</td><td class="mono">${escapeHtml(r[2])}</td><td class="mono">${escapeHtml(r[3])}</td></tr>`
    ).join("");

    // Breakpoint overlap within +/-200 bp
    const tol = 200;
    const matchedB = new Set();
    const pairs = [];
    const onlyA = [];
    evA.forEach((a) => {
      let bestIdx = -1;
      let bestDist = Infinity;
      evB.forEach((b, idx) => {
        if (matchedB.has(idx)) return;
        const d = Math.abs(a.most_plausible_bp_orig - b.most_plausible_bp_orig);
        if (d <= tol && d < bestDist) {
          bestDist = d;
          bestIdx = idx;
        }
      });
      if (bestIdx >= 0) {
        matchedB.add(bestIdx);
        const b = evB[bestIdx];
        pairs.push(`Matched: A#${a.event_id} (${fmtInt(a.most_plausible_bp_orig)} bp) ↔ B#${b.event_id} (${fmtInt(b.most_plausible_bp_orig)} bp, Δ=${bestDist} bp)`);
      } else {
        onlyA.push(`Only Run A: #${a.event_id} @ ${fmtInt(a.most_plausible_bp_orig)} bp (z=${a.score_z.toFixed(2)})`);
      }
    });
    const onlyB = evB
      .filter((_, idx) => !matchedB.has(idx))
      .map((b) => `Only Run B: #${b.event_id} @ ${fmtInt(b.most_plausible_bp_orig)} bp (z=${b.score_z.toFixed(2)})`);
    const lines = [...pairs, ...onlyA, ...onlyB];
    diffBox.innerHTML = lines.length
      ? lines.map((l) => `<div>${escapeHtml(l)}</div>`).join("")
      : "No breakpoints detected in either run.";
  }

  // -------------------------------------------------------------------------
  // Region Diversity & Pairwise Identity Inspector (Plot 5)
  // -------------------------------------------------------------------------
  async function openRegionInspectorModal() {
    if (!state.currentRun || !state.selectedRegion) return;
    el("regionModal").classList.remove("hidden");
    const { startBp, endBp } = state.selectedRegion;
    let defaultRef = state.selectedCarrierSeqIdx ?? 0;
    const overlapEv = (state.currentRun.consolidated_events || []).find(
      (ev) => ev.most_plausible_bp_orig >= startBp && ev.most_plausible_bp_orig <= endBp
    );
    if (overlapEv && overlapEv.recombinants && overlapEv.recombinants.length > 0) {
      defaultRef = overlapEv.recombinants[0].seq_idx;
    }
    await loadRegionStats(defaultRef);
  }

  async function loadRegionStats(refSeqIdx) {
    if (!state.currentRun || !state.selectedRegion) return;
    const kpi = el("regionKpiStrip");
    const body = el("regionPairwiseBody");
    const sel = el("regionRefSeqSelect");
    const { startBp, endBp } = state.selectedRegion;
    const runId = state.currentRun.timings.run_id;
    kpi.innerHTML = "<span>Loading region statistics...</span>";
    body.innerHTML = "";
    try {
      const qs = new URLSearchParams({
        run_id: runId,
        start_bp: String(startBp),
        end_bp: String(endBp),
        ref_seq_idx: String(refSeqIdx ?? 0),
      });
      const res = await fetch(`/api/alignment/region-stats?${qs.toString()}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        kpi.innerHTML = `<span class="mono-muted">${escapeHtml(err.detail || "Unable to compute region stats.")}</span>`;
        return;
      }
      const stats = await res.json();
      state.lastRegionStats = stats;
      kpi.innerHTML =
        `<span><strong>Window:</strong> ${fmtInt(stats.start_bp)}–${fmtInt(stats.end_bp)} bp (${fmtInt(stats.length_bp)} bp)</span>` +
        `<span><strong>Informative:</strong> ${fmtInt(stats.informative_sites)}</span>` +
        `<span><strong>GC%:</strong> ${stats.gc_pct}%</span>` +
        `<span><strong>Gaps/Ambig:</strong> ${stats.gap_pct}%</span>` +
        `<span><strong>Diversity (π):</strong> ${stats.nucleotide_diversity_pi}</span>`;
      if (sel) {
        sel.innerHTML = (stats.sequences || [])
          .map((s) => `<option value="${s.seq_idx}">${s.seq_idx}: ${escapeHtml(s.name)}</option>`)
          .join("");
        sel.value = String(stats.ref_seq_idx);
      }
      const rows = stats.pairwise || [];
      body.innerHTML = rows
        .map((r) => {
          const inStr = r.inside_pct !== null ? `${r.inside_pct.toFixed(2)}%` : "-";
          const outStr = r.outside_pct !== null ? `${r.outside_pct.toFixed(2)}%` : "-";
          const dCls = r.delta_pct !== null ? (r.delta_pct > 0 ? "delta-pos" : r.delta_pct < 0 ? "delta-neg" : "") : "";
          const dStr = r.delta_pct !== null ? `${r.delta_pct >= 0 ? "+" : ""}${r.delta_pct.toFixed(2)}%` : "-";
          return `<tr><td class="mono">${r.seq_idx}: ${escapeHtml(r.seq_name)}</td><td class="mono">${inStr}</td><td class="mono">${outStr}</td><td class="mono ${dCls}">${dStr}</td></tr>`;
        })
        .join("");
      drawRegionIdentityChart(stats);
    } catch (err) {
      kpi.innerHTML = `<span class="mono-muted">${escapeHtml(err.message)}</span>`;
    }
  }

  function drawRegionIdentityChart(stats) {
    const canvas = el("regionIdentityCanvas");
    if (!canvas || !stats) return;
    const { ctx, width, height } = setupHiDpiCanvas(canvas, 740, 240);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 50, padR = 18, padT = 32, padB = 58;
    const pw = Math.max(120, width - padL - padR);
    const ph = Math.max(80, height - padT - padB);
    ctx.fillStyle = "#fafbfc";
    ctx.fillRect(padL, padT, pw, ph);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, pw, ph);
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 11.5px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      `Pairwise Identity vs ${stats.ref_seq_name} (${stats.start_bp}–${stats.end_bp} bp): Inside (Blue) vs Outside (Gray)`,
      padL + pw / 2,
      19
    );

    const items = (stats.pairwise || []).slice(0, 14);
    if (!items.length) return;
    ctx.font = "9.5px Consolas, monospace";
    ctx.textAlign = "right";
    [0, 25, 50, 75, 100].forEach((pct) => {
      const y = padT + (1 - pct / 100) * ph;
      ctx.fillStyle = "#334155";
      ctx.fillText(`${pct}%`, padL - 6, y + 3);
      ctx.strokeStyle = "#e2e8f0";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
      ctx.stroke();
    });

    const slotW = pw / items.length;
    const barW = Math.min(20, Math.max(5, slotW * 0.34));
    items.forEach((r, idx) => {
      const cx = padL + (idx + 0.5) * slotW;
      const hIn = r.inside_pct !== null ? (Math.min(100, Math.max(0, r.inside_pct)) / 100) * (ph - 4) : 0;
      const hOut = r.outside_pct !== null ? (Math.min(100, Math.max(0, r.outside_pct)) / 100) * (ph - 4) : 0;
      ctx.fillStyle = "#1d4ed8";
      ctx.fillRect(cx - barW - 1, padT + ph - hIn, barW, hIn);
      ctx.fillStyle = "#94a3b8";
      ctx.fillRect(cx + 1, padT + ph - hOut, barW, hOut);

      if (slotW >= 46) {
        ctx.font = "8.5px Consolas, monospace";
        ctx.textAlign = "center";
        if (r.inside_pct !== null) {
          ctx.fillStyle = "#1e40af";
          ctx.fillText(`${r.inside_pct.toFixed(0)}%`, cx - barW / 2 - 1, Math.max(padT + 9, padT + ph - hIn - 2));
        }
        if (r.outside_pct !== null) {
          ctx.fillStyle = "#475569";
          ctx.fillText(`${r.outside_pct.toFixed(0)}%`, cx + barW / 2 + 1, Math.max(padT + 9, padT + ph - hOut - 2));
        }
      }

      ctx.save();
      ctx.translate(cx, padT + ph + 8);
      ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = "#1e293b";
      ctx.font = "9px Consolas, monospace";
      ctx.textAlign = "right";
      ctx.fillText(String(r.seq_name).slice(0, 14), 0, 0);
      ctx.restore();
    });
  }

  // -------------------------------------------------------------------------
  // Plot 5 (Graph 5): Sliding-Window Sequence Similarity (SimPlot)
  // -------------------------------------------------------------------------
  function getSimplotParentRoles(data, run) {
    const leftNames = new Set();
    const rightNames = new Set();
    const leftIdxs = new Set();
    const rightIdxs = new Set();
    if (!run || !data) return { leftNames, rightNames, leftIdxs, rightIdxs };
    const evs = run.consolidated_events || [];
    const selEv = evs.find((e) => e.event_id === state.selectedEventId) || evs[0] || null;
    const matchEvs = [];
    if (selEv) matchEvs.push(selEv);
    evs.forEach((ev) => {
      if (ev !== selEv && (ev.recombinants || []).some((r) => r.seq_idx === data.query_seq_idx || r.seq_name === data.query_seq_name)) {
        matchEvs.push(ev);
      }
    });
    matchEvs.forEach((ev) => {
      const rec = (ev.recombinants || []).find((r) => r.seq_idx === data.query_seq_idx || r.seq_name === data.query_seq_name) || (ev.recombinants && ev.recombinants[0]);
      if (!rec) return;
      if (rec.ext_parent_left !== undefined && rec.ext_parent_left !== null) leftIdxs.add(rec.ext_parent_left);
      if (rec.ext_parent_right !== undefined && rec.ext_parent_right !== null) rightIdxs.add(rec.ext_parent_right);
      if (rec.ext_parent_left_name) String(rec.ext_parent_left_name).split(" or ").forEach((n) => leftNames.add(n.trim()));
      if (rec.ext_parent_right_name) String(rec.ext_parent_right_name).split(" or ").forEach((n) => rightNames.add(n.trim()));
    });
    return { leftNames, rightNames, leftIdxs, rightIdxs };
  }

  async function loadSimplotProfile(queryOverride = null) {
    const run = state.currentRun;
    if (!run || !run.timings || !run.timings.run_id) return;
    const evs = run.consolidated_events || [];
    const selEv = evs.find((e) => e.event_id === state.selectedEventId) || evs[0] || null;
    let qIdx = queryOverride !== null && !Number.isNaN(queryOverride)
      ? queryOverride
      : (state.selectedCarrierSeqIdx !== null && state.selectedCarrierSeqIdx !== undefined
        ? state.selectedCarrierSeqIdx
        : (selEv && selEv.recombinants && selEv.recombinants[0] ? selEv.recombinants[0].seq_idx : 0));

    const priSet = new Set();
    evs.forEach((ev) => {
      (ev.recombinants || []).slice(0, 3).forEach((r) => {
        if (r.ext_parent_left !== undefined) priSet.add(r.ext_parent_left);
        if (r.ext_parent_right !== undefined) priSet.add(r.ext_parent_right);
      });
    });

    const winSel = el("simplotWindowSelect");
    const winBp = winSel ? Number(winSel.value) || 100 : 100;
    const stepBp = Math.max(5, Math.min(40, Math.round(winBp / 6)));
    const qs = new URLSearchParams({
      run_id: run.timings.run_id,
      query_seq_idx: String(qIdx),
      window_bp: String(winBp),
      step_bp: String(stepBp),
      max_refs: "8",
      priority_seqs: Array.from(priSet).join(","),
    });

    const res = await fetch(`/api/alignment/simplot?${qs.toString()}`);
    if (!res.ok) return;
    const data = await res.json();
    state.simplotData = data;
    const qSel = el("simplotQuerySelect");
    if (qSel && Array.isArray(data.sequences)) {
      qSel.innerHTML = data.sequences
        .map((s) => `<option value="${s.seq_idx}">${s.seq_idx}: ${escapeHtml(s.name)}</option>`)
        .join("");
      qSel.value = String(data.query_seq_idx);
    }
    drawSimplotCanvas(data, run);
  }

  function drawSimplotCanvas(data, run) {
    const canvas = el("simplotCanvas");
    const legend = el("simplotLegend");
    if (!canvas || !data) return;
    const { ctx, width, height } = setupHiDpiCanvas(canvas, 680, 290);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 52, padR = 18, padT = 38, padB = 34;
    const pw = Math.max(120, width - padL - padR);
    const ph = Math.max(100, height - padT - padB);
    const L = Math.max(1, data.genome_length_bp || 1);
    const pos = data.positions_bp || [];
    const series = data.series || [];
    const autoScale = el("simplotAutoScaleY") ? el("simplotAutoScaleY").checked : true;
    const parentRoles = getSimplotParentRoles(data, run);

    // Assign colors: Left Parent = royal blue, Right Parent = crimson red, others = clean distinct palette
    const bgPalette = ["#0f766e", "#b45309", "#6d28d9", "#4b5563", "#0369a1", "#65a30d", "#9333ea", "#475569"];
    let bgIdx = 0;
    const seriesMeta = series.map((ser) => {
      const isLeft = parentRoles.leftIdxs.has(ser.seq_idx) || parentRoles.leftNames.has(ser.seq_name);
      const isRight = parentRoles.rightIdxs.has(ser.seq_idx) || parentRoles.rightNames.has(ser.seq_name);
      let col;
      if (isLeft && isRight) col = "#7c3aed";
      else if (isLeft) col = "#1d4ed8";
      else if (isRight) col = "#dc2626";
      else col = bgPalette[(bgIdx++) % bgPalette.length];
      return {
        ser,
        col,
        isLeft,
        isRight,
        isVisible: !state.simplotHiddenSeqs.has(ser.seq_idx),
      };
    });

    // Dynamic Y-axis bounds so crossovers fill the canvas instead of squashing at the top
    let yMin = 0;
    const yMax = 100;
    if (autoScale) {
      let obsMin = 100;
      seriesMeta.forEach((m) => {
        if (!m.isVisible) return;
        (m.ser.identity_pct || []).forEach((v) => {
          if (v !== null && v !== undefined && v < obsMin) obsMin = v;
        });
      });
      if (obsMin < 99.5) {
        yMin = Math.max(0, Math.min(85, Math.floor((obsMin - 4) / 5) * 5));
      } else {
        yMin = 70;
      }
    }
    const ySpan = Math.max(5, yMax - yMin);
    const xOfBp = (bp) => padL + (Math.max(0, Math.min(L, bp)) / L) * pw;
    const yOfPct = (pct) => {
      const clamped = Math.max(yMin, Math.min(yMax, pct));
      return padT + (1 - (clamped - yMin) / ySpan) * ph;
    };

    ctx.fillStyle = "#fafbfc";
    ctx.fillRect(padL, padT, pw, ph);

    // Top Dominant Parental Lineage Ribbon (shows which reference wins at each window)
    if (pos.length > 0 && seriesMeta.some((m) => m.isVisible)) {
      const ribbonY = padT;
      const ribbonH = 7;
      for (let i = 0; i < pos.length; i++) {
        let bestMeta = null;
        let bestVal = -1;
        seriesMeta.forEach((m) => {
          if (!m.isVisible) return;
          const v = m.ser.identity_pct[i];
          if (v !== null && v !== undefined && v > bestVal) {
            bestVal = v;
            bestMeta = m;
          }
        });
        if (bestMeta) {
          const xPrev = i === 0 ? padL : xOfBp((pos[i - 1] + pos[i]) / 2);
          const xNext = i === pos.length - 1 ? padL + pw : xOfBp((pos[i] + pos[i + 1]) / 2);
          ctx.fillStyle = bestMeta.col;
          ctx.globalAlpha = 0.72;
          ctx.fillRect(xPrev, ribbonY, Math.max(1.5, xNext - xPrev + 0.5), ribbonH);
          ctx.globalAlpha = 1.0;
        }
      }
      ctx.fillStyle = "#475569";
      ctx.font = "8.5px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText("Closest:", padL - 6, ribbonY + 6.5);
    }

    // Horizontal Y grid & dynamic ticks
    ctx.font = "9.5px Consolas, monospace";
    const stepTick = ySpan <= 25 ? 5 : ySpan <= 55 ? 10 : 25;
    const ticks = [];
    for (let t = Math.ceil(yMin / stepTick) * stepTick; t <= 100; t += stepTick) {
      ticks.push(t);
    }
    if (ticks[0] !== yMin) ticks.unshift(yMin);
    ticks.forEach((pct) => {
      const y = yOfPct(pct);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
      ctx.stroke();
      ctx.fillStyle = "#1e293b";
      ctx.textAlign = "right";
      ctx.fillText(`${pct}%`, padL - 6, y + 3);
    });

    // Shaded breakpoint spans & vertical lines
    state.simplotBpMarkers = [];
    const evs = (run && run.consolidated_events) || [];
    evs.forEach((ev) => {
      const x1 = xOfBp(ev.span_orig_start || ev.most_plausible_bp_orig);
      const x2 = xOfBp(ev.span_orig_end || ev.most_plausible_bp_orig);
      const bx = xOfBp(ev.most_plausible_bp_orig);
      const isSel = ev.event_id === state.selectedEventId;
      state.simplotBpMarkers.push({ eventId: ev.event_id, x: bx });
      if (x2 > x1 + 1) {
        ctx.fillStyle = isSel ? "rgba(37, 99, 235, 0.11)" : "rgba(148, 163, 184, 0.08)";
        ctx.fillRect(x1, padT + 7, x2 - x1, ph - 7);
      }
      ctx.save();
      ctx.strokeStyle = isSel ? "#1e293b" : "#94a3b8";
      ctx.lineWidth = isSel ? 1.8 : 1.0;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(bx, padT + 7);
      ctx.lineTo(bx, padT + ph);
      ctx.stroke();
      ctx.restore();
    });

    // Draw background non-parent curves first, then bold Left & Right parents on top
    const orderedSeries = [
      ...seriesMeta.filter((m) => !m.isLeft && !m.isRight),
      ...seriesMeta.filter((m) => m.isLeft || m.isRight),
    ];

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT + 7, pw, ph - 7);
    ctx.clip();

    orderedSeries.forEach((m) => {
      if (!m.isVisible) return;
      const isKeyParent = m.isLeft || m.isRight;
      ctx.strokeStyle = m.col;
      ctx.lineWidth = isKeyParent ? 2.8 : 1.35;
      ctx.globalAlpha = isKeyParent ? 1.0 : 0.72;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < pos.length; i++) {
        const v = m.ser.identity_pct[i];
        if (v === null || v === undefined) {
          started = false;
          continue;
        }
        const x = xOfBp(pos[i]);
        const y = yOfPct(v);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    });

    // Live interactive vertical hover crosshair + intersection dots
    if (state.simplotHoverX !== null && pos.length > 0) {
      const lx = Math.max(padL, Math.min(padL + pw, state.simplotHoverX));
      const targetBp = ((lx - padL) / pw) * L;
      let bestI = 0;
      let bestDist = Infinity;
      for (let i = 0; i < pos.length; i++) {
        const d = Math.abs(pos[i] - targetBp);
        if (d < bestDist) { bestDist = d; bestI = i; }
      }
      const snapX = xOfBp(pos[bestI]);
      ctx.strokeStyle = "rgba(15, 23, 42, 0.55)";
      ctx.lineWidth = 1.1;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(snapX, padT + 7);
      ctx.lineTo(snapX, padT + ph);
      ctx.stroke();
      ctx.setLineDash([]);

      orderedSeries.forEach((m) => {
        if (!m.isVisible) return;
        const v = m.ser.identity_pct[bestI];
        if (v === null || v === undefined) return;
        const cy = yOfPct(v);
        ctx.fillStyle = m.col;
        ctx.beginPath();
        ctx.arc(snapX, cy, m.isLeft || m.isRight ? 4.5 : 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });
    }
    ctx.restore();

    // Plot outer frame
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, pw, ph);

    // Compact metadata subtitle in top margin (never clips)
    ctx.fillStyle = "#475569";
    ctx.font = "9.5px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Query: ${String(data.query_seq_name || "").slice(0, 28)} | win=${data.window_bp}bp, step=${data.step_bp}bp`, padL, 14);
    ctx.textAlign = "right";
    ctx.fillText(autoScale ? `Y-Scale: Auto (${yMin}% to 100%)` : "Y-Scale: Full (0% to 100%)", padL + pw, 14);

    // Non-overlapping top event flags above frame
    let lastFlagRight = -999;
    [...evs].sort((a, b) => a.most_plausible_bp_orig - b.most_plausible_bp_orig).forEach((ev) => {
      const bx = xOfBp(ev.most_plausible_bp_orig);
      const isSel = ev.event_id === state.selectedEventId;
      const lbl = `E${ev.event_id} (${ev.most_plausible_bp_orig}bp)`;
      ctx.font = isSel ? "bold 9px Consolas, monospace" : "9px Consolas, monospace";
      const tw = ctx.measureText(lbl).width + 8;
      if (bx - tw / 2 > lastFlagRight + 2) {
        const fx = Math.max(padL + tw / 2, Math.min(padL + pw - tw / 2, bx));
        ctx.fillStyle = isSel ? "#1d4ed8" : "#f1f5f9";
        ctx.strokeStyle = isSel ? "#1e3a8a" : "#94a3b8";
        ctx.fillRect(fx - tw / 2, padT - 15, tw, 13);
        ctx.strokeRect(fx - tw / 2, padT - 15, tw, 13);
        ctx.fillStyle = isSel ? "#ffffff" : "#334155";
        ctx.textAlign = "center";
        ctx.fillText(lbl, fx, padT - 5.5);
        lastFlagRight = fx + tw / 2;
      }
    });

    // Y-axis title
    ctx.save();
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 10px sans-serif";
    ctx.translate(14, padT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Nucleotide Identity (%)", 0, 0);
    ctx.restore();

    // X ticks
    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i <= 5; i++) {
      const bp = Math.round((L * i) / 5);
      const tx = xOfBp(bp);
      ctx.fillStyle = "#1e293b";
      ctx.fillText(fmtInt(bp), tx, padT + ph + 14);
    }
    ctx.font = "bold 10.5px sans-serif";
    ctx.fillText("Position in Alignment (bp)", padL + pw / 2, height - 4);

    // Interactive click-to-toggle legend chips
    if (legend) {
      legend.innerHTML = "";
      seriesMeta.forEach((m) => {
        const chip = document.createElement("button");
        chip.type = "button";
        const roleTag = m.isLeft && m.isRight
          ? " [L/R Parent]"
          : (m.isLeft ? " [Left Parent]" : (m.isRight ? " [Right Parent]" : ""));
        chip.className = `simplot-chip${!m.isVisible ? " is-hidden" : ""}${m.isLeft ? " is-parent-left" : ""}${m.isRight ? " is-parent-right" : ""}`;
        chip.title = "Click to toggle this curve on/off";
        chip.innerHTML = `<span class="simplot-swatch" style="background:${m.col};"></span>${escapeHtml(m.ser.seq_name)}${escapeHtml(roleTag)}`;
        chip.addEventListener("click", () => {
          if (state.simplotHiddenSeqs.has(m.ser.seq_idx)) state.simplotHiddenSeqs.delete(m.ser.seq_idx);
          else state.simplotHiddenSeqs.add(m.ser.seq_idx);
          drawSimplotCanvas(state.simplotData, state.currentRun);
        });
        legend.appendChild(chip);
      });
    }
  }

  // -------------------------------------------------------------------------
  // Multi-File Batch FASTA Runner & Exports
  // -------------------------------------------------------------------------
  async function runBatchFastaFiles(files) {
    el("batchModal").classList.remove("hidden");
    const prog = el("batchProgressText");
    const body = el("batchResultsBody");
    state.batchResults = files.map((f) => ({
      file,
      filename: f.name,
      status: "Pending",
      result: null,
      error: null,
    }));

    const renderBatchTable = () => {
      const doneCnt = state.batchResults.filter((x) => x.status === "Done" || x.status === "Error").length;
      prog.textContent = `Processed ${doneCnt} / ${state.batchResults.length} alignment(s)`;
      body.innerHTML = "";
      state.batchResults.forEach((row) => {
        const tr = document.createElement("tr");
        const dims = row.result ? `${row.result.alignment.n_sequences} × ${fmtInt(row.result.alignment.genome_length_bp)} bp` : "-";
        const nEv = row.result ? String((row.result.consolidated_events || []).length) : "-";
        const ms = row.result ? `${Math.round(row.result.timings.total_ms)} ms` : "-";
        const stText = row.error ? `Error: ${row.error}` : row.status;
        tr.innerHTML =
          `<td>${escapeHtml(row.filename)}</td>` +
          `<td class="mono">${escapeHtml(stText)}</td>` +
          `<td class="mono">${escapeHtml(dims)}</td>` +
          `<td class="num">${escapeHtml(nEv)}</td>` +
          `<td class="mono">${escapeHtml(ms)}</td>` +
          `<td>${row.result ? '<button type="button" class="mini-btn btn-batch-load">Load</button>' : "-"}</td>`;
        if (row.result) {
          tr.querySelector(".btn-batch-load").addEventListener("click", async () => {
            displayRunPayload(row.result);
            el("batchModal").classList.add("hidden");
            const evs = row.result.consolidated_events || [];
            const startBp = evs.length > 0 ? Math.max(0, evs[0].most_plausible_bp_orig - Math.floor(state.alnWidth / 2)) : 0;
            await loadAlignmentWindow(startBp);
          });
        }
        body.appendChild(tr);
      });
    };

    renderBatchTable();
    for (const item of state.batchResults) {
      item.status = "Queued";
      renderBatchTable();
      try {
        const fd = new FormData();
        fd.append("checkpoint", el("checkpointSelect").value);
        fd.append("scoring_mode", el("scoringModeSelect").value);
        fd.append("use_mirror_tta", el("optMirrorTta")?.checked ? "true" : "false");
        fd.append("null_policy", el("nullPolicyInput").value);
        fd.append("flank_window", el("flankWindowInput").value);
        fd.append("beta", el("betaInput").value);
        fd.append("z_thresh", el("zThreshInput").value);
        fd.append("tf_logit_thresh", el("tfLogitThreshInput").value);
        fd.append("merge_radius_bp", el("mergeRadiusInput").value);
        fd.append("eval_tolerance_bp", el("evalToleranceInput").value);
        fd.append("fasta_file", item.file);

        const initResp = await fetch("/api/jobs", { method: "POST", body: fd });
        if (!initResp.ok) {
          const err = await initResp.json().catch(() => ({}));
          item.status = "Error";
          item.error = err.detail || `HTTP ${initResp.status}`;
          renderBatchTable();
          continue;
        }
        const { job_id } = await initResp.json();
        while (true) {
          const pResp = await fetch(`/api/jobs/${encodeURIComponent(job_id)}`);
          if (!pResp.ok) {
            item.status = "Error";
            item.error = `HTTP ${pResp.status}`;
            break;
          }
          const job = await pResp.json();
          item.status = job.status === "done" ? "Done" : `Running: ${job.stage || ""}`;
          if (job.status === "done" && job.result) {
            item.status = "Done";
            item.result = job.result;
            rememberRun(job.result);
            break;
          }
          if (job.status === "error" || job.status === "cancelled") {
            item.status = "Error";
            item.error = job.error || job.status;
            break;
          }
          renderBatchTable();
          await sleepMs(600);
        }
      } catch (err) {
        item.status = "Error";
        item.error = err.message;
      }
      renderBatchTable();
    }
  }

  function exportBatchCsv() {
    if (!state.batchResults.length) return;
    const header = ["filename", "status", "n_sequences", "genome_length_bp", "n_events", "breakpoints_bp", "total_ms", "error"];
    const rows = state.batchResults.map((r) => {
      const evs = r.result ? (r.result.consolidated_events || []) : [];
      return [
        `"${String(r.filename).replace(/"/g, '""')}"`,
        r.status,
        r.result ? r.result.alignment.n_sequences : "",
        r.result ? r.result.alignment.genome_length_bp : "",
        r.result ? evs.length : "",
        `"${evs.map((e) => e.most_plausible_bp_orig).join(";")}"`,
        r.result ? r.result.timings.total_ms.toFixed(1) : "",
        `"${String(r.error || "").replace(/"/g, '""')}"`,
      ].join(",");
    });
    downloadBlob("batch_recombination_summary.csv", [header.join(","), ...rows].join("\n"), "text/csv");
  }

  function exportBatchJson() {
    if (!state.batchResults.length) return;
    const payload = state.batchResults.map((r) => ({
      filename: r.filename,
      status: r.status,
      error: r.error,
      result: r.result,
    }));
    downloadBlob("batch_recombination_results.json", JSON.stringify(payload, null, 2), "application/json");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
