"""AI Assistant service for explaining recombination analysis results to students and researchers."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ (without overwriting non-empty env)."""
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and val and not os.environ.get(key):
            os.environ[key] = val


def get_active_model(provider: str | None) -> str | None:
    """Return the active model name configured for a given LLM provider."""
    if provider == "anthropic":
        raw = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip() or "claude-haiku-4-5-20251001"
        return _resolve_anthropic_api_model(raw)
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return None


def _resolve_anthropic_api_model(configured_model: str | None) -> str:
    """Normalize non-existent/alias Anthropic model slugs to a live API model ID."""
    m = (configured_model or "").strip()
    aliases = {
        "": "claude-haiku-4-5-20251001",
        "claude-haiku-5-5": "claude-haiku-4-5-20251001",
        "claude-5-5-haiku": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    }
    return aliases.get(m, m)


def detect_provider() -> tuple[str | None, str | None]:
    """Return (provider_name, api_key) honoring AI_PROVIDER if set, otherwise first configured key."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    pref = os.environ.get("AI_PROVIDER", "").strip().lower()
    if pref in key_map:
        val = os.environ.get(key_map[pref], "").strip()
        if val:
            return pref, val
    for env_key, provider in (
        ("OPENAI_API_KEY", "openai"),
        ("GEMINI_API_KEY", "gemini"),
        ("ANTHROPIC_API_KEY", "anthropic"),
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            return provider, val
    return None, None


def _summarize_run_context(run_context: dict[str, Any]) -> str:
    """Produce a compact, lossless structured digest omitting raw genome track arrays."""
    aln = run_context.get("alignment") or {}
    cfg = run_context.get("config") or {}
    timings = run_context.get("timings") or {}
    warnings = run_context.get("warnings") or []
    events = run_context.get("consolidated_events") or run_context.get("events") or []
    diag = run_context.get("diagnostics") or {}
    track_summary = run_context.get("track_summary") or {}

    lines = [
        "### Active Alignment & Run Configuration",
        (
            f"- Dataset: {aln.get('dataset_name', 'unknown')} ({aln.get('source_type', 'upload')}), "
            f"{aln.get('n_sequences', '?')} seqs × {aln.get('genome_length_bp', '?')} bp, "
            f"informative={aln.get('n_informative_cols', '?')}, M_tx={aln.get('n_transformed_cols', '?')}, "
            f"GC={aln.get('gc_content_pct', '?')}%, edge_dead_zone=±{aln.get('edge_dead_zone_bp', 30)} bp"
        ),
        (
            f"- Checkpoint: {cfg.get('checkpoint', '?')}, Mode: {cfg.get('scoring_mode', '?')}, "
            f"MirrorTTA: {cfg.get('use_mirror_tta', False)}, z_thresh={cfg.get('z_thresh', '?')}, "
            f"tf_logit_thresh={cfg.get('tf_logit_thresh', '?')}, flank_window={cfg.get('flank_window', '?')}, "
            f"merge_radius={cfg.get('merge_radius_bp', '?')}bp, Runtime: {timings.get('total_ms', '?')} ms"
        ),
    ]
    if track_summary:
        lines.append(
            f"- Scan Peaks: max Flanking KL z={track_summary.get('max_flank_kl_z')} @ {track_summary.get('max_flank_kl_bp')} bp "
            f"(null z_thresh={track_summary.get('threshold_z')}), max Transformer P(bp)={track_summary.get('max_tf_prob')} @ {track_summary.get('max_tf_prob_bp')} bp"
        )
    if run_context.get("selected_event_id") is not None:
        lines.append(f"- User UI Focus: Event #{run_context.get('selected_event_id')}, SimPlot Query={run_context.get('simplot_query') or 'auto'}")
    if run_context.get("selected_region"):
        sr = run_context["selected_region"]
        lines.append(f"- Selected Graph Region: {sr.get('startBp')}..{sr.get('endBp')} bp")
    if run_context.get("region_stats"):
        rs = run_context["region_stats"]
        lines.append(
            f"- Region Diversity Stats ({rs.get('start_bp')}..{rs.get('end_bp')}bp vs {rs.get('ref_seq_name')}): "
            f"informative={rs.get('informative_sites')}, GC={rs.get('gc_pct')}%, gaps={rs.get('gap_pct')}%, pi={rs.get('nucleotide_diversity_pi')}"
        )

    if warnings:
        lines.append("### Warnings")
        for w in warnings[:10]:
            lines.append(f"- {w}")

    lines.append(f"### Consolidated Events ({len(events)})")
    if not events:
        lines.append("- None detected.")
    for ev in events:
        bp_val = ev.get("most_plausible_bp_orig", ev.get("bp_orig"))
        s_start = ev.get("span_orig_start", (ev.get("span") or [None, None])[0])
        s_end = ev.get("span_orig_end", (ev.get("span") or [None, None])[1])
        kl_z = ev.get("score_z", ev.get("kl_z"))
        lines.append(
            f"- Event #{ev.get('event_id')} (Ancestral #{ev.get('ancestral_event_id', ev.get('event_id'))}): "
            f"bp={bp_val} (span {s_start}..{s_end}, peaks={ev.get('n_peaks_consolidated', '?')}), "
            f"z={kl_z}, P(bp)={ev.get('tf_prob')}, logit={ev.get('tf_logit')}, "
            f"attribution={ev.get('signal_attribution', ev.get('attribution'))}, status={ev.get('match_status')}"
        )
        recs = ev.get("recombinants") or []
        for rec in recs[:6]:
            lines.append(
                f"  - Carrier {rec.get('seq_name')} (p_switch={rec.get('p_switch')}): "
                f"{rec.get('ext_parent_left_name', rec.get('ext_parent_left'))} (sim={rec.get('left_sim')}) -> "
                f"{rec.get('ext_parent_right_name', rec.get('ext_parent_right'))} (sim={rec.get('right_sim')})"
            )
        if not recs and ev.get("parents_L_to_R"):
            for p_line in ev["parents_L_to_R"][:6]:
                lines.append(f"  - Carrier/Parents: {p_line}")
        cands = ev.get("parent_candidates") or []
        if cands:
            top_c = ", ".join(
                f"{c.get('seq_name')}(L={c.get('left_sim')},R={c.get('right_sim')})"
                for c in cands[:4]
            )
            lines.append(f"  - Level-3 Top Parental Candidates: {top_c}")

    if diag.get("has_ground_truth"):
        lines.append("### Ground-Truth Diagnostics")
        for tm in diag.get("tolerance_metrics") or []:
            lines.append(
                f"- Tol ±{tm.get('tolerance_bp')}bp: TP={tm.get('tp')}, FP={tm.get('fp')}, FN={tm.get('fn')}, "
                f"Prec={tm.get('precision')}, Rec={tm.get('recall')}, F1={tm.get('f1')}, MedErr={tm.get('median_error_bp')}bp"
            )
        for gt in diag.get("ground_truth_items") or []:
            lines.append(
                f"- GT bp={gt.get('gt_bp_orig')} ({gt.get('event_type')}): detected={gt.get('detected')}, "
                f"matched_bp={gt.get('matched_pred_bp_orig')}, err={gt.get('bp_error')}, note={gt.get('failure_hypothesis', '')}"
            )
        for note in diag.get("summary_notes") or []:
            lines.append(f"- Note: {note}")

    return "\n".join(lines)[:14000]


def _build_system_prompt(run_context: dict[str, Any] | None) -> str:
    base = (
        "You are the embedded Expert AI Bioinformatics Assistant for **gt-transformer**, a hybrid neural + analytical "
        "phylogenetic recombination detection workbench designed for viral, bacterial, and population-genetic alignments.\n\n"
        "### Complete System Knowledge (What You Know Inside and Out)\n"
        "1. **Detection Architecture (6-Stage Pipeline)**:\n"
        "   - *Stage 1–2 (Parsing & Filtering)*: Extracts biallelic parsimony-informative sites (>=2 alleles in >=2 sequences; 3rd/4th states handled via duplicate/zero/drop null policy).\n"
        "   - *Stage 3 (4-Gamete / Two-Site Representation)*: Encodes overlapping W=314 informative-site circular/sliding windows of pairwise 2-site genotype patterns (00, 01, 10, 11) and inter-site distances.\n"
        "   - *Stage 4 (Transformer Boundary Scorer)*: Axial/1D Window Transformer outputs per-column boundary probability `P(bp) = sigmoid(logit)`. Supports bidirectional **Mirror TTA** (averages forward + reverse-complement passes for sharper localization).\n"
        "   - *Stage 5 (Analytical Flanking KL Scan)*: Computes asymmetric KL divergence between left and right flanking nearest-neighbor affinity distributions across multi-scale windows (`L_w` + tight `w=16` refinement for short tracts). Uses **dual sum + max-sequence KL** (so even rare 1-carrier recombinants in large N=96+ clades aren't diluted) normalized against a column-shuffled permutation null (`z_thresh`, default ~6.0).\n"
        "   - *Stage 6 (3-Level Hierarchical Consolidation & Ancestral Grouping)*:\n"
        "     - Level 1: Merges nearby raw peaks (`merge_radius_bp`, default 75bp) into consolidated events (`event_id`, `span_orig_start..end`, `most_plausible_bp_orig`). Groups co-descendant events sharing parental lineages + breakpoint locality into `ancestral_event_id` (e.g. `A1`).\n"
        "     - Level 2: Per-carrier sequence attribution (`p_switch` KL shift, internal & external `Left Parent` -> `Right Parent`, `left_sim`, `right_sim`), including co-descendant exclusion so siblings share true external parents.\n"
        "     - Level 3: Ranked parental similarity table across all candidate sequences.\n"
        "2. **Checkpoints & Scoring Modes**:\n"
        "   - `ckpt_N40000`: Default 40,000-alignment corpus model (28k train / 6k val / 6k test).\n"
        "   - `ckpt_N10000`: 10,000-alignment standard checkpoint.\n"
        "   - `ckpt_N10000_extreme`: Trained on 0.2x-5x out-of-distribution mutation/recombination rate regimes.\n"
        "   - `ckpt_N10000_v2`: LD-invariant multi-rate Conv+Contrast architecture (resistant to linkage disequilibrium shifts).\n"
        "   - `ckpt_ensemble`: 3-model LogMeanExp ensemble combining strongest Standard (40k), Extreme, and V2 neural predictions.\n"
        "   - `ckpt_user_adaptive`: Continually adapted checkpoint from user-confirmed patterns.\n"
        "   - Scoring Modes: `hybrid_flank` (recommended consensus + tight-peak pinning), `flank_kl` (pure analytical permutation-null scan), `transformer` (neural-only), `global_kl` (global KL z=3.0).\n"
        "   - **Evaluation Clarity (Always State the Exact Unit of Measurement)**: Never cite an unqualified AUPRC. Always explicitly state what TP, FP, and FN count: (a) dataset-level FPR on clonal controls and dataset-level detection power, (b) breakpoint-level 1-to-1 matching within a stated ±bp tolerance + median localization error, (c) recombinant sequence/carrier identification precision/recall, (d) ancestral co-descendant grouping accuracy, and (e) per-informative-column neural boundary ranking.\n"
        "3. **Interactive Workbench Graphs & How to Interpret Them**:\n"
        "   - **Graph 1 (Overview Density Strip)**: Magenta density shows polymorphic/informative site concentration; blue bars mark breakpoint spans; click anywhere to jump the alignment viewer.\n"
        "   - **Multiple Sequence Alignment Viewer (Top-Left)**: Highlights recombinant carriers (`[REC]`), parental lineages (`[PAR]`), informative columns, and the active breakpoint red column divider.\n"
        "   - **Graph 2 (Breakpoint Scan Plot)**: Black curve = Flanking KL Z-score (left axis); Purple curve = Transformer `P(bp)` (right axis 0–1); Red dashed line = permutation null Z-threshold. Drag across the plot to inspect diversity/pairwise shifts in any region, or zoom in.\n"
        "   - **Graph 3 (Mosaic Architecture Map)**: Horizontal parental tract map showing `[0..bp]` Left Parent color -> `[bp..L]` Right Parent color with uncertainty box.\n"
        "   - **Graph 4A/4B (Evidence Scatter & Confidence Bars)**: Quadrant plot separating *Dual-Supported Consensus* (top-right, highest confidence) from *KL-Driven / Short-Tract* (bottom-right) and *Transformer-Only* (top-left).\n"
        "   - **Graph 5 (Sliding-Window SimPlot)**: Auto-zoomed `% nucleotide identity` of the recombinant query against candidate parents. True crossovers appear cleanly where the bold **Blue (Left Parent)** curve drops and bold **Red (Right Parent)** curve rises across the dashed breakpoint line. Top ribbon shows closest lineage per window.\n"
        "4. **Known Biological Caveats**:\n"
        "   - Edge dead-zone (~30 bp at alignment ends) lacks sufficient flanking informative columns.\n"
        "   - Short tracts (<60–100 bp) or recombination near low-diversity tracts may show strong KL Z-score with lower wide-window Transformer P(bp) (`flank_kl_heuristic_only`).\n"
        "   - Homoplasy/saturation can inflate weak KL peaks unless backed by parental sequence switching in SimPlot.\n"
        "5. **Export Formats**: `.rdp` Project File, `.gff3` & `.bed` Genome Tracks, Split Recombinant Tracts `.fasta`, `.csv`, `.tsv`, `.json`, Printable HTML Report, Individual/Combined PNGs, and Complete `.zip` Bundle.\n\n"
        "### Response Guidelines\n"
        "- Be clear, deeply insightful, scientifically rigorous yet approachable for both students and principal investigators (like evolutionary virologists).\n"
        "- Directly cite exact event IDs, ancestral group IDs (`A#`), breakpoint coordinates (bp), Z-scores, `P(bp)`, sequence names, similarities, and warnings from the active run digest below.\n"
        "- Suggest concrete next UI actions when helpful (e.g., viewing SimPlot for a specific carrier, enabling Mirror TTA or `ckpt_ensemble`, zooming into a region, or exporting `.rdp`/`.gff3`).\n"
        "- Never invent run numbers, sequences, or events not present in the digest."
    )
    if not run_context:
        return base + "\n\nNo alignment has been run yet — answer general questions about gt-transformer, inputs, models, graphs, or exports."
    return base + "\n\n" + _summarize_run_context(run_context)


def ask_ai_assistant(
    question: str,
    history: list[dict[str, str]],
    run_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Send a question + run context to the configured LLM provider."""
    provider, api_key = detect_provider()
    if not provider or not api_key:
        return {
            "configured": False,
            "provider": None,
            "answer": (
                "No AI API key is configured yet. Open `web/.env`, paste your "
                "`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`, and restart the server."
            ),
        }

    sys_prompt = _build_system_prompt(run_context)
    trimmed_hist = [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:2000]}
        for m in (history or [])[-8:]
        if m.get("role") in ("user", "assistant")
    ]

    try:
        if provider == "openai":
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            messages = [{"role": "system", "content": sys_prompt}] + trimmed_hist + [
                {"role": "user", "content": question}
            ]
            payload = json.dumps({"model": model, "messages": messages, "temperature": 0.3}).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ans = data["choices"][0]["message"]["content"]
            return {"configured": True, "provider": "openai", "model": model, "answer": ans}

        if provider == "gemini":
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            contents = []
            for m in trimmed_hist:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            contents.append({"role": "user", "parts": [{"text": question}]})
            payload = json.dumps(
                {
                    "system_instruction": {"parts": [{"text": sys_prompt}]},
                    "contents": contents,
                }
            ).encode("utf-8")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ans = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"configured": True, "provider": "gemini", "model": model, "answer": ans}

        if provider == "anthropic":
            raw_model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            model = _resolve_anthropic_api_model(raw_model)
            msgs = trimmed_hist + [{"role": "user", "content": question}]
            try_models = [model] if model == "claude-haiku-4-5-20251001" else [model, "claude-haiku-4-5-20251001"]
            last_exc: Exception | None = None
            for idx, m_try in enumerate(try_models):
                payload = json.dumps(
                    {
                        "model": m_try,
                        "max_tokens": 2048,
                        "system": sys_prompt,
                        "messages": msgs,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    ans = data["content"][0]["text"]
                    return {"configured": True, "provider": "anthropic", "model": m_try, "answer": ans}
                except urllib.error.HTTPError as http_exc:
                    last_exc = http_exc
                    if http_exc.code in (400, 404) and idx + 1 < len(try_models):
                        continue
                    raise
            if last_exc is not None:
                raise last_exc

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:300]
        return {
            "configured": True,
            "provider": provider,
            "model": get_active_model(provider),
            "answer": f"API error ({exc.code}) from {provider}: {body}",
        }
    except Exception as exc:
        return {
            "configured": True,
            "provider": provider,
            "model": get_active_model(provider),
            "answer": f"Failed to reach {provider}: {exc}",
        }

    return {"configured": False, "provider": None, "answer": "Unsupported provider."}
