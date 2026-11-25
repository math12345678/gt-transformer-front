#!/usr/bin/env python3
"""
FastAPI Web Application Entrypoint for GT-Transformer Recombination Diagnostic Interface.

Runs live inference using the trained JAX Window Transformer checkpoints,
variance-stabilised local flanking KL (`flank_kl`), permutation-null calibration,
and connected-component event consolidation (`report.py`).
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.runs import router as runs_router
from schemas import FeedbackAnnotation, InferenceConfig, InferenceResponse
from services.benchmarks import load_benchmark_summaries
from services.chat import load_env_file
from services.db import (
    add_annotation as db_add_annotation,
    get_run_by_any_id,
    get_run_for_user,
    get_user_by_session,
    init_db,
    list_annotations as db_list_annotations,
    save_run,
)
from services.self_train import request_shutdown as request_self_train_shutdown, retrain_on_user_input
from services.fasta_utils import (
    MAX_UPLOAD_BYTES,
    AlignmentValidationError,
    BinaryPayloadError,
    EmptyPayloadError,
    InsufficientInformativeSitesError,
    parse_and_validate_fasta,
    parse_raw_fasta_records,
    validate_alignment,
)
from services.inference_adapter import PRESET_REGISTRY, GTTransformerInferenceEngine
from services.timing import TimingHistoryStore

HERE = Path(__file__).resolve().parent
load_env_file(HERE / ".env")
STATIC_DIR = HERE / "static"
TEMPLATES_DIR = HERE / "templates"
ANNOTATIONS_PATH = HERE / "reviewer_annotations.json"

logger = logging.getLogger("uvicorn.error")

init_db()
timing_store = TimingHistoryStore(log_path=str(HERE / "timing_history.json"))
engine = GTTransformerInferenceEngine(timing_store=timing_store)

# Pre-load checkpoints once at module startup and log load time
engine.warmup("ckpt_N10000")
logger.info(
    "GT-Transformer checkpoints pre-loaded in %.2f ms (warmed=%s)",
    engine.startup_load_ms,
    engine.is_warmed_up,
)

# Background job queue: 1 worker thread, max 1 running + 5 queued, 1-hour retention
JOB_RETENTION_SECONDS = 3600.0
MAX_ACTIVE_JOBS = 6  # 1 running + 5 queued
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gt_infer_worker")
_RETRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gt_retrain_worker")
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _prune_expired_jobs(now: Optional[float] = None) -> None:
    now_ts = now if now is not None else time.time()
    expired = [
        jid
        for jid, j in _JOBS.items()
        if j["status"] in ("done", "error", "cancelled")
        and j.get("finished_at") is not None
        and (now_ts - float(j["finished_at"])) > JOB_RETENTION_SECONDS
    ]
    for jid in expired:
        _JOBS.pop(jid, None)


def _shutdown_executors() -> None:
    request_self_train_shutdown()
    for ex in (_JOB_EXECUTOR, _RETRAIN_EXECUTOR):
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


atexit.register(_shutdown_executors)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    if not engine.is_warmed_up:
        engine.warmup("ckpt_N10000")
    try:
        yield
    finally:
        _shutdown_executors()


app = FastAPI(
    title="gt-transformer | Recombination Inference & Diagnostic Interface",
    description=(
        "Web interface for running live gt-transformer inference, recording wall-clock timings, "
        "inspecting 3-level outputs, and diagnosing model successes (TP) vs failures (FP/FN)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
server = app
app.state.engine = engine
app.include_router(auth_router)
app.include_router(runs_router)
app.include_router(chat_router)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.exception_handler(AlignmentValidationError)
async def alignment_validation_handler(request: Request, exc: AlignmentValidationError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    detail = str(exc)
    status_code = (
        422
        if any(
            k in detail
            for k in (
                "identical lengths",
                "at least 4 sequences",
                "polymorphic sites",
                "gap characters",
            )
        )
        else 400
    )
    return JSONResponse(status_code=status_code, content={"detail": detail})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


AVAILABLE_SCORING_MODES = [
    {
        "id": "hybrid_flank",
        "label": "Hybrid (flank_kl + Window Transformer) — Default",
        "description": "Combines null-calibrated local flanking KL z-score (Lw=250) with Window Transformer boundary logits.",
    },
    {
        "id": "standalone_tf",
        "label": "Standalone Window Transformer",
        "description": "Trained 4-layer pre-norm encoder over circular (W=314, 4) pairwise 4-gamete windows + carrier switch verification.",
    },
    {
        "id": "flank_kl",
        "label": "Local Flanking KL (flank_kl only)",
        "description": "Symmetric flanking window [b-w, b) vs [b, b+w) soft-parent KL localiser calibrated against column-permutation null.",
    },
    {
        "id": "global_kl",
        "label": "Legacy Global Segment KL (kl, min_seg=200)",
        "description": "Original prefix [0, b) vs suffix [b, M) KL scorer (retained for side-by-side comparison).",
    },
]


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    """Serve the diagnostic interface HTML page."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="Frontend template index.html not found.")
    return FileResponse(str(index_file), headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/privacy", response_class=HTMLResponse)
@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy() -> FileResponse:
    """Serve the Privacy Policy page (used for Google OAuth consent screen compliance)."""
    page = TEMPLATES_DIR / "privacy.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Privacy policy page not found.")
    return FileResponse(str(page), headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/terms", response_class=HTMLResponse)
@app.get("/terms-of-service", response_class=HTMLResponse)
async def terms_of_service() -> FileResponse:
    """Serve the Terms of Service page."""
    page = TEMPLATES_DIR / "terms.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Terms of service page not found.")
    return FileResponse(str(page), headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Deployment health check returning 200 once the model is loaded."""
    if not engine.is_warmed_up:
        engine.warmup("ckpt_N10000")
    return {
        "status": "ok",
        "model_loaded": engine.is_warmed_up,
        "startup_load_ms": engine.startup_load_ms,
        "checkpoints": [c["id"] for c in engine.list_checkpoints()],
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Return server status, live engine state, and available checkpoints."""
    return {
        "status": "ok",
        "engine_mode": "live",
        "model_loaded": engine.is_warmed_up,
        "startup_load_ms": engine.startup_load_ms,
        "infrastructure_imported": True,
        "checkpoints": engine.list_checkpoints(),
        "scoring_modes": AVAILABLE_SCORING_MODES,
    }


@app.get("/api/presets")
async def list_presets() -> dict[str, Any]:
    """List verified benchmark, 10k simulation, extreme, and empirical alignments on disk."""
    return {
        "presets": engine.list_presets(),
        "checkpoints": engine.list_checkpoints(),
        "scoring_modes": AVAILABLE_SCORING_MODES,
    }


@app.get("/api/benchmarks")
async def get_benchmarks() -> dict[str, Any]:
    """Return verified 10,000-alignment scaling and out-of-range specificity summaries."""
    return load_benchmark_summaries()


def _hydrate_alignment_if_needed(request: Request, run_id: str) -> None:
    with engine._alignment_cache_lock:
        if run_id in engine._alignment_cache:
            return
    user = get_user_by_session(request.cookies.get("recomb_session"))
    try:
        saved = get_run_by_any_id(run_id, user)
    except PermissionError as pexc:
        raise HTTPException(status_code=403, detail=str(pexc)) from pexc
    if saved is not None:
        null_policy = saved.get("result", {}).get("config", {}).get("null_policy", "duplicate")
        engine.hydrate_alignment_cache(run_id, saved["fasta_text"], null_policy=null_policy)


@app.get("/api/alignment/window")
async def get_alignment_window(
    request: Request,
    run_id: str,
    start: int = 0,
    width: int = 110,
) -> dict[str, Any]:
    """Return a real nucleotide window slice and informative-site flags for the alignment viewer."""
    _hydrate_alignment_if_needed(request, run_id)
    try:
        return engine.get_alignment_window(run_id=run_id, start_bp=start, width=width)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/alignment/region-stats")
async def get_alignment_region_stats(
    request: Request,
    run_id: str,
    start_bp: int,
    end_bp: int,
    ref_seq_idx: Optional[int] = None,
) -> dict[str, Any]:
    """Return nucleotide diversity and inside-vs-outside pairwise identity for a selected alignment region."""
    _hydrate_alignment_if_needed(request, run_id)
    try:
        return engine.get_region_stats(
            run_id=run_id,
            start_bp=start_bp,
            end_bp=end_bp,
            ref_seq_idx=ref_seq_idx,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/alignment/simplot")
async def get_alignment_simplot(
    request: Request,
    run_id: str,
    query_seq_idx: int = 0,
    window_bp: int = 100,
    step_bp: int = 20,
    max_refs: int = 8,
    priority_seqs: str = "",
) -> dict[str, Any]:
    """Return vectorized sliding-window pairwise identity curves against candidate parental sequences."""
    _hydrate_alignment_if_needed(request, run_id)
    pri_list: list[int] = []
    for tok in priority_seqs.split(","):
        t = tok.strip()
        if t.lstrip("-").isdigit():
            pri_list.append(int(t))
    try:
        return engine.get_simplot_profile(
            run_id=run_id,
            query_seq_idx=query_seq_idx,
            window_bp=window_bp,
            step_bp=step_bp,
            max_refs=max_refs,
            priority_seq_idxs=pri_list or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _parse_request_inputs(
    request: Request,
    preset_id: Optional[str],
    checkpoint: str,
    scoring_mode: str,
    null_policy: str,
    flank_window: int,
    min_seg: int,
    beta: float,
    z_thresh: float,
    tf_logit_thresh: float,
    merge_radius_bp: int,
    eval_tolerance_bp: int,
    use_mirror_tta: bool,
    pasted_fasta: Optional[str],
    pasted_gt: Optional[str],
    fasta_file: Optional[UploadFile],
    gt_file: Optional[UploadFile],
) -> tuple[InferenceConfig, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    content_type = request.headers.get("content-type", "")
    fasta_text: Optional[str] = None
    fasta_filename: Optional[str] = None
    gt_text: Optional[str] = None
    gt_filename: Optional[str] = None
    has_upload = False

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
        preset_id = body.get("preset_id", preset_id)
        checkpoint = body.get("checkpoint", checkpoint)
        scoring_mode = body.get("scoring_mode", scoring_mode)
        null_policy = body.get("null_policy", null_policy)
        flank_window = int(body.get("flank_window", flank_window))
        min_seg = int(body.get("min_seg", min_seg))
        beta = float(body.get("beta", beta))
        z_thresh = float(body.get("z_thresh", z_thresh))
        tf_logit_thresh = float(body.get("tf_logit_thresh", tf_logit_thresh))
        merge_radius_bp = int(body.get("merge_radius_bp", merge_radius_bp))
        eval_tolerance_bp = int(body.get("eval_tolerance_bp", eval_tolerance_bp))
        raw_tta = body.get("use_mirror_tta", use_mirror_tta)
        if isinstance(raw_tta, str):
            use_mirror_tta = raw_tta.strip().lower() in ("1", "true", "yes", "on")
        else:
            use_mirror_tta = bool(raw_tta)
        fasta_text = body.get("fasta_text") or body.get("pasted_fasta")
        if "fasta_text" in body or "pasted_fasta" in body:
            has_upload = True
        fasta_filename = body.get("fasta_filename")
        gt_text = body.get("gt_text") or body.get("pasted_gt")
        gt_filename = body.get("gt_filename")
    else:
        if fasta_file is not None and fasta_file.filename:
            has_upload = True
            raw_bytes = await fasta_file.read()
            if len(raw_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Upload exceeds maximum size of 20 MB ({len(raw_bytes) / (1024 * 1024):.1f} MB uploaded).",
                )
            if len(raw_bytes.strip()) == 0:
                raise HTTPException(status_code=400, detail="Empty alignment payload")
            if b"\x00" in raw_bytes:
                raise HTTPException(status_code=400, detail="Binary or non-text alignment content detected")
            try:
                fasta_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail="Binary or non-text alignment content detected"
                ) from exc
            fasta_filename = fasta_file.filename
        elif pasted_fasta is not None:
            has_upload = True
            fasta_text = pasted_fasta
            fasta_filename = "pasted_alignment.fasta"

        if gt_file is not None and gt_file.filename:
            gt_bytes = await gt_file.read()
            gt_text = gt_bytes.decode("utf-8", errors="replace")
            gt_filename = gt_file.filename
        elif pasted_gt and pasted_gt.strip():
            gt_text = pasted_gt
            gt_filename = "pasted_ground_truth.txt"

    if has_upload and (fasta_text is None or not fasta_text.strip()):
        raise HTTPException(status_code=400, detail="Empty alignment payload")
    if not has_upload and not preset_id:
        raise HTTPException(status_code=400, detail="Empty alignment payload")
    if fasta_text and "\x00" in fasta_text:
        raise HTTPException(status_code=400, detail="Binary or non-text alignment content detected")

    cfg = InferenceConfig(
        checkpoint=checkpoint,
        scoring_mode=scoring_mode,  # type: ignore[arg-type]
        null_policy=null_policy,  # type: ignore[arg-type]
        flank_window=flank_window,
        min_seg=min_seg,
        beta=beta,
        z_thresh=z_thresh,
        tf_logit_thresh=tf_logit_thresh,
        merge_radius_bp=merge_radius_bp,
        eval_tolerance_bp=eval_tolerance_bp,
        use_mirror_tta=use_mirror_tta,
    )
    return cfg, preset_id, fasta_text, fasta_filename, gt_text, gt_filename


def _run_job_worker(
    job_id: str,
    cfg: InferenceConfig,
    preset_id: Optional[str],
    fasta_text: Optional[str],
    fasta_filename: Optional[str],
    gt_text: Optional[str],
    gt_filename: Optional[str],
    user: Optional[dict[str, Any]] = None,
) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["error"] = "Job cancelled by user"
            job["finished_at"] = time.time()
            return
        job["status"] = "running"
        job["stage"] = "parsing"
        job["started_at"] = time.time()

    def _progress_cb(stage_label: str) -> None:
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j is not None and j["status"] == "running":
                j["stage"] = stage_label

    def _cancel_check() -> bool:
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            return bool(j and j.get("cancel_requested"))

    try:
        res = engine.run_inference(
            config=cfg,
            preset_id=preset_id,
            fasta_text=fasta_text,
            fasta_filename=fasta_filename,
            gt_text=gt_text,
            gt_filename=gt_filename,
            progress_cb=_progress_cb,
            cancel_check=_cancel_check,
            strict_phase2=True,
        )
        logger.info("Job %s completed: %s", job_id, res.timings.stage_summary_line)
        res_dict = res.model_dump()
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j is not None:
                if j.get("cancel_requested"):
                    j["status"] = "cancelled"
                    j["stage"] = "cancelled"
                    j["error"] = "Job cancelled by user"
                else:
                    j["status"] = "done"
                    j["stage"] = "done"
                    j["result"] = res_dict
                j["finished_at"] = time.time()

        # Persist every run to SQLite and continuously self-train ONLY on user-submitted FASTAs
        try:
            if fasta_text is not None:
                save_run(job_id, user, "upload", fasta_text, res_dict)
                _RETRAIN_EXECUTOR.submit(retrain_on_user_input, engine, job_id, fasta_text, res_dict)
            else:
                save_run(job_id, user, "preset", f"# preset:{preset_id}", res_dict)
        except RuntimeError:
            pass
        except Exception:
            logger.exception("Failed to persist completed run %s", job_id)
    except InterruptedError:
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j is not None:
                j["status"] = "cancelled"
                j["stage"] = "cancelled"
                j["error"] = "Job cancelled by user"
                j["finished_at"] = time.time()
    except Exception as exc:
        logger.exception("Background job %s failed", job_id)
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["stage"] = "error"
                j["error"] = str(exc)
                j["finished_at"] = time.time()


@app.post("/api/jobs")
async def create_job_endpoint(
    request: Request,
    preset_id: Optional[str] = Form(default=None),
    checkpoint: str = Form(default="ckpt_N40000"),
    scoring_mode: str = Form(default="hybrid_flank"),
    null_policy: str = Form(default="duplicate"),
    flank_window: int = Form(default=250),
    min_seg: int = Form(default=50),
    beta: float = Form(default=20.0),
    z_thresh: float = Form(default=6.0),
    tf_logit_thresh: float = Form(default=0.8),
    merge_radius_bp: int = Form(default=75),
    eval_tolerance_bp: int = Form(default=30),
    use_mirror_tta: bool = Form(default=False),
    pasted_fasta: Optional[str] = Form(default=None),
    pasted_gt: Optional[str] = Form(default=None),
    fasta_file: Optional[UploadFile] = File(default=None),
    gt_file: Optional[UploadFile] = File(default=None),
) -> dict[str, Any]:
    """Validate alignment upfront (Phase 2), enqueue background inference job (Phase 1), and return job_id immediately."""
    try:
        cfg, preset_id, fasta_text, fasta_filename, gt_text, gt_filename = await _parse_request_inputs(
            request=request,
            preset_id=preset_id,
            checkpoint=checkpoint,
            scoring_mode=scoring_mode,
            null_policy=null_policy,
            flank_window=flank_window,
            min_seg=min_seg,
            beta=beta,
            z_thresh=z_thresh,
            tf_logit_thresh=tf_logit_thresh,
            merge_radius_bp=merge_radius_bp,
            eval_tolerance_bp=eval_tolerance_bp,
            use_mirror_tta=use_mirror_tta,
            pasted_fasta=pasted_fasta,
            pasted_gt=pasted_gt,
            fasta_file=fasta_file,
            gt_file=gt_file,
        )

        # Upfront Phase 2 validation before queuing
        if fasta_text is not None:
            records, _, _ = parse_raw_fasta_records(fasta_text)
            validate_alignment(records)
            _, seqs_val, _, _ = parse_and_validate_fasta(fasta_text, strict_phase2=True)
            kept, cols, _, _ = engine._select_informative_and_binarise(seqs_val, cfg.null_policy)
            if len(kept) < 10 or len(cols) < 10:
                raise InsufficientInformativeSitesError(found=min(len(kept), len(cols)), required=10)
        else:
            if not preset_id or preset_id not in PRESET_REGISTRY:
                raise HTTPException(status_code=400, detail=f"Unknown preset alignment ID: '{preset_id}'")
    except HTTPException:
        raise
    except AlignmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user = get_user_by_session(request.cookies.get("recomb_session"))
    now_ts = time.time()
    with _JOBS_LOCK:
        _prune_expired_jobs(now_ts)
        active_count = sum(1 for j in _JOBS.values() if j["status"] in ("queued", "running"))
        if active_count >= MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="Server busy, try again in a few minutes")

        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "job_id": job_id,
            "owner_id": user["id"] if user else None,
            "status": "queued",
            "stage": "queued",
            "created_at": now_ts,
            "started_at": None,
            "finished_at": None,
            "cancel_requested": False,
            "result": None,
            "error": None,
        }

    _JOB_EXECUTOR.submit(
        _run_job_worker,
        job_id,
        cfg,
        preset_id,
        fasta_text,
        fasta_filename,
        gt_text,
        gt_filename,
        user,
    )
    return {"job_id": job_id, "status": "queued", "stage": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request) -> dict[str, Any]:
    """Return status, live stage, elapsed seconds, and result/error for a background job."""
    user = get_user_by_session(request.cookies.get("recomb_session"))
    now_ts = time.time()
    with _JOBS_LOCK:
        _prune_expired_jobs(now_ts)
        job = _JOBS.get(job_id)
        if job is None:
            # Fallback to persistent SQLite store if job expired from memory
            if user is not None:
                try:
                    saved = get_run_for_user(job_id, user)
                    if saved is not None:
                        return {
                            "job_id": job_id,
                            "status": "done",
                            "stage": "done",
                            "elapsed_s": round(float(saved["total_ms"]) / 1000.0, 2),
                            "result": saved["result"],
                        }
                except PermissionError as exc:
                    raise HTTPException(status_code=403, detail=str(exc)) from exc
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or expired.")
        owner_id = job.get("owner_id")
        if owner_id is not None and (user is None or (user["id"] != owner_id and user["role"] != "admin")):
            raise HTTPException(status_code=403, detail="Private job belongs to another account.")
        t_ref = job["started_at"] or job["created_at"]
        t_end = job["finished_at"] or now_ts
        elapsed_s = round(max(0.0, float(t_end - t_ref)), 2)
        payload: dict[str, Any] = {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "elapsed_s": elapsed_s,
        }
        if job["result"] is not None:
            payload["result"] = job["result"]
        if job["error"] is not None:
            payload["error"] = job["error"]
        return payload


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Set cancellation flag on a queued or running job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            # Fallback to persistent SQLite store if job expired from memory
            if user is not None:
                try:
                    saved = get_run_for_user(job_id, user)
                    if saved is not None:
                        return {
                            "job_id": job_id,
                            "status": "done",
                            "stage": "done",
                            "elapsed_s": round(float(saved["total_ms"]) / 1000.0, 2),
                            "result": saved["result"],
                        }
                except PermissionError as exc:
                    raise HTTPException(status_code=403, detail=str(exc)) from exc
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or expired.")
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["error"] = "Job cancelled by user"
            job["finished_at"] = time.time()
        return {"job_id": job_id, "status": job["status"], "stage": job["stage"]}


@app.post("/api/infer", response_model=InferenceResponse)
async def run_inference_endpoint(
    request: Request,
    preset_id: Optional[str] = Form(default=None),
    checkpoint: str = Form(default="ckpt_N40000"),
    scoring_mode: str = Form(default="hybrid_flank"),
    null_policy: str = Form(default="duplicate"),
    flank_window: int = Form(default=250),
    min_seg: int = Form(default=50),
    beta: float = Form(default=20.0),
    z_thresh: float = Form(default=6.0),
    tf_logit_thresh: float = Form(default=0.8),
    merge_radius_bp: int = Form(default=75),
    eval_tolerance_bp: int = Form(default=30),
    use_mirror_tta: bool = Form(default=False),
    pasted_fasta: Optional[str] = Form(default=None),
    pasted_gt: Optional[str] = Form(default=None),
    fasta_file: Optional[UploadFile] = File(default=None),
    gt_file: Optional[UploadFile] = File(default=None),
) -> InferenceResponse:
    """Execute live inference from multipart form upload or JSON body."""
    try:
        cfg, preset_id, fasta_text, fasta_filename, gt_text, gt_filename = await _parse_request_inputs(
            request=request,
            preset_id=preset_id,
            checkpoint=checkpoint,
            scoring_mode=scoring_mode,
            null_policy=null_policy,
            flank_window=flank_window,
            min_seg=min_seg,
            beta=beta,
            z_thresh=z_thresh,
            tf_logit_thresh=tf_logit_thresh,
            merge_radius_bp=merge_radius_bp,
            eval_tolerance_bp=eval_tolerance_bp,
            use_mirror_tta=use_mirror_tta,
            pasted_fasta=pasted_fasta,
            pasted_gt=pasted_gt,
            fasta_file=fasta_file,
            gt_file=gt_file,
        )
        res = engine.run_inference(
            config=cfg,
            preset_id=preset_id,
            fasta_text=fasta_text,
            fasta_filename=fasta_filename,
            gt_text=gt_text,
            gt_filename=gt_filename,
        )
        logger.info("Inference completed (%s): %s", res.alignment.dataset_name, res.timings.stage_summary_line)
        return res
    except HTTPException:
        raise
    except AlignmentValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = (
            422
            if any(
                k in detail
                for k in (
                    "identical lengths",
                    "at least 4 sequences",
                    "polymorphic sites",
                    "gap characters",
                )
            )
            else 400
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        logger.exception("Unhandled inference exception")
        raise HTTPException(status_code=500, detail=f"Internal inference error: {str(exc)}") from exc


@app.get("/api/timings")
async def get_timing_history() -> dict[str, Any]:
    """Return recorded live inference timing runs."""
    records = timing_store.list_history()
    return {
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


@app.get("/api/timings/export.csv")
async def export_timings_csv() -> PlainTextResponse:
    """Download all recorded inference timings as a CSV file."""
    csv_text = timing_store.export_csv()
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="gt_transformer_inference_timings.csv"'},
    )


def _load_annotations() -> list[dict[str, Any]]:
    stored = db_list_annotations(200)
    if stored:
        return stored
    if ANNOTATIONS_PATH.exists():
        try:
            return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


@app.get("/api/annotations")
async def list_annotations() -> dict[str, Any]:
    """Return saved reviewer diagnostic feedback annotations."""
    items = _load_annotations()
    return {"count": len(items), "annotations": items}


@app.post("/api/annotations")
async def save_annotation(annotation: FeedbackAnnotation) -> dict[str, Any]:
    """Save a reviewer feedback note on a predicted or missed recombination event."""
    items = _load_annotations()
    entry = annotation.model_dump()
    entry["created_at_iso"] = datetime.now(timezone.utc).isoformat()
    db_add_annotation(entry)
    items.insert(0, entry)
    return {"status": "saved", "annotation": entry, "total_count": len(items)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
