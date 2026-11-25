"""Private user run history and Admin data-collection routes."""
from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from services.db import (
    admin_overview,
    delete_run_for_user,
    get_run_for_user,
    get_user_by_session,
    list_runs_for_admin_export,
    list_user_runs,
)
from services.self_train import get_adaptive_status, reset_adaptive_checkpoint

router = APIRouter(tags=["runs_and_admin"])


def require_user(request: Request) -> dict[str, Any]:
    user = get_user_by_session(request.cookies.get("recomb_session"))
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in first.")
    return user


def require_admin(request: Request) -> dict[str, Any]:
    user = require_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return user


@router.get("/api/runs")
async def list_my_runs(request: Request) -> dict[str, Any]:
    user = get_user_by_session(request.cookies.get("recomb_session"))
    if user is None:
        return {"runs": []}
    return {"runs": list_user_runs(user["id"])}


@router.get("/api/runs/{run_id}")
async def get_my_run(run_id: str, request: Request) -> dict[str, Any]:
    user = require_user(request)
    try:
        run = get_run_for_user(run_id, user)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@router.delete("/api/runs/{run_id}")
async def delete_my_run(run_id: str, request: Request) -> dict[str, Any]:
    user = require_user(request)
    try:
        deleted = delete_run_for_user(run_id, user)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"status": "deleted", "run_id": run_id}


@router.get("/api/admin/overview")
async def get_admin_overview(request: Request) -> dict[str, Any]:
    require_admin(request)
    return admin_overview()


def _build_admin_csv_text(runs: list[dict[str, Any]]) -> str:
    lines = ["run_id,username,dataset_name,source_type,n_sequences,genome_length_bp,n_events,checkpoint,scoring_mode,total_ms,created_at"]
    for r in runs:
        lines.append(
            f"{r['id']},{r['username']},{r['dataset_name']},{r['source_type']},{r['n_sequences']},{r['genome_length_bp']},{r['n_events']},{r['checkpoint']},{r['scoring_mode']},{r['total_ms']:.2f},{r['created_at']}"
        )
    return "\n".join(lines)


@router.get("/api/admin/export.csv")
async def admin_export_csv(request: Request) -> PlainTextResponse:
    require_admin(request)
    ov = admin_overview(limit=1000)
    return PlainTextResponse(
        content=_build_admin_csv_text(ov["runs"]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="all_user_runs.csv"'},
    )


@router.get("/api/admin/export.json")
async def admin_export_json(request: Request) -> JSONResponse:
    require_admin(request)
    return JSONResponse(
        content=admin_overview(limit=1000),
        headers={"Content-Disposition": 'attachment; filename="admin_all_runs.json"'},
    )


@router.get("/api/admin/export.zip")
async def admin_export_zip(request: Request) -> Response:
    require_admin(request)
    ov = admin_overview(limit=1000)
    full_rows = list_runs_for_admin_export(limit=1000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", _build_admin_csv_text(ov["runs"]))
        zf.writestr("overview.json", json.dumps(ov, indent=2))
        for r in full_rows:
            fasta = (r.get("fasta_text") or "").strip()
            if not fasta or fasta.startswith("# preset:"):
                continue
            safe_user = re.sub(r"[^A-Za-z0-9._-]+", "_", str(r.get("username", "anon")))
            safe_ds = re.sub(r"[^A-Za-z0-9._-]+", "_", str(r.get("dataset_name", "dataset")))
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(r.get("id", "run")))
            zf.writestr(f"fastas/{safe_user}_{safe_ds}_{safe_id}.fasta", fasta)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="admin_collected_runs.zip"'},
    )


@router.delete("/api/admin/runs/{run_id}")
async def delete_admin_run(run_id: str, request: Request) -> dict[str, Any]:
    user = require_admin(request)
    deleted = delete_run_for_user(run_id, user)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"status": "deleted", "run_id": run_id}


@router.get("/api/admin/adaptive/status")
async def admin_adaptive_status(request: Request) -> dict[str, Any]:
    require_admin(request)
    return get_adaptive_status()


@router.post("/api/admin/adaptive/reset")
async def admin_adaptive_reset(request: Request) -> dict[str, Any]:
    require_admin(request)
    return reset_adaptive_checkpoint(getattr(request.app.state, "engine", None))


@router.get("/api/admin/runs/{run_id}/fasta")
async def download_admin_run_fasta(run_id: str, request: Request) -> PlainTextResponse:
    user = require_admin(request)
    run = get_run_for_user(run_id, user)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    fname = f"{run['dataset_name']}_{run_id}.fasta"
    return PlainTextResponse(
        content=run["fasta_text"],
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/api/admin/runs/{run_id}/result")
async def get_admin_run_result(run_id: str, request: Request) -> dict[str, Any]:
    user = require_admin(request)
    run = get_run_for_user(run_id, user)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run
