"""Authentication routes: local PBKDF2 accounts + Google OAuth 2.0."""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from services.chat import load_env_file
from services.db import (
    authenticate_user,
    create_session,
    delete_session,
    get_user_by_session,
    register_user,
    upsert_google_user,
)

HERE = Path(__file__).resolve().parents[1]
router = APIRouter(prefix="/api/auth", tags=["auth"])

_RATE_LOCK = threading.Lock()
_AUTH_ATTEMPTS: dict[str, list[float]] = {}


def _check_rate_limit(request: Request, max_attempts: int = 20, window_s: float = 60.0) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with _RATE_LOCK:
        hits = [t for t in _AUTH_ATTEMPTS.get(ip, []) if now - t < window_s]
        if len(hits) >= max_attempts:
            raise HTTPException(status_code=429, detail="Too many authentication attempts; try again in a minute.")
        hits.append(now)
        _AUTH_ATTEMPTS[ip] = hits


@router.get("/me")
async def auth_me(request: Request) -> dict[str, Any]:
    user = get_user_by_session(request.cookies.get("recomb_session"))
    return {"user": user}


@router.post("/register")
async def auth_register(request: Request) -> JSONResponse:
    _check_rate_limit(request)
    body = await request.json()
    try:
        user = register_user(str(body.get("username", "")), str(body.get("password", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = create_session(user["id"])
    resp = JSONResponse({"user": user})
    resp.set_cookie("recomb_session", token, httponly=True, samesite="lax", max_age=7 * 86400)
    return resp


@router.post("/login")
async def auth_login(request: Request) -> JSONResponse:
    _check_rate_limit(request)
    body = await request.json()
    user = authenticate_user(str(body.get("username", "")), str(body.get("password", "")))
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_session(user["id"])
    resp = JSONResponse({"user": user})
    resp.set_cookie("recomb_session", token, httponly=True, samesite="lax", max_age=7 * 86400)
    return resp


@router.post("/logout")
async def auth_logout(request: Request) -> JSONResponse:
    delete_session(request.cookies.get("recomb_session"))
    resp = JSONResponse({"status": "logged_out"})
    resp.delete_cookie("recomb_session")
    return resp


@router.get("/google/status")
async def google_auth_status() -> dict[str, Any]:
    load_env_file(HERE / ".env")
    cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    csec = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    return {"configured": bool(cid and csec)}


@router.get("/google/login")
async def google_auth_login(request: Request):
    load_env_file(HERE / ".env")
    cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    csec = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        raise HTTPException(
            status_code=400,
            detail="Google login not configured yet. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to web/.env.",
        )
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip() or str(request.url_for("google_auth_callback"))
    state_tok = secrets.token_urlsafe(24)
    params = urllib.parse.urlencode(
        {
            "client_id": cid,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state_tok,
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    resp = RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")
    resp.set_cookie("google_oauth_state", state_tok, httponly=True, samesite="lax", max_age=600)
    return resp


@router.get("/google/callback", name="google_auth_callback")
async def google_auth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        raise HTTPException(status_code=400, detail=f"Google OAuth error: {error}")
    expected_state = request.cookies.get("google_oauth_state", "")
    if not state or not expected_state or state != expected_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")

    load_env_file(HERE / ".env")
    cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    csec = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip() or str(request.url_for("google_auth_callback"))

    token_data = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": cid,
            "client_secret": csec,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    token_req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=token_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(token_req, timeout=15) as r:
        tok_json = json.loads(r.read().decode("utf-8"))
    access_token = tok_json["access_token"]

    info_req = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(info_req, timeout=15) as r:
        uinfo = json.loads(r.read().decode("utf-8"))

    user = upsert_google_user(
        email=uinfo.get("email", ""),
        google_sub=uinfo.get("sub", ""),
        display_name=uinfo.get("name", ""),
    )
    sess_token = create_session(user["id"])
    resp = RedirectResponse("/")
    resp.delete_cookie("google_oauth_state")
    resp.set_cookie("recomb_session", sess_token, httponly=True, samesite="lax", max_age=7 * 86400)
    return resp
