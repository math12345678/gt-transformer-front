"""Persistent store for private accounts, sessions, runs, timings and annotations.

Uses Supabase Postgres when ``SUPABASE_DB_DSN`` (or ``DATABASE_URL``) is set,
and falls back to a local SQLite file for offline development and tests.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parents[1] / "recomb_app.db"
_DB_LOCK = threading.Lock()


def _dsn() -> str:
    return os.environ.get("SUPABASE_DB_DSN") or os.environ.get("DATABASE_URL") or ""


def uses_postgres() -> bool:
    return bool(_dsn())


# ---------------------------------------------------------------------------
# Minimal psycopg adapter: the original SQLite call sites stay unchanged.
# ---------------------------------------------------------------------------
_INSERT_OR_REPLACE = re.compile(
    r"^\s*INSERT\s+OR\s+REPLACE\s+INTO\s+([\w.]+)\s*\(([^)]*)\)\s*VALUES\s*(\([^)]*\))\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _pg_sql(sql: str) -> str:
    """Convert the small subset of SQLite SQL used here into Postgres SQL."""
    sql = re.sub(r"\?", "%s", sql)
    sql = re.sub(r"\bAUTOINCREMENT\b", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bINTEGER\s+PRIMARY\s+KEY\b", "SERIAL PRIMARY KEY", sql, flags=re.IGNORECASE)
    match = _INSERT_OR_REPLACE.match(sql)
    if match:
        table, columns_raw, values = match.groups()
        columns = [c.strip() for c in columns_raw.split(",")]
        conflict = "id" if "id" in columns else columns[0]
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != conflict)
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES {values} "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        )
    return sql


class _PgResult:
    def __init__(self, cursor: Any, lastrowid: int = 0) -> None:
        self._cursor = cursor
        self.lastrowid = lastrowid

    def fetchone(self) -> Optional[dict[str, Any]]:
        if self._cursor is None:
            return None
        row = self._cursor.fetchone()
        return _pg_dict(row, self._cursor.description) if row is not None else None

    def fetchall(self) -> list[dict[str, Any]]:
        if self._cursor is None:
            return []
        return [_pg_dict(row, self._cursor.description) for row in self._cursor.fetchall()]


def _pg_dict(row: Any, description: Any) -> dict[str, Any]:
    return {col.name: row[idx] for idx, col in enumerate(description)}


class _PgConnection:
    """Thin psycopg wrapper exposing the few sqlite3 methods this module uses."""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._conn = psycopg.connect(dsn, connect_timeout=15)

    def execute(self, sql: str, params: Any = ()) -> _PgResult:
        if sql.strip().upper().startswith("PRAGMA"):
            return _PgResult(None)
        sql = _pg_sql(sql)
        # Serial-id inserts need their generated id; RETURNING avoids a second
        # statement (and avoids aborting the transaction with lastval()).
        wants_id = bool(
            re.match(r"^\s*INSERT\s+INTO\s+(users|reviewer_annotations)\b", sql, re.IGNORECASE)
            and "ON CONFLICT" not in sql.upper()
            and "RETURNING" not in sql.upper()
        )
        if wants_id:
            sql = sql.rstrip() + " RETURNING id"
        cursor = self._conn.execute(sql, tuple(params or ()))
        lastrowid = 0
        if wants_id:
            row = cursor.fetchone()
            if row is not None:
                lastrowid = int(row[0])
        return _PgResult(cursor, lastrowid)

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            statement = statement.strip()
            if statement and not statement.upper().startswith("PRAGMA"):
                self._conn.execute(_pg_sql(statement))

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _connect() -> Any:
    if uses_postgres():
        return _PgConnection(_dsn())
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _hash_password(password: str, salt_hex: Optional[str] = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$", 1)
    except ValueError:
        return False
    expected = _hash_password(password, salt_hex).split("$", 1)[1]
    return hmac.compare_digest(expected, dk_hex)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    email TEXT DEFAULT '',
    auth_provider TEXT NOT NULL DEFAULT 'local',
    google_sub TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username TEXT NOT NULL DEFAULT 'anonymous',
    dataset_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    fasta_text TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    scoring_mode TEXT NOT NULL,
    n_sequences INTEGER NOT NULL,
    genome_length_bp INTEGER NOT NULL,
    n_events INTEGER NOT NULL,
    total_ms REAL NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    timings_run_id TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS retrain_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    status TEXT NOT NULL,
    loss_before REAL,
    loss_after REAL,
    steps INTEGER NOT NULL DEFAULT 0,
    n_positive INTEGER NOT NULL DEFAULT 0,
    n_negative INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timing_history (
    run_id TEXT PRIMARY KEY,
    timestamp_iso TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviewer_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_user_created ON runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_timings_run_id ON runs(timings_run_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_retrain_run ON retrain_log(run_id);
CREATE INDEX IF NOT EXISTS idx_timing_created ON timing_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_annotations_created ON reviewer_annotations(created_at DESC);
"""

_PG_MIGRATIONS_SQL = """
ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT DEFAULT '';
ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider TEXT NOT NULL DEFAULT 'local';
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub TEXT DEFAULT '';
ALTER TABLE runs ADD COLUMN IF NOT EXISTS timings_run_id TEXT DEFAULT '';
"""


def init_db() -> None:
    """Create tables if missing and seed the admin account from environment variables."""
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            if uses_postgres():
                conn.executescript(_PG_MIGRATIONS_SQL)
            else:
                # Safe schema migrations for existing SQLite DBs
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
                if "email" not in cols:
                    conn.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
                if "auth_provider" not in cols:
                    conn.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'local'")
                if "google_sub" not in cols:
                    conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT DEFAULT ''")
                run_cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
                if "timings_run_id" not in run_cols:
                    conn.execute("ALTER TABLE runs ADD COLUMN timings_run_id TEXT DEFAULT ''")

            admin_user = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
            admin_pass = os.environ.get("ADMIN_PASSWORD", "admin-recomb-2026").strip() or "admin-recomb-2026"
            row = conn.execute("SELECT id FROM users WHERE username = ?", (admin_user,)).fetchone()
            now = datetime.now(timezone.utc).isoformat()
            if row is None:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
                    (admin_user, _hash_password(admin_pass), now),
                )
            else:
                conn.execute(
                    "UPDATE users SET role = 'admin' WHERE id = ?",
                    (row["id"],),
                )
            conn.commit()
        finally:
            conn.close()


def register_user(username: str, password: str) -> dict[str, Any]:
    uname = username.strip()
    if len(uname) < 2 or len(uname) > 40:
        raise ValueError("Username must be 2–40 characters.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")
    now = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn = _connect()
        try:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (uname,)).fetchone()
            if existing is not None:
                raise ValueError(f"Username '{uname}' is already taken.")
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'user', ?)",
                (uname, _hash_password(password), now),
            )
            uid = int(cur.lastrowid)
            conn.commit()
            return {"id": uid, "username": uname, "role": "user", "created_at": now}
        finally:
            conn.close()


def authenticate_user(username: str, password: str) -> Optional[dict[str, Any]]:
    uname = username.strip()
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
                (uname,),
            ).fetchone()
            if row is None or not _verify_password(password, row["password_hash"]):
                return None
            return {
                "id": int(row["id"]),
                "username": row["username"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
        finally:
            conn.close()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    exp = now + timedelta(days=7)
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now_iso,))
            conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, now_iso, exp.isoformat()),
            )
            conn.commit()
            return token
        finally:
            conn.close()


def get_user_by_session(token: Optional[str]) -> Optional[dict[str, Any]]:
    if not token:
        return None
    now_iso = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.role, u.email, u.auth_provider, u.created_at, s.expires_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            ).fetchone()
            if row is None or row["expires_at"] < now_iso:
                return None
            return {
                "id": int(row["id"]),
                "username": row["username"],
                "role": row["role"],
                "email": row["email"] or "",
                "auth_provider": row["auth_provider"] or "local",
                "created_at": row["created_at"],
            }
        finally:
            conn.close()


def delete_session(token: Optional[str]) -> None:
    if not token:
        return
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        finally:
            conn.close()


def save_run(
    run_id: str,
    user: Optional[dict[str, Any]],
    source_type: str,
    fasta_text: str,
    result_dict: dict[str, Any],
) -> None:
    aln = result_dict["alignment"]
    cfg = result_dict["config"]
    tim = result_dict["timings"]
    evs = result_dict.get("consolidated_events", [])
    now = datetime.now(timezone.utc).isoformat()
    uid = user["id"] if user else None
    uname = user["username"] if user else "anonymous"
    timings_run_id = str(tim.get("run_id", ""))
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, user_id, username, dataset_name, source_type, fasta_text, checkpoint, scoring_mode,
                 n_sequences, genome_length_bp, n_events, total_ms, result_json, created_at, timings_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    uid,
                    uname,
                    aln["dataset_name"],
                    source_type,
                    fasta_text,
                    cfg["checkpoint"],
                    cfg["scoring_mode"],
                    int(aln["n_sequences"]),
                    int(aln["genome_length_bp"]),
                    len(evs),
                    float(tim["total_ms"]),
                    json.dumps(result_dict),
                    now,
                    timings_run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def list_user_runs(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, dataset_name, source_type, checkpoint, scoring_mode,
                       n_sequences, genome_length_bp, n_events, total_ms, created_at
                FROM runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_run_for_user(run_id: str, user: dict[str, Any]) -> Optional[dict[str, Any]]:
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            if user["role"] != "admin" and row["user_id"] != user["id"]:
                raise PermissionError("Access denied: this run belongs to another account.")
            d = dict(row)
            d["result"] = json.loads(d.pop("result_json"))
            return d
        finally:
            conn.close()


def get_run_by_any_id(run_or_job_id: str, user: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Look up a run by job id or `timings_run_id`, enforcing account privacy."""
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ? OR timings_run_id = ? ORDER BY created_at DESC LIMIT 1",
                (run_or_job_id, run_or_job_id),
            ).fetchone()
            if row is None:
                return None
            if row["user_id"] is not None:
                if user is None or (user["role"] != "admin" and row["user_id"] != user["id"]):
                    raise PermissionError("Access denied: this run belongs to another account.")
            d = dict(row)
            d["result"] = json.loads(d.pop("result_json"))
            return d
        finally:
            conn.close()


def upsert_google_user(email: str, google_sub: str, display_name: str = "") -> dict[str, Any]:
    """Find or create a private user account for a Google OAuth login."""
    clean_email = email.strip().lower()
    base_name = (display_name.strip() or clean_email.split("@")[0] or "google_user")[:32]
    now = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, username, role, email, auth_provider, created_at FROM users WHERE google_sub = ? OR (email != '' AND email = ?)",
                (google_sub, clean_email),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE users SET google_sub = ?, email = ?, auth_provider = 'google' WHERE id = ?",
                    (google_sub, clean_email, row["id"]),
                )
                conn.commit()
                return {
                    "id": int(row["id"]),
                    "username": row["username"],
                    "role": row["role"],
                    "email": clean_email,
                    "auth_provider": "google",
                    "created_at": row["created_at"],
                }

            uname = base_name
            suffix = 1
            while conn.execute("SELECT 1 FROM users WHERE username = ?", (uname,)).fetchone() is not None:
                uname = f"{base_name}_{suffix}"
                suffix += 1
            cur = conn.execute(
                """
                INSERT INTO users (username, password_hash, role, email, auth_provider, google_sub, created_at)
                VALUES (?, '', 'user', ?, 'google', ?, ?)
                """,
                (uname, clean_email, google_sub, now),
            )
            uid = int(cur.lastrowid)
            conn.commit()
            return {
                "id": uid,
                "username": uname,
                "role": "user",
                "email": clean_email,
                "auth_provider": "google",
                "created_at": now,
            }
        finally:
            conn.close()


def delete_run_for_user(run_id: str, user: dict[str, Any]) -> bool:
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT user_id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return False
            if user["role"] != "admin" and row["user_id"] != user["id"]:
                raise PermissionError("Access denied.")
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            return True
        finally:
            conn.close()


def list_runs_for_admin_export(limit: int = 1000) -> list[dict[str, Any]]:
    """Return full stored run rows (including `fasta_text`) for admin bulk archive export."""
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, username, dataset_name, source_type, fasta_text, checkpoint,
                       scoring_mode, n_sequences, genome_length_bp, n_events, total_ms, created_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def admin_overview(limit: int = 200) -> dict[str, Any]:
    with _DB_LOCK:
        conn = _connect()
        try:
            users = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT u.id, u.username, u.role, u.email, u.auth_provider, u.created_at,
                           COUNT(r.id) AS run_count
                    FROM users u
                    LEFT JOIN runs r ON r.user_id = u.id
                    GROUP BY u.id
                    ORDER BY u.id ASC
                    """
                ).fetchall()
            ]
            runs = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT id, user_id, username, dataset_name, source_type, checkpoint, scoring_mode,
                           n_sequences, genome_length_bp, n_events, total_ms, created_at
                    FROM runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            ]
            retrains = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM retrain_log ORDER BY id DESC LIMIT 100"
                ).fetchall()
            ]
            stats_row = conn.execute(
                """
                SELECT COUNT(*) AS total_runs,
                       SUM(CASE WHEN source_type = 'upload' THEN 1 ELSE 0 END) AS total_uploads,
                       AVG(total_ms) AS avg_ms
                FROM runs
                """
            ).fetchone()
            return {
                "users": users,
                "runs": runs,
                "retrain_log": retrains,
                "stats": {
                    "total_users": len(users),
                    "total_runs": int(stats_row["total_runs"] or 0),
                    "total_uploads": int(stats_row["total_uploads"] or 0),
                    "avg_ms": round(float(stats_row["avg_ms"] or 0.0), 1),
                    "retrain_updates": sum(1 for x in retrains if x["status"] == "updated"),
                },
            }
        finally:
            conn.close()


def record_retrain_event(
    run_id: str,
    dataset_name: str,
    status: str,
    loss_before: Optional[float] = None,
    loss_after: Optional[float] = None,
    steps: int = 0,
    n_positive: int = 0,
    n_negative: int = 0,
    notes: str = "",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO retrain_log
                (run_id, dataset_name, status, loss_before, loss_after, steps, n_positive, n_negative, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, dataset_name, status, loss_before, loss_after, steps, n_positive, n_negative, notes, now),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Timing history & reviewer annotations (same backend as accounts/runs)
# ---------------------------------------------------------------------------
def append_timing_report(report: dict[str, Any]) -> None:
    """Store one live inference timing report, replacing a repeated run id."""
    now = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO timing_history
                (run_id, timestamp_iso, report_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(report.get("run_id", "")),
                    str(report.get("timestamp_iso", now)),
                    json.dumps(report),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def list_timing_reports(limit: int = 150) -> list[dict[str, Any]]:
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT report_json FROM timing_history
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                try:
                    result.append(json.loads(row["report_json"]))
                except Exception:
                    continue
            return result
        finally:
            conn.close()


def add_annotation(entry: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO reviewer_annotations (entry_json, created_at) VALUES (?, ?)",
                (json.dumps(entry), now),
            )
            conn.commit()
        finally:
            conn.close()


def list_annotations(limit: int = 200) -> list[dict[str, Any]]:
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT entry_json FROM reviewer_annotations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                try:
                    result.append(json.loads(row["entry_json"]))
                except Exception:
                    continue
            return result
        finally:
            conn.close()
