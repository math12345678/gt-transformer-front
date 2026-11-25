"""Continuous background self-training exclusively on user-submitted FASTA alignments."""
from __future__ import annotations

import logging
import math
import os
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gt_core import model as M
from services.db import record_retrain_event
from services.fasta_utils import parse_and_validate_fasta

logger = logging.getLogger("uvicorn.error")

WEB_DIR = Path(__file__).resolve().parents[1]
BASE_CKPT = WEB_DIR / "checkpoints" / "ckpt_N10000" / "params.pkl"
ADAPTIVE_DIR = WEB_DIR / "checkpoints" / "ckpt_user_adaptive"
ADAPTIVE_CKPT = ADAPTIVE_DIR / "params.pkl"
_RETRAIN_LOCK = threading.Lock()
_JIT_CACHE_LOCK = threading.Lock()
_EVAL_LOSS_CACHE: dict[M.Config, Any] = {}
_TRAIN_STEP_CACHE: dict[M.Config, Any] = {}
_SHUTDOWN_EVENT = threading.Event()


def _bucket_size(n: int) -> int:
    """Round batch size up to a standard power-of-2 bucket to avoid repeated JAX JIT recompilation."""
    for b in (64, 128, 256, 512, 1024):
        if n <= b:
            return b
    return ((n + 63) // 64) * 64


def _get_masked_loss_and_step(cfg: M.Config) -> tuple[Any, Any]:
    """Return cached JIT-compiled `(eval_masked_loss, train_step)` functions for `cfg`."""
    with _JIT_CACHE_LOCK:
        eval_fn = _EVAL_LOSS_CACHE.get(cfg)
        step_fn = _TRAIN_STEP_CACHE.get(cfg)
        if eval_fn is not None and step_fn is not None:
            return eval_fn, step_fn

        @jax.jit
        def eval_masked_loss(
            params: Any, x_pad: jnp.ndarray, y_pad: jnp.ndarray, mask_pad: jnp.ndarray
        ) -> jnp.ndarray:
            logits = M.forward(params, x_pad, cfg)
            w = jnp.where(y_pad > 0.5, cfg.pos_weight, 1.0) * mask_pad
            return (M.bce_logits(logits, y_pad) * w).sum() / jnp.maximum(w.sum(), 1e-6)

        @jax.jit
        def train_step(
            params: Any,
            orig_params: Any,
            x_pad: jnp.ndarray,
            y_pad: jnp.ndarray,
            mask_pad: jnp.ndarray,
            lr: jnp.ndarray,
            anchor_weight: jnp.ndarray,
        ) -> Any:
            def _anchored(p: Any) -> jnp.ndarray:
                base = eval_masked_loss(p, x_pad, y_pad, mask_pad)
                l2_anchor = sum(
                    jnp.sum((a - b) ** 2)
                    for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(orig_params))
                )
                return base + anchor_weight * l2_anchor

            _, grads = jax.value_and_grad(_anchored)(params)
            g_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
            scale = jnp.minimum(1.0, 1.0 / jnp.maximum(g_norm, 1e-6))
            return jax.tree_util.tree_map(lambda p, g: p - lr * scale * g, params, grads)

        _EVAL_LOSS_CACHE[cfg] = eval_masked_loss
        _TRAIN_STEP_CACHE[cfg] = train_step
        return eval_masked_loss, train_step


def request_shutdown() -> None:
    """Signal queued or in-flight self-training jobs to exit promptly on app shutdown."""
    _SHUTDOWN_EVENT.set()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a unique staging file, fsync, and atomically rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _ensure_adaptive_ckpt() -> Path:
    if not ADAPTIVE_CKPT.exists():
        _atomic_write_bytes(ADAPTIVE_CKPT, BASE_CKPT.read_bytes())
    return ADAPTIVE_CKPT


def retrain_on_user_input(
    engine: Any,
    run_id: str,
    fasta_text: str,
    result_dict: dict[str, Any],
) -> None:
    """Fine-tune `ckpt_user_adaptive` on high-confidence pseudo-labels from a user-uploaded alignment."""
    if _SHUTDOWN_EVENT.is_set():
        return
    ds_name = result_dict.get("alignment", {}).get("dataset_name", "user_input")
    try:
        _, seqs, _, _ = parse_and_validate_fasta(fasta_text, strict_phase2=True)
        kept, cols, origin, _ = engine._select_informative_and_binarise(seqs, "duplicate")
        M_tx = len(cols)
        if M_tx < 314:
            record_retrain_event(
                run_id,
                ds_name,
                "skipped",
                notes=f"M_tx={M_tx} < 314 (too short for transformer window training)",
            )
            return

        haplo = np.array([[1 if v == "1" else 0 for v in col] for col in cols], dtype=np.uint8)
        X = engine._build_circular_windows_freq(haplo, 314)  # default W=314 check after loading cfg

        # Build sharp Gaussian soft targets centered at high-confidence breakpoints
        evs = result_dict.get("consolidated_events", [])
        pos_targets: dict[int, float] = {}
        center_bps: list[int] = []
        for ev in evs:
            b = int(ev.get("boundary_tx", -1))
            z = float(ev.get("score_z", 0.0))
            p = float(ev.get("tf_prob", 0.0))
            if 0 <= b < M_tx and (z >= 6.0 or p >= 0.65):
                center_bps.append(b)
                pos_targets[b] = 1.0
                if b - 1 >= 0:
                    pos_targets[b - 1] = max(pos_targets.get(b - 1, 0.0), 0.25)
                if b + 1 < M_tx:
                    pos_targets[b + 1] = max(pos_targets.get(b + 1, 0.0), 0.25)

        if not pos_targets:
            record_retrain_event(
                run_id,
                ds_name,
                "skipped",
                notes="No high-confidence positive breakpoints to train on",
            )
            return

        pos_idx = np.asarray(sorted(pos_targets.keys()), dtype=np.int64)
        pos_y = np.asarray([pos_targets[int(i)] for i in pos_idx], dtype=np.float32)

        # Hard negatives near events (12..45 cols away, outside +-10 exclusion) + distant background (>= 25 cols away)
        excl_near = np.zeros(M_tx, dtype=bool)
        hard_band = np.zeros(M_tx, dtype=bool)
        far_excl = np.zeros(M_tx, dtype=bool)
        for b in center_bps:
            hard_band[max(0, b - 45) : min(M_tx, b + 46)] = True
            excl_near[max(0, b - 10) : min(M_tx, b + 11)] = True
            far_excl[max(0, b - 25) : min(M_tx, b + 26)] = True
        for b in pos_idx:
            excl_near[int(b)] = True
            far_excl[int(b)] = True

        hard_neg_idx = np.flatnonzero(hard_band & (~excl_near))
        far_neg_idx = np.flatnonzero(~far_excl)
        if len(far_neg_idx) == 0 and len(hard_neg_idx) == 0:
            record_retrain_event(
                run_id,
                ds_name,
                "skipped",
                n_positive=int(len(pos_idx)),
                notes="No confident background negative sites available",
            )
            return

        rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
        n_neg_target = max(32, len(pos_idx) * 8)
        n_hard_want = int(round(0.35 * n_neg_target)) if len(hard_neg_idx) > 0 else 0
        hard_chosen = (
            rng.choice(hard_neg_idx, size=min(len(hard_neg_idx), n_hard_want), replace=False)
            if n_hard_want > 0
            else np.array([], dtype=np.int64)
        )
        n_far_want = max(0, n_neg_target - len(hard_chosen))
        far_chosen = (
            rng.choice(far_neg_idx, size=min(len(far_neg_idx), n_far_want), replace=False)
            if len(far_neg_idx) > 0 and n_far_want > 0
            else np.array([], dtype=np.int64)
        )
        neg_chosen = np.concatenate([hard_chosen, far_chosen]).astype(np.int64)

        sel_idx = np.concatenate([pos_idx, neg_chosen])
        y_single = np.concatenate([pos_y, np.zeros(len(neg_chosen), dtype=np.float32)]).astype(np.float32)

        with _RETRAIN_LOCK:
            ckpt_path = _ensure_adaptive_ckpt()
            with open(ckpt_path, "rb") as fh:
                payload = pickle.load(fh)
            cfg = M.config_from_dict(payload.get("cfg", {}))
            if cfg.W != X.shape[1]:
                X = engine._build_circular_windows_freq(haplo, cfg.W)

            # Exact reverse-genome mirror augmentation doubles effective adaptation signal
            X_rev = engine._build_circular_windows_freq(haplo[::-1].copy(), cfg.W)
            mir_idx = (M_tx - sel_idx) % M_tx
            x_sel = np.concatenate([X[sel_idx], X_rev[mir_idx]], axis=0).astype(np.float32)
            y_raw = np.concatenate([y_single, y_single], axis=0).astype(np.float32)

            n_real = len(y_raw)
            bucket = _bucket_size(n_real)
            x_pad_np = np.zeros((bucket, x_sel.shape[1], x_sel.shape[2]), dtype=np.float32)
            y_pad_np = np.zeros((bucket,), dtype=np.float32)
            mask_np = np.zeros((bucket,), dtype=np.float32)
            x_pad_np[:n_real] = x_sel
            y_pad_np[:n_real] = y_raw
            mask_np[:n_real] = 1.0

            X_pad = jnp.asarray(x_pad_np, dtype=jnp.float32)
            y_pad = jnp.asarray(y_pad_np, dtype=jnp.float32)
            mask_pad = jnp.asarray(mask_np, dtype=jnp.float32)

            params = jax.tree_util.tree_map(lambda a: jnp.asarray(a), payload["params"])
            orig_params = params
            eval_fn, step_fn = _get_masked_loss_and_step(cfg)

            lr = jnp.float32(5e-5)
            anchor_w = jnp.float32(1e-3)
            n_steps = 6
            loss_before = float(eval_fn(params, X_pad, y_pad, mask_pad))

            for _ in range(n_steps):
                if _SHUTDOWN_EVENT.is_set():
                    return
                params = step_fn(params, orig_params, X_pad, y_pad, mask_pad, lr, anchor_w)

            if _SHUTDOWN_EVENT.is_set():
                return
            loss_after = float(eval_fn(params, X_pad, y_pad, mask_pad))
            orig_sq = sum(float(jnp.sum(o * o)) for o in jax.tree_util.tree_leaves(orig_params))
            delta_sq = sum(
                float(jnp.sum((a - b) ** 2))
                for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(orig_params))
            )
            rel_drift = math.sqrt(delta_sq / max(orig_sq, 1e-12))
            if (
                not math.isfinite(loss_after)
                or rel_drift > 0.05
                or (math.isfinite(loss_before) and loss_after > loss_before * 1.05)
            ):
                record_retrain_event(
                    run_id=run_id,
                    dataset_name=ds_name,
                    status="rejected",
                    loss_before=round(loss_before, 5) if math.isfinite(loss_before) else None,
                    loss_after=round(loss_after, 5) if math.isfinite(loss_after) else None,
                    steps=n_steps,
                    n_positive=int(len(pos_idx)),
                    n_negative=int(len(neg_chosen)),
                    notes=f"Rejected update (rel_drift={rel_drift:.4f}, loss {loss_before:.4f}->{loss_after:.4f})",
                )
                return

            payload["params"] = jax.tree_util.tree_map(lambda a: np.asarray(a), params)
            meta = dict(payload.get("meta", {}))
            meta["updates"] = int(meta.get("updates", 0)) + 1
            meta["last_run_id"] = run_id
            meta["last_updated_iso"] = datetime.now(timezone.utc).isoformat()
            payload["meta"] = meta
            _atomic_write_bytes(ckpt_path, pickle.dumps(payload))

            # Invalidate engine cache so next run picks up updated adaptive checkpoint
            engine._ckpt_cache.pop("ckpt_user_adaptive", None)
            engine._preset_meta_cache = None

        record_retrain_event(
            run_id=run_id,
            dataset_name=ds_name,
            status="updated",
            loss_before=round(loss_before, 5) if loss_before is not None else None,
            loss_after=round(loss_after, 5),
            steps=n_steps,
            n_positive=int(len(pos_idx)),
            n_negative=int(len(neg_chosen)),
            notes=f"Updated ckpt_user_adaptive from user input ({len(pos_idx)} pos, {len(neg_chosen)} neg windows)",
        )
        logger.info(
            "Self-trained ckpt_user_adaptive on user run %s (%s): loss %.4f -> %.4f",
            run_id,
            ds_name,
            loss_before or 0.0,
            loss_after,
        )
    except Exception as exc:
        logger.exception("Self-training error on run %s", run_id)
        record_retrain_event(run_id, ds_name, "error", notes=str(exc)[:300])


def _read_adaptive_status_unlocked() -> dict[str, Any]:
    if not ADAPTIVE_CKPT.exists():
        return {
            "exists": False,
            "updates": 0,
            "last_run_id": None,
            "last_updated_iso": None,
            "size_kb": 0.0,
        }
    size_kb = round(ADAPTIVE_CKPT.stat().st_size / 1024.0, 1)
    try:
        with open(ADAPTIVE_CKPT, "rb") as fh:
            payload = pickle.load(fh)
        meta = dict(payload.get("meta") or {})
        return {
            "exists": True,
            "updates": int(meta.get("updates", 0)),
            "last_run_id": meta.get("last_run_id"),
            "last_updated_iso": meta.get("last_updated_iso"),
            "size_kb": size_kb,
        }
    except Exception:
        return {
            "exists": True,
            "updates": 0,
            "last_run_id": None,
            "last_updated_iso": None,
            "size_kb": size_kb,
        }


def get_adaptive_status() -> dict[str, Any]:
    """Return metadata about the user-adaptive checkpoint file."""
    with _RETRAIN_LOCK:
        return _read_adaptive_status_unlocked()


def reset_adaptive_checkpoint(engine: Any) -> dict[str, Any]:
    """Reset `ckpt_user_adaptive` back to the base `ckpt_N10000` weights."""
    with _RETRAIN_LOCK:
        _atomic_write_bytes(ADAPTIVE_CKPT, BASE_CKPT.read_bytes())
        if engine is not None:
            engine._ckpt_cache.pop("ckpt_user_adaptive", None)
            engine._preset_meta_cache = None
        record_retrain_event(
            run_id="admin_reset",
            dataset_name="ckpt_user_adaptive",
            status="reset",
            notes="Reset adaptive weights to base ckpt_N10000",
        )
        return _read_adaptive_status_unlocked()
