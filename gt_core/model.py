"""
The window transformer (S6 / M3 scaffold).

One training example is a query column.  Input is the circular pair-window
`(W, 4)` of gamete counts (self-pair at the centre); output is a single logit for
the boundary immediately to the left of that column.

This is deliberately small and explicit - a hand-rolled pre-norm encoder in pure
JAX, no Flax.  PyTorch is the obvious first choice for this kind of model, but
there is no torch wheel for the Python 3.15 interpreter in this environment
(`pip` finds nothing), so the scaffold is written against JAX + optax, which are
available.  The parameter tree is plain nested dicts/lists of arrays, so porting
to `torch.nn.Module` is mechanical if that is preferred.
"""
from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Config(NamedTuple):
    W: int = 314
    centre: int = 157          # index of offset 0 in the window
    n_in: int = 4              # gamete counts
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    # BCE positive-class weight.  With the default balanced batches
    # (--pos-frac 0.5) this must stay at 1.0; it is only useful when sampling
    # naturally, where the base rate is ~1 in 476.
    pos_weight: float = 1.0
    arch_version: int = 1      # 1: original quartet encoder, 2: LD-invariant Conv+Contrast encoder


def config_from_dict(d):
    """Construct Config safely from a checkpoint dict, ignoring unknown/legacy keys."""
    if not d:
        return Config()
    valid = set(Config._fields)
    return Config(**{k: v for k, v in d.items() if k in valid})


# ------------------------------------------------------------------ features
def enrich_gamete_features(x):
    """Expand `(..., W, 4)` gamete frequencies `[p00, p01, p10, p11]` to 10 LD-invariant channels."""
    p00 = x[..., 0]
    p01 = x[..., 1]
    p10 = x[..., 2]
    p11 = x[..., 3]
    p_a = jnp.clip(p10 + p11, 1e-6, 1.0 - 1e-6)
    p_b = jnp.clip(p01 + p11, 1e-6, 1.0 - 1e-6)
    fg = ((p00 > 1e-5) & (p01 > 1e-5) & (p10 > 1e-5) & (p11 > 1e-5)).astype(jnp.float32)
    min_g = jnp.minimum(jnp.minimum(p00, p01), jnp.minimum(p10, p11))
    d_val = p00 * p11 - p01 * p10
    denom_r2 = jnp.maximum(p_a * (1.0 - p_a) * p_b * (1.0 - p_b), 1e-6)
    r2 = jnp.clip((d_val * d_val) / denom_r2, 0.0, 1.0)
    dmax_pos = jnp.minimum(p_a * (1.0 - p_b), (1.0 - p_a) * p_b)
    dmax_neg = jnp.minimum(p_a * p_b, (1.0 - p_a) * (1.0 - p_b))
    dmax = jnp.where(d_val >= 0.0, dmax_pos, dmax_neg)
    d_prime = jnp.clip(jnp.abs(d_val) / jnp.maximum(dmax, 1e-6), 0.0, 1.0)
    eps = 1e-6
    h_joint = -(
        p00 * jnp.log(p00 + eps)
        + p01 * jnp.log(p01 + eps)
        + p10 * jnp.log(p10 + eps)
        + p11 * jnp.log(p11 + eps)
    )
    h_a = -(p_a * jnp.log(p_a + eps) + (1.0 - p_a) * jnp.log(1.0 - p_a + eps))
    h_b = -(p_b * jnp.log(p_b + eps) + (1.0 - p_b) * jnp.log(1.0 - p_b + eps))
    nmi = jnp.clip((h_a + h_b - h_joint) / jnp.maximum(jnp.minimum(h_a, h_b), 1e-4), 0.0, 1.0)
    return jnp.stack([p00, p01, p10, p11, fg, min_g, d_val, r2, d_prime, nmi], axis=-1)


# ------------------------------------------------------------------ init
def _linear(key, din, dout, scale=None):
    scale = scale if scale is not None else 1.0 / (din ** 0.5)
    return {"w": jax.random.normal(key, (din, dout)) * scale,
            "b": jnp.zeros((dout,))}


def init_params(key, cfg):
    arch = int(getattr(cfg, "arch_version", 1))
    if arch == 2:
        n_keys = 8 + 4 * cfg.n_layers
        keys = jax.random.split(key, n_keys)
        i = 0
        p = {}
        p["embed"] = _linear(keys[i], 10, cfg.d_model)
        i += 1
        p["pos"] = jax.random.normal(keys[i], (cfg.W, cfg.d_model)) * 0.02
        i += 1
        p["dw_conv_w"] = jax.random.normal(keys[i], (5, cfg.d_model)) * 0.1
        i += 1
        p["dw_conv_b"] = jnp.zeros((cfg.d_model,))
        p["conv_ln"] = {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))}
        layers = []
        for _ in range(cfg.n_layers):
            qkv = _linear(keys[i], cfg.d_model, 3 * cfg.d_model)
            i += 1
            attn_out = _linear(keys[i], cfg.d_model, cfg.d_model)
            i += 1
            ff1 = _linear(keys[i], cfg.d_model, cfg.d_ff)
            i += 1
            ff2 = _linear(keys[i], cfg.d_ff, cfg.d_model)
            i += 1
            layers.append({
                "ln1": {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))},
                "qkv": qkv,
                "attn_out": attn_out,
                "ln2": {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))},
                "ff1": ff1,
                "ff2": ff2,
            })
        p["layers"] = layers
        p["ln_f"] = {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))}
        p["head1"] = _linear(keys[i], 6 * cfg.d_model, cfg.d_model)
        i += 1
        p["head2"] = _linear(keys[i], cfg.d_model, 1)
        i += 1
        assert i <= n_keys, (i, n_keys)
        return p

    n_keys = 3 + 4 * cfg.n_layers
    keys = jax.random.split(key, n_keys)
    i = 0
    p = {}
    p["embed"] = _linear(keys[i], cfg.n_in, cfg.d_model)
    i += 1
    p["pos"] = jax.random.normal(keys[i], (cfg.W, cfg.d_model)) * 0.02
    i += 1
    layers = []
    for _ in range(cfg.n_layers):
        qkv = _linear(keys[i], cfg.d_model, 3 * cfg.d_model)
        i += 1
        attn_out = _linear(keys[i], cfg.d_model, cfg.d_model)
        i += 1
        ff1 = _linear(keys[i], cfg.d_model, cfg.d_ff)
        i += 1
        ff2 = _linear(keys[i], cfg.d_ff, cfg.d_model)
        i += 1
        layers.append({
            "ln1": {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))},
            "qkv": qkv,
            "attn_out": attn_out,
            "ln2": {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))},
            "ff1": ff1,
            "ff2": ff2,
        })
    p["layers"] = layers
    p["head"] = _linear(keys[i], cfg.d_model, 1)
    i += 1
    assert i <= n_keys, (i, n_keys)
    return p


# ------------------------------------------------------------------ forward
def _ln(x, p, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * p["w"] + p["b"]


def _lin(x, p):
    return x @ p["w"] + p["b"]


def _block(x, lp, cfg):
    h = _ln(x, lp["ln1"])
    B, W, D = h.shape
    H, dh = cfg.n_heads, D // cfg.n_heads
    q, k, v = jnp.split(_lin(h, lp["qkv"]), 3, axis=-1)

    def heads(t):
        return t.reshape(B, W, H, dh).transpose(0, 2, 1, 3)   # (B, H, W, dh)

    q, k, v = heads(q), heads(k), heads(v)
    att = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(dh), -1)
    o = jnp.einsum("bhqk,bhkd->bhqd", att, v).transpose(0, 2, 1, 3)
    x = x + _lin(o.reshape(B, W, D), lp["attn_out"])
    h = _ln(x, lp["ln2"])
    h = jax.nn.gelu(_lin(h, lp["ff1"]))
    return x + _lin(h, lp["ff2"])


def forward(params, x, cfg):
    """x: (B, W, n_in) -> logits (B,) for the boundary left of each query column."""
    if int(getattr(cfg, "arch_version", 1)) == 2:
        feats = enrich_gamete_features(x)
        h = _lin(feats, params["embed"]) + params["pos"][None, :, :]
        h_pad = jnp.pad(h, ((0, 0), (2, 2), (0, 0)), mode="edge")
        dw_w = params["dw_conv_w"]
        dw = (
            h_pad[:, 0:-4, :] * dw_w[0]
            + h_pad[:, 1:-3, :] * dw_w[1]
            + h_pad[:, 2:-2, :] * dw_w[2]
            + h_pad[:, 3:-1, :] * dw_w[3]
            + h_pad[:, 4:, :] * dw_w[4]
            + params["dw_conv_b"]
        )
        h = h + jax.nn.gelu(_ln(dw, params["conv_ln"]))
        for lp in params["layers"]:
            h = _block(h, lp, cfg)
        h = _ln(h, params["ln_f"])
        c = int(cfg.centre)
        w_len = int(cfg.W)
        hr0 = h[:, c, :]
        hl0 = h[:, max(0, c - 1), :]
        d0 = hr0 - hl0
        d_near = h[:, c:min(w_len, c + 16), :].mean(axis=1) - h[:, max(0, c - 16):c, :].mean(axis=1)
        d_mid = h[:, c:min(w_len, c + 64), :].mean(axis=1) - h[:, max(0, c - 64):c, :].mean(axis=1)
        d_far = h[:, c:, :].mean(axis=1) - h[:, :c, :].mean(axis=1)
        pooled = jnp.concatenate([hr0, hl0, d0, d_near, d_mid, d_far], axis=-1)
        return _lin(jax.nn.gelu(_lin(pooled, params["head1"])), params["head2"]).squeeze(-1)

    h = _lin(x, params["embed"]) + params["pos"][None, :, :]
    for lp in params["layers"]:
        h = _block(h, lp, cfg)
    return _lin(h[:, cfg.centre, :], params["head"]).squeeze(-1)


def bce_logits(logits, y):
    """Numerically stable BCE-with-logits, elementwise."""
    return jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def loss_fn(params, x, y, cfg):
    logits = forward(params, x, cfg)
    w = jnp.where(y > 0.5, cfg.pos_weight, 1.0)
    return (bce_logits(logits, y) * w).sum() / w.sum()


def n_params(params):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(params)))
