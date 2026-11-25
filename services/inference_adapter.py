"""
Live GT-Transformer Inference Engine Adapter.

Connects the FastAPI web application directly to the real `gt-transformer`
pipeline, JAX Window Transformer checkpoints, local flanking KL localiser
(`flank_kl`), permutation-null calibrator, and 3-level event consolidation:
  - `pipeline.py`        : informative_columns, binarise, build_maps, unshrink, normalise_boundary
  - `model.py`           : Config, forward, n_params
  - `train.py`           : make_score
  - `haldec.py`          : boundary_zscore_flank, boundary_zscore, flank_soft_switch_detail,
                           soft_switch_detail, pick_peaks
  - `report.py`          : consolidate_peaks
  - `event_eval_orig.py` : load_truth_orig
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Set up web root on sys.path
WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

# Optional parent root for multi-checkpoint discovery when running in monorepo
GT_ROOT = WEB_ROOT.parent
WORKSPACE_ROOT = GT_ROOT.parent if (GT_ROOT / "pipeline.py").exists() else WEB_ROOT
SAMPLE_DATA_DIR = WEB_ROOT / "sample_data"

# Primary import from bundled gt_core with fallback to parent monorepo
try:
    from gt_core import haldec as H
    from gt_core import model as M
    from gt_core import pipeline as P
    from gt_core import report as REP
    from gt_core import train as T
    CORE_ENGINE_SOURCE = "bundled:web/gt_core"
except ImportError:
    if (GT_ROOT / "pipeline.py").exists():
        if str(GT_ROOT) not in sys.path:
            sys.path.insert(0, str(GT_ROOT))
        import haldec as H  # noqa: E402
        import model as M  # noqa: E402
        import pipeline as P  # noqa: E402
        import report as REP  # noqa: E402
        import train as T  # noqa: E402
        CORE_ENGINE_SOURCE = f"monorepo_parent:{GT_ROOT}"
    else:
        raise ImportError(
            f"Failed to load core gt-transformer inference engine. "
            f"Neither bundled 'web/gt_core' nor parent '{GT_ROOT}' contains valid modules."
        )

from schemas import (  # noqa: E402
    AlignmentMetadata,
    ConsolidatedEvent,
    GenomeTrackProfile,
    InferenceConfig,
    InferenceResponse,
    ParentalCandidate,
    RawPeakCall,
    RecombinantCall,
)
from services.diagnostics import evaluate_predictions_against_ground_truth  # noqa: E402
from services.fasta_utils import (  # noqa: E402
    AlignmentValidationError,
    AllGapSequenceError,
    BinaryPayloadError,
    EmptyPayloadError,
    InsufficientInformativeSitesError,
    InsufficientSequencesError,
    UnequalLengthError,
    compute_alignment_base_stats,
    parse_and_validate_fasta,
    parse_fasta_text,
    parse_ground_truth_payload,
    parse_raw_fasta_records,
)
from services.timing import TimingHistoryStore, TimingRecorder  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint Metadata & Multi-Discovery
# ---------------------------------------------------------------------------
KNOWN_CHECKPOINT_METADATA: dict[str, dict[str, Any]] = {
    "ckpt_N40000": {
        "label": "ckpt_N40000: 40k Corpus Standard Model (Recommended Default)",
        "description": (
            "Default checkpoint trained for 20 epochs on the 40,000-alignment SantaSim corpus "
            "(28,000 train / 6,000 val / 6,000 test). Evaluated separately at dataset, breakpoint, "
            "carrier, and co-descendant levels."
        ),
        "recommended": True,
    },
    "ckpt_N10000": {
        "label": "ckpt_N10000: 10k Corpus Standard Model",
        "description": (
            "Standard 10k checkpoint trained on 7,000 in-range SantaSim alignments (10,000 corpus)."
        ),
        "recommended": True,
    },
    "ckpt_N10000_extreme": {
        "label": "ckpt_N10000_extreme: Multi-Rate Extreme Regime (0.2x-5.0x Rates)",
        "description": (
            "Trained jointly on 7,000 in-range alignments and extreme mutation/recombination "
            "arms (0.2x-5.0x rates). Use for hyper-divergent or ultra-conserved alignments."
        ),
        "recommended": True,
    },
    "ckpt_N10000_v2": {
        "label": "ckpt_N10000_v2: V2 LD-Invariant Multi-Rate Model",
        "description": (
            "Warm-started V2 architecture (arch_version=2) with explicit linkage invariants "
            "(|D'|, r², 4-gamete flag, NMI), Conv1D stem, and multi-scale L/R contrast head."
        ),
        "recommended": True,
    },
    "ckpt_ensemble": {
        "label": "ckpt_ensemble: Multi-Model LogMeanExp Ensemble",
        "description": (
            "Virtual ensemble combining boundary logits via LogMeanExp across the strongest standard "
            "backbone (ckpt_N40000 or ckpt_N10000), ckpt_N10000_extreme, and converged ckpt_N10000_v2."
        ),
        "recommended": True,
        "is_virtual": True,
    },
    "ckpt_N8000": {
        "label": "ckpt_N8000: 8k Scaling Checkpoint",
        "description": "Trained on 5,600 in-range alignments from the 8k scaling tier.",
        "recommended": False,
    },
    "ckpt_N4000": {
        "label": "ckpt_N4000: 4k Scaling Checkpoint",
        "description": "Intermediate scaling checkpoint trained on 2,800 alignments.",
        "recommended": False,
    },
    "ckpt_N2000": {
        "label": "ckpt_N2000: 2k Scaling Checkpoint",
        "description": "Early scaling checkpoint trained on 1,400 alignments.",
        "recommended": False,
    },
    "ckpt_N1000": {
        "label": "ckpt_N1000: 1k Scaling Checkpoint",
        "description": "1,000-alignment baseline scaling tier (700 training alignments).",
        "recommended": False,
    },
    "ckpt_user_adaptive": {
        "label": "ckpt_user_adaptive: Adaptive (User-Trained)",
        "description": "Continuously fine-tuned on user-submitted alignments only.",
        "recommended": False,
    },
    "ckpt_4x_panel": {
        "label": "ckpt_4x_panel: 50-Dataset Panel (4x Augmented Regime)",
        "description": "Original 2,000-step Window Transformer trained on the 35-alignment panel.",
        "recommended": False,
    },
}


def discover_available_checkpoints() -> dict[str, dict[str, Any]]:
    """Discover checkpoints locally in web/checkpoints and optionally in parent directories."""
    ckpts: dict[str, dict[str, Any]] = {}

    # 1. Local web/checkpoints directory (Highest precedence)
    local_dir = WEB_ROOT / "checkpoints"
    if local_dir.is_dir():
        for p in sorted(local_dir.iterdir()):
            if p.is_dir() and (p / "params.pkl").is_file():
                cid = p.name
                meta = KNOWN_CHECKPOINT_METADATA.get(cid, {
                    "label": f"{cid} (Bundled Local)",
                    "description": f"Local checkpoint in web/checkpoints/{cid}",
                    "recommended": (cid == "ckpt_N10000"),
                })
                ckpts[cid] = {
                    "id": cid,
                    "path": p / "params.pkl",
                    "label": meta["label"],
                    "description": meta["description"],
                    "recommended": meta.get("recommended", False),
                    "source": "local",
                }
            elif p.is_file() and p.suffix == ".pkl":
                cid = p.stem
                meta = KNOWN_CHECKPOINT_METADATA.get(cid, {
                    "label": f"{cid} (Bundled Local)",
                    "description": f"Local checkpoint {p.name}",
                    "recommended": False,
                })
                ckpts[cid] = {
                    "id": cid,
                    "path": p,
                    "label": meta["label"],
                    "description": meta["description"],
                    "recommended": meta.get("recommended", False),
                    "source": "local",
                }

    # 2. Parent gt-transformer directory (Optional fallback in monorepo)
    if GT_ROOT.is_dir():
        parent_ckpts = GT_ROOT / "checkpoints"
        if parent_ckpts.is_dir():
            for p in sorted(parent_ckpts.iterdir()):
                cid = p.name if p.is_dir() else p.stem
                if cid not in ckpts:
                    fpath = (p / "params.pkl") if p.is_dir() else p
                    if fpath.is_file():
                        meta = KNOWN_CHECKPOINT_METADATA.get(cid, {
                            "label": f"{cid} (Parent Research Repo)",
                            "description": f"Discovered in {parent_ckpts}",
                            "recommended": False,
                        })
                        ckpts[cid] = {
                            "id": cid,
                            "path": fpath,
                            "label": meta["label"],
                            "description": meta["description"],
                            "recommended": meta.get("recommended", False),
                            "source": "parent",
                        }
        for p in sorted(GT_ROOT.glob("ckpt*")):
            if p.is_dir():
                cid = p.name
                if cid not in ckpts and (p / "params.pkl").is_file():
                    meta = KNOWN_CHECKPOINT_METADATA.get(cid, {
                        "label": f"{cid} (Parent Research Repo)",
                        "description": f"Discovered in parent {p.name}",
                        "recommended": False,
                    })
                    ckpts[cid] = {
                        "id": cid,
                        "path": p / "params.pkl",
                        "label": meta["label"],
                        "description": meta["description"],
                        "recommended": meta.get("recommended", False),
                        "source": "parent",
                    }
                elif cid == "ckpt" and p.is_dir():
                    p_4x = p / "model_4x_data.pkl"
                    if p_4x.is_file() and "ckpt_4x_panel" not in ckpts:
                        meta = KNOWN_CHECKPOINT_METADATA["ckpt_4x_panel"]
                        ckpts["ckpt_4x_panel"] = {
                            "id": "ckpt_4x_panel",
                            "path": p_4x,
                            "label": meta["label"],
                            "description": meta["description"],
                            "recommended": False,
                            "source": "parent",
                        }

    ens_members = resolve_ensemble_member_ids(ckpts)
    if len(ens_members) >= 2:
        meta_e = KNOWN_CHECKPOINT_METADATA["ckpt_ensemble"]
        ckpts["ckpt_ensemble"] = {
            "id": "ckpt_ensemble",
            "path": ckpts[ens_members[0]]["path"],
            "label": f"ckpt_ensemble: Multi-Model LogMeanExp Ensemble ({' + '.join(ens_members)})",
            "description": meta_e["description"],
            "recommended": meta_e.get("recommended", True),
            "source": "virtual",
            "is_virtual": True,
            "members": ens_members,
        }

    preferred_order = [
        "ckpt_N40000",
        "ckpt_ensemble",
        "ckpt_N10000_extreme",
        "ckpt_N10000_v2",
        "ckpt_N10000",
    ]
    ordered: dict[str, dict[str, Any]] = {}
    for cid in preferred_order:
        if cid in ckpts:
            ordered[cid] = ckpts[cid]
    for cid, entry in ckpts.items():
        if cid not in ordered:
            ordered[cid] = entry
    return ordered


def resolve_ensemble_member_ids(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Select the strongest available non-redundant ensemble members from discovered checkpoints."""
    members: list[str] = []
    std_id = (
        "ckpt_N40000"
        if ("ckpt_N40000" in registry and Path(registry["ckpt_N40000"]["path"]).is_file())
        else "ckpt_N10000"
    )
    if std_id in registry and Path(registry[std_id]["path"]).is_file():
        members.append(std_id)
    if "ckpt_N10000_extreme" in registry and Path(registry["ckpt_N10000_extreme"]["path"]).is_file():
        members.append("ckpt_N10000_extreme")
    if "ckpt_N10000_v2" in registry and Path(registry["ckpt_N10000_v2"]["path"]).is_file():
        try:
            v2_path = Path(registry["ckpt_N10000_v2"]["path"])
            with open(v2_path, "rb") as fh:
                v2_pl = pickle.load(fh)
            v2_cfg = M.config_from_dict(v2_pl.get("cfg", {}))
            v2_score = float(v2_pl.get("val_auprc") or 0.0)
            if int(getattr(v2_cfg, "arch_version", 1)) == 2 and v2_score >= 0.08:
                members.append("ckpt_N10000_v2")
        except Exception:
            pass
    return members


CHECKPOINT_REGISTRY: dict[str, dict[str, Any]] = discover_available_checkpoints()


# ---------------------------------------------------------------------------
# Preset Benchmark Datasets (Resolves Local sample_data first)
# ---------------------------------------------------------------------------
def resolve_preset_fasta_path(filename: str, fallback_path: Path) -> Path:
    sample_path = SAMPLE_DATA_DIR / filename
    if sample_path.exists():
        return sample_path
    return fallback_path


def resolve_preset_cache_dir(preset_id: str, fallback_dir: Optional[Path]) -> Optional[Path]:
    annot_dir = SAMPLE_DATA_DIR / "annotations"
    if (annot_dir / preset_id / "meta.json").exists():
        return annot_dir
    return fallback_dir


PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "d005_L3536_s96_r001": {
        "id": "d005_L3536_s96_r001",
        "label": "d005_L3536_s96_r001 — Benchmark Diagnostic (S=96, L=3,536 bp, 13 GT bps)",
        "category": "Benchmark Panel (Annotated by Dr. Martin)",
        "description": (
            "96 sequences × 3,536 bp SantaSim alignment with 10 events (13 real breakpoints). "
            "Tests edge events (78 bp, 3399 bp) and internal Event 4 (2523 bp)."
        ),
        "fasta_path": resolve_preset_fasta_path("d005_L3536_s96_r001.fas", WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d005_L3536_s96_r001.fas"),
        "cache_dir": resolve_preset_cache_dir("d005_L3536_s96_r001", GT_ROOT / "cache"),
        "is_control": False,
    },
    "d048_L4577_s79_r001": {
        "id": "d048_L4577_s79_r001",
        "label": "d048_L4577_s79_r001 — Held-Out Benchmark Test (S=79, L=4,577 bp, 9 GT bps)",
        "category": "Benchmark Panel (Held-Out Test Split)",
        "description": "Unseen held-out benchmark alignment (79 sequences × 4,577 bp, r=0.001, 6 events / 9 real bps).",
        "fasta_path": resolve_preset_fasta_path("d048_L4577_s79_r001.fas", WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d048_L4577_s79_r001.fas"),
        "cache_dir": resolve_preset_cache_dir("d048_L4577_s79_r001", GT_ROOT / "cache"),
        "is_control": False,
    },
    "d008_L8383_s79_r001": {
        "id": "d008_L8383_s79_r001",
        "label": "d008_L8383_s79_r001 — Held-Out Benchmark Test (S=79, L=8,383 bp, 8 GT bps)",
        "category": "Benchmark Panel (Held-Out Test Split)",
        "description": "Unseen held-out benchmark alignment (79 sequences × 8,383 bp, r=0.001, 6 events / 8 real bps).",
        "fasta_path": resolve_preset_fasta_path("d008_L8383_s79_r001.fas", WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d008_L8383_s79_r001.fas"),
        "cache_dir": resolve_preset_cache_dir("d008_L8383_s79_r001", GT_ROOT / "cache"),
        "is_control": False,
    },
    "d047_L15068_s27_ctrl": {
        "id": "d047_L15068_s27_ctrl",
        "label": "d047_L15068_s27_ctrl — Held-Out Clonal Control (S=27, L=15,068 bp, 0 GT)",
        "category": "Benchmark Panel (Held-Out Negative Control)",
        "description": "Strictly clonal held-out benchmark control alignment with 0 recombination events (specificity test).",
        "fasta_path": resolve_preset_fasta_path("d047_L15068_s27_ctrl.fas", WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d047_L15068_s27_ctrl.fas"),
        "cache_dir": resolve_preset_cache_dir("d047_L15068_s27_ctrl", GT_ROOT / "cache"),
        "is_control": True,
    },
    "d00003_L2487_s28_r005": {
        "id": "d00003_L2487_s28_r005",
        "label": "d00003_L2487_s28_r005 — 10k Corpus Held-Out Test (S=28, L=2,487 bp, r=0.005)",
        "category": "10,000-Alignment Corpus (Held-Out Test Split)",
        "description": "Unseen high-recombination test alignment from the 10k simulation corpus (94 events).",
        "fasta_path": resolve_preset_fasta_path("d00003_L2487_s28_r005.fas", GT_ROOT / "datasets_sims" / "d00003_L2487_s28_r005.fas"),
        "cache_dir": resolve_preset_cache_dir("d00003_L2487_s28_r005", GT_ROOT / "cache_sims"),
        "is_control": False,
    },
    "d00011_L9812_s43_r0005": {
        "id": "d00011_L9812_s43_r0005",
        "label": "d00011_L9812_s43_r0005 — 10k Corpus Sparse Test (S=43, L=9,812 bp, 2 GT bps)",
        "category": "10,000-Alignment Corpus (Held-Out Test Split)",
        "description": "Unseen sparse-recombination test alignment from the 10k corpus (2 events / 2 real breakpoints).",
        "fasta_path": resolve_preset_fasta_path("d00011_L9812_s43_r0005.fas", GT_ROOT / "datasets_sims" / "d00011_L9812_s43_r0005.fas"),
        "cache_dir": resolve_preset_cache_dir("d00011_L9812_s43_r0005", GT_ROOT / "cache_sims"),
        "is_control": False,
    },
    "d00005_L7268_s23_ctrl": {
        "id": "d00005_L7268_s23_ctrl",
        "label": "d00005_L7268_s23_ctrl — 10k Corpus Clonal Control (S=23, L=7,268 bp, 0 GT)",
        "category": "10,000-Alignment Corpus (Held-Out Negative Control)",
        "description": "Unseen clonal negative control from the 10k corpus (0 true events; tests false-positive rate).",
        "fasta_path": resolve_preset_fasta_path("d00005_L7268_s23_ctrl.fas", GT_ROOT / "datasets_sims" / "d00005_L7268_s23_ctrl.fas"),
        "cache_dir": resolve_preset_cache_dir("d00005_L7268_s23_ctrl", GT_ROOT / "cache_sims"),
        "is_control": True,
    },
    "ext_0076_L4413_s95_high_mu_high_r_mu25_r025": {
        "id": "ext_0076_L4413_s95_high_mu_high_r_mu25_r025",
        "label": "ext_0076 — Extreme 5x Mu + 5x Recomb (S=95, L=4,413 bp, mu=2.5e-3, r=0.25)",
        "category": "Out-of-Range Extreme Suite (500% Rate Arm)",
        "description": "Extreme out-of-range alignment at 5x mutation and 5x recombination rates.",
        "fasta_path": resolve_preset_fasta_path("ext_0076_L4413_s95_high_mu_high_r_mu25_r025.fas", GT_ROOT / "datasets_extreme" / "ext_0076_L4413_s95_high_mu_high_r_mu25_r025.fas"),
        "cache_dir": resolve_preset_cache_dir("ext_0076_L4413_s95_high_mu_high_r_mu25_r025", GT_ROOT / "cache_extreme"),
        "is_control": False,
    },
    "ext_0001_L6809_s57_low_mu_low_r_mu01_r0001": {
        "id": "ext_0001_L6809_s57_low_mu_low_r_mu01_r0001",
        "label": "ext_0001 — Extreme 0.2x Mu + 0.2x Recomb (S=57, L=6,809 bp, mu=1e-4, r=0.001)",
        "category": "Out-of-Range Extreme Suite (20% Rate Arm)",
        "description": "Extreme low-diversity, low-recombination alignment (20% rates).",
        "fasta_path": resolve_preset_fasta_path("ext_0001_L6809_s57_low_mu_low_r_mu01_r0001.fas", GT_ROOT / "datasets_extreme" / "ext_0001_L6809_s57_low_mu_low_r_mu01_r0001.fas"),
        "cache_dir": resolve_preset_cache_dir("ext_0001_L6809_s57_low_mu_low_r_mu01_r0001", GT_ROOT / "cache_extreme"),
        "is_control": False,
    },
    "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl": {
        "id": "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl",
        "label": "ext_0101 — Extreme Low-Mu Clonal Control (S=27, L=3,291 bp, 0 GT)",
        "category": "Out-of-Range Extreme Suite (Negative Control)",
        "description": "Extreme low-mutation clonal control (r=0.0, 0 true breakpoints).",
        "fasta_path": resolve_preset_fasta_path("ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl.fas", GT_ROOT / "datasets_extreme" / "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl.fas"),
        "cache_dir": resolve_preset_cache_dir("ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl", GT_ROOT / "cache_extreme"),
        "is_control": True,
    },
    "sarbecovirus_aligned": {
        "id": "sarbecovirus_aligned",
        "label": "sarbecovirus_aligned — Empirical Coronavirus Alignment (S=19, L=30,619 bp)",
        "category": "Empirical Viral Alignment (Unlabelled)",
        "description": "19 empirical aligned Sarbecovirus genomes (30,619 bp) with gaps/ambiguities.",
        "fasta_path": resolve_preset_fasta_path("sarbecovirus_aligned.fasta", WORKSPACE_ROOT / "sarbecovirus_aligned.fasta"),
        "cache_dir": None,
        "is_control": False,
    },
    "test_neisseria": {
        "id": "test_neisseria",
        "label": "test_neisseria — Empirical Bacterial Benchmark (S=4, L=787 bp)",
        "category": "Empirical Bacterial Alignment",
        "description": "4 Neisseria sequences (787 bp, 62 informative sites).",
        "fasta_path": resolve_preset_fasta_path("test_neisseria.fasta", WORKSPACE_ROOT / "tools" / "openrdp" / "tests" / "test_neisseria.fasta"),
        "cache_dir": None,
        "is_control": False,
    },
}


def compute_max_pairwise_identity(seqs: list[str], max_sample_cols: int = 5000) -> float:
    """Compute maximum pairwise sequence identity across all sequences in the alignment.

    Converts sequences to uppercase to prevent case-sensitivity mismatches on soft-masked
    FASTAs, and evaluates across all sequences without artificial sequence truncation limits.
    """
    S = len(seqs)
    if S < 2:
        return 1.0
    L = len(seqs[0])
    if L == 0:
        return 0.0
    if L > max_sample_cols:
        step = max(1, L // max_sample_cols)
        sampled_seqs = [s[::step][:max_sample_cols] for s in seqs]
        L_eff = len(sampled_seqs[0])
    else:
        sampled_seqs = seqs
        L_eff = L

    # Case-insensitive ASCII encoding
    arr = np.array([np.frombuffer(s.upper().encode("ascii", errors="replace"), dtype=np.uint8) for s in sampled_seqs])
    max_id = 0.0
    # Evaluate across all sequences (early-stopping if biological homology >= 99% is found)
    for i in range(S - 1):
        matches = (arr[i : i + 1] == arr[i + 1 :]).sum(axis=1) / float(L_eff)
        if len(matches) > 0:
            m = float(matches.max())
            if m > max_id:
                max_id = m
            if max_id >= 0.99:
                break
    return max_id


class BaseInferenceEngine(ABC):
    """Abstract interface for recombination inference engines."""

    @abstractmethod
    def list_presets(self) -> list[dict[str, Any]]:
        """Return available preset benchmark datasets."""

    @abstractmethod
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return available trained checkpoints on disk."""

    @abstractmethod
    def run_inference(
        self,
        config: InferenceConfig,
        preset_id: Optional[str] = None,
        fasta_text: Optional[str] = None,
        fasta_filename: Optional[str] = None,
        gt_text: Optional[str] = None,
        gt_filename: Optional[str] = None,
    ) -> InferenceResponse:
        """Execute inference and return real timings + 3-level report + diagnostics."""


def _parse_parent_tokens(label: str) -> set[str]:
    """Extract normalized parental taxon names from a possibly tie-annotated label."""
    tokens: set[str] = set()
    for part in str(label or "").split(" or "):
        for sub in part.split("/"):
            tok = sub.strip()
            if tok and tok != "?":
                tokens.add(tok)
    return tokens


def assign_ancestral_event_ids(events: list[ConsolidatedEvent], max_bp_gap: int = 150) -> None:
    """Group co-descendant events sharing breakpoint locality and parental donors under 1-based `ancestral_event_id`."""
    n = len(events)
    if n == 0:
        return

    def _share_origin(a: ConsolidatedEvent, b: ConsolidatedEvent) -> bool:
        if abs(int(a.most_plausible_bp_orig) - int(b.most_plausible_bp_orig)) > max_bp_gap:
            return False
        if not a.recombinants or not b.recombinants:
            return False
        ra, rb = a.recombinants[0], b.recombinants[0]
        la = _parse_parent_tokens(ra.ext_parent_left_name)
        lb = _parse_parent_tokens(rb.ext_parent_left_name)
        ra_p = _parse_parent_tokens(ra.ext_parent_right_name)
        rb_p = _parse_parent_tokens(rb.ext_parent_right_name)
        return bool(la & lb) and bool(ra_p & rb_p)

    visited = [False] * n
    next_gid = 1
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp: list[int] = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not visited[v] and _share_origin(events[u], events[v]):
                    visited[v] = True
                    stack.append(v)
        for idx in comp:
            events[idx].ancestral_event_id = next_gid
        next_gid += 1


class GTTransformerInferenceEngine(BaseInferenceEngine):
    """Live GT-Transformer inference engine wired directly to model & pipeline infrastructure."""

    CHUNK_SIZE = 512

    def __init__(self, timing_store: TimingHistoryStore) -> None:
        self.timing_store = timing_store
        self._ckpt_cache: dict[str, tuple[Any, M.Config, int]] = {}
        self._score_fn_cache: dict[M.Config, Any] = {}
        self._warmed_configs: set[M.Config] = set()
        self._preset_meta_cache: Optional[list[dict[str, Any]]] = None
        self._alignment_cache_lock = threading.Lock()
        self._alignment_cache: dict[str, dict[str, Any]] = {}
        self._alignment_cache_order: list[str] = []
        self.is_warmed_up: bool = False
        self.startup_load_ms: float = 0.0

    def _store_alignment_cache(
        self,
        run_id: str,
        names: list[str],
        seqs: list[str],
        kept: list[int] | np.ndarray,
        L: int,
    ) -> None:
        with self._alignment_cache_lock:
            self._alignment_cache[run_id] = {
                "names": list(names),
                "seqs": list(seqs),
                "kept_set": set(int(c) for c in kept),
                "L": int(L),
            }
            if run_id in self._alignment_cache_order:
                self._alignment_cache_order.remove(run_id)
            self._alignment_cache_order.append(run_id)
            while len(self._alignment_cache_order) > 16:
                old_id = self._alignment_cache_order.pop(0)
                self._alignment_cache.pop(old_id, None)

    def hydrate_alignment_cache(
        self, run_id: str, fasta_text: str, null_policy: str = "duplicate"
    ) -> bool:
        """Populate the in-memory alignment viewer cache from saved FASTA text or a preset reference."""
        stripped = (fasta_text or "").strip()
        if not stripped:
            return False
        actual_fasta = stripped
        if stripped.startswith("# preset:"):
            preset_id = stripped.split(":", 1)[1].strip()
            preset_meta = PRESET_REGISTRY.get(preset_id)
            if not preset_meta:
                return False
            fasta_path = Path(preset_meta["fasta_path"])
            if not fasta_path.exists():
                return False
            actual_fasta = fasta_path.read_text(encoding="utf-8", errors="ignore")
        try:
            names, seqs, _, _ = parse_and_validate_fasta(actual_fasta, strict_phase2=False)
            if not seqs or not seqs[0]:
                return False
            kept, _, _, _ = self._select_informative_and_binarise(seqs, null_policy)
        except Exception:
            return False
        self._store_alignment_cache(run_id, names, seqs, kept, len(seqs[0]))
        return True

    def get_alignment_window(
        self, run_id: str, start_bp: int = 0, width: int = 120
    ) -> dict[str, Any]:
        """Return actual nucleotide alignment slice & informative site coordinates for the sequence viewer."""
        with self._alignment_cache_lock:
            entry = self._alignment_cache.get(run_id)
        if entry is None:
            raise ValueError(f"Alignment session '{run_id}' not found or expired; re-run inference.")
        L = int(entry["L"])
        width = max(20, min(int(width), 300))
        start = max(0, min(int(start_bp), max(0, L - 1)))
        end = min(L, start + width)
        if end - start < width and L > width:
            start = max(0, L - width)
            end = L

        kept_set: set[int] = entry["kept_set"]
        info_cols = [c for c in range(start, end) if c in kept_set]
        names: list[str] = entry["names"]
        seqs: list[str] = entry["seqs"]
        seq_rows = [
            {
                "seq_idx": i,
                "name": names[i],
                "slice": seqs[i][start:end],
            }
            for i in range(len(seqs))
        ]
        return {
            "run_id": run_id,
            "start_bp": start,
            "end_bp": end,
            "genome_length_bp": L,
            "informative_cols": info_cols,
            "sequences": seq_rows,
        }

    @staticmethod
    def _seqs_to_ascii_array(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.vstack(
            [
                np.frombuffer(
                    seq.upper().replace("U", "T").encode("ascii", errors="replace"),
                    dtype=np.uint8,
                )
                for seq in seqs
            ]
        )
        valid = (arr == 65) | (arr == 67) | (arr == 71) | (arr == 84)
        return arr, valid

    def get_region_stats(
        self,
        run_id: str,
        start_bp: int,
        end_bp: int,
        ref_seq_idx: Optional[int] = None,
    ) -> dict[str, Any]:
        """Compute nucleotide diversity, GC/gap stats, and pairwise inside-vs-outside identity for a selected region."""
        with self._alignment_cache_lock:
            entry = self._alignment_cache.get(run_id)
        if entry is None:
            raise ValueError(f"Alignment session '{run_id}' not found or expired; re-run inference.")

        L = int(entry["L"])
        names: list[str] = entry["names"]
        seqs: list[str] = entry["seqs"]
        kept_set: set[int] = entry["kept_set"]
        S = len(seqs)

        start_1 = max(1, min(L, int(start_bp)))
        end_1 = max(start_1, min(L, int(end_bp)))
        s0 = start_1 - 1
        e0 = end_1
        reg_len = e0 - s0
        info_sites = sum(1 for c in range(s0, e0) if c in kept_set)

        arr, valid_all = self._seqs_to_ascii_array(seqs)
        reg = arr[:, s0:e0]
        valid_reg = valid_all[:, s0:e0]

        gc_cnt = int(np.sum((reg == 71) | (reg == 67)))
        acgt_cnt = int(np.sum(valid_reg))
        gc_pct = round((gc_cnt / acgt_cnt) * 100.0, 2) if acgt_cnt > 0 else 0.0
        gap_pct = round((1.0 - (acgt_cnt / max(1, reg.size))) * 100.0, 2)

        sub_idx = np.linspace(0, S - 1, min(S, 40), dtype=int) if S > 40 else np.arange(S)
        reg_sub = reg[sub_idx]
        v_sub = valid_reg[sub_idx]
        diffs_sum = 0.0
        pair_cnt = 0
        for i_idx in range(len(sub_idx)):
            for j_idx in range(i_idx + 1, len(sub_idx)):
                both = v_sub[i_idx] & v_sub[j_idx]
                nb = int(np.sum(both))
                if nb > 0:
                    diffs_sum += float(np.sum(reg_sub[i_idx, both] != reg_sub[j_idx, both])) / nb
                    pair_cnt += 1
        pi_div = round(diffs_sum / pair_cnt, 5) if pair_cnt > 0 else 0.0

        ref_idx = max(0, min(S - 1, int(ref_seq_idx or 0)))
        both_in = valid_reg & valid_reg[ref_idx : ref_idx + 1, :]
        match_in = both_in & (reg == reg[ref_idx : ref_idx + 1, :])
        cnt_both_in = np.sum(both_in, axis=1)
        cnt_match_in = np.sum(match_in, axis=1)

        out_mask = np.ones(L, dtype=bool)
        out_mask[s0:e0] = False
        has_outside = bool(np.any(out_mask))
        if has_outside:
            arr_out = arr[:, out_mask]
            valid_out = valid_all[:, out_mask]
            both_out = valid_out & valid_out[ref_idx : ref_idx + 1, :]
            match_out = both_out & (arr_out == arr_out[ref_idx : ref_idx + 1, :])
            cnt_both_out = np.sum(both_out, axis=1)
            cnt_match_out = np.sum(match_out, axis=1)
        else:
            cnt_both_out = np.zeros(S, dtype=np.int64)
            cnt_match_out = np.zeros(S, dtype=np.int64)

        pairwise: list[dict[str, Any]] = []
        for j in range(S):
            if j == ref_idx:
                continue
            n_in = int(cnt_both_in[j])
            in_pct = round(float(cnt_match_in[j]) / n_in * 100.0, 2) if n_in > 0 else None
            n_out = int(cnt_both_out[j])
            out_pct = round(float(cnt_match_out[j]) / n_out * 100.0, 2) if n_out > 0 else None
            delta_pct = (
                round(in_pct - out_pct, 2)
                if (in_pct is not None and out_pct is not None)
                else None
            )
            pairwise.append(
                {
                    "seq_idx": j,
                    "seq_name": names[j],
                    "inside_pct": in_pct,
                    "outside_pct": out_pct,
                    "delta_pct": delta_pct,
                    "compared_in": n_in,
                    "compared_out": n_out,
                }
            )
        pairwise.sort(key=lambda r: (r["inside_pct"] is not None, r["inside_pct"] or -1.0), reverse=True)

        return {
            "run_id": run_id,
            "start_bp": start_1,
            "end_bp": end_1,
            "length_bp": reg_len,
            "genome_length_bp": L,
            "informative_sites": info_sites,
            "gc_pct": gc_pct,
            "gap_pct": gap_pct,
            "nucleotide_diversity_pi": pi_div,
            "ref_seq_idx": ref_idx,
            "ref_seq_name": names[ref_idx],
            "sequences": [{"seq_idx": i, "name": names[i]} for i in range(S)],
            "pairwise": pairwise,
        }

    def get_simplot_profile(
        self,
        run_id: str,
        query_seq_idx: int = 0,
        window_bp: int = 100,
        step_bp: int = 20,
        max_refs: int = 8,
        priority_seq_idxs: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Compute vectorized sliding-window pairwise identity curves against candidate parental sequences."""
        with self._alignment_cache_lock:
            entry = self._alignment_cache.get(run_id)
        if entry is None:
            raise ValueError(f"Alignment session '{run_id}' not found or expired; re-run inference.")

        L = int(entry["L"])
        names: list[str] = entry["names"]
        seqs: list[str] = entry["seqs"]
        S = len(seqs)

        arr, valid = self._seqs_to_ascii_array(seqs)
        q_idx = max(0, min(S - 1, int(query_seq_idx)))
        win = max(20, min(L, int(window_bp)))
        step = max(1, min(win, int(step_bp)))
        k_refs = max(1, min(16, int(max_refs)))

        both = valid & valid[q_idx : q_idx + 1, :]
        match = both & (arr == arr[q_idx : q_idx + 1, :])
        cs_both = np.concatenate([np.zeros((S, 1), dtype=np.int32), np.cumsum(both, axis=1, dtype=np.int32)], axis=1)
        cs_match = np.concatenate([np.zeros((S, 1), dtype=np.int32), np.cumsum(match, axis=1, dtype=np.int32)], axis=1)

        max_start = max(0, L - win)
        starts_list = list(range(0, max_start + 1, step))
        if not starts_list or starts_list[-1] != max_start:
            starts_list.append(max_start)
        starts = np.asarray(starts_list, dtype=np.int32)
        ends = np.minimum(L, starts + win)
        centers_1based = ((starts + ends) // 2 + 1).tolist()

        denom = cs_both[:, ends] - cs_both[:, starts]
        numer = cs_match[:, ends] - cs_match[:, starts]
        ident_pct = np.where(denom > 0, (numer / np.maximum(denom, 1)) * 100.0, np.nan)

        chosen: list[int] = []
        seen: set[int] = {q_idx}
        for p in priority_seq_idxs or []:
            if 0 <= p < S and p not in seen and len(chosen) < k_refs:
                seen.add(p)
                chosen.append(p)

        remaining: list[tuple[float, int]] = []
        for j in range(S):
            if j in seen:
                continue
            vals = ident_pct[j]
            finite = vals[np.isfinite(vals)]
            score = float(np.std(finite) + 0.25 * np.mean(finite)) if len(finite) > 0 else -1.0
            remaining.append((score, j))
        remaining.sort(key=lambda x: x[0], reverse=True)
        for _, j in remaining:
            if len(chosen) >= k_refs:
                break
            chosen.append(j)

        series = [
            {
                "seq_idx": j,
                "seq_name": names[j],
                "identity_pct": [round(float(v), 2) if np.isfinite(v) else None for v in ident_pct[j]],
            }
            for j in chosen
        ]

        return {
            "run_id": run_id,
            "genome_length_bp": L,
            "window_bp": win,
            "step_bp": step,
            "query_seq_idx": q_idx,
            "query_seq_name": names[q_idx],
            "positions_bp": centers_1based,
            "sequences": [{"seq_idx": i, "name": names[i]} for i in range(S)],
            "series": series,
        }

    def _discover_checkpoints(self) -> dict[str, dict[str, Any]]:
        return discover_available_checkpoints()

    def list_checkpoints(self) -> list[dict[str, Any]]:
        registry = self._discover_checkpoints()
        out = []
        for ckpt_id, meta in registry.items():
            p = Path(meta["path"])
            if p.exists():
                label = meta["label"]
                updates: Optional[int] = None
                if ckpt_id == "ckpt_user_adaptive":
                    try:
                        with open(p, "rb") as fh:
                            payload = pickle.load(fh)
                        updates = int((payload.get("meta") or {}).get("updates", 0))
                        if updates > 0:
                            suffix = "s" if updates != 1 else ""
                            label = f"ckpt_user_adaptive: Adaptive (User-Trained, {updates} update{suffix})"
                    except Exception:
                        updates = None
                item: dict[str, Any] = {
                    "id": ckpt_id,
                    "label": label,
                    "description": meta["description"],
                    "recommended": meta["recommended"],
                    "path": str(p),
                    "source": meta.get("source", "local"),
                }
                if updates is not None:
                    item["updates"] = updates
                out.append(item)
        rank_map = {
            "ckpt_N40000": 0,
            "ckpt_ensemble": 1,
            "ckpt_N10000_extreme": 2,
            "ckpt_N10000_v2": 3,
            "ckpt_N10000": 4,
        }
        out.sort(key=lambda x: (rank_map.get(x["id"], 99), not x["recommended"], x["id"]))
        return out

    def list_presets(self) -> list[dict[str, Any]]:
        if self._preset_meta_cache is not None:
            return self._preset_meta_cache

        out = []
        for pid, pmeta in PRESET_REGISTRY.items():
            fpath = Path(pmeta["fasta_path"])
            if not fpath.exists():
                continue
            cdir = pmeta.get("cache_dir")
            n_seq = None
            length = None
            n_info = None
            n_gt = 0
            if cdir is not None and (Path(cdir) / pid / "meta.json").exists():
                try:
                    m = json.loads((Path(cdir) / pid / "meta.json").read_text(encoding="utf-8"))
                    n_seq = m.get("n_sequences")
                    length = m.get("genome_length")
                    n_info = m.get("informative_columns")
                    evs = json.loads((Path(cdir) / pid / "events.json").read_text(encoding="utf-8"))
                    n_gt = len({int(b) for e in evs for b in e.get("breakpoints_orig", [])})
                except Exception:
                    pass
            if n_seq is None:
                try:
                    names, seqs = P.read_fasta(str(fpath))
                    n_seq = len(names)
                    length = max(len(s) for s in seqs)
                except Exception:
                    n_seq, length = 0, 0
            out.append(
                {
                    "id": pid,
                    "label": pmeta["label"],
                    "category": pmeta["category"],
                    "description": pmeta["description"],
                    "n_sequences": n_seq,
                    "genome_length_bp": length,
                    "n_informative_cols": n_info,
                    "n_ground_truth_breakpoints": n_gt,
                    "is_control": bool(pmeta.get("is_control", False)),
                }
            )
        self._preset_meta_cache = out
        return out

    def _resolve_ensemble_members(self) -> list[str]:
        return resolve_ensemble_member_ids(self._discover_checkpoints())

    def warmup(self, checkpoint_id: str = "ckpt_N10000") -> None:
        """Load core checkpoints once and pre-compile static-shape JAX score functions."""
        import time
        t0 = time.perf_counter()
        registry = self._discover_checkpoints()
        target_ids = []
        req_ids = self._resolve_ensemble_members() if checkpoint_id == "ckpt_ensemble" else [checkpoint_id]
        for cid in (*req_ids, "ckpt_N40000", "ckpt_N10000", "ckpt_N10000_extreme"):
            if (
                cid not in target_ids
                and cid in registry
                and not registry[cid].get("is_virtual")
                and Path(registry[cid]["path"]).exists()
            ):
                target_ids.append(cid)
        if not target_ids:
            avail = [c for c in self.list_checkpoints() if c["id"] != "ckpt_ensemble"]
            if not avail:
                return
            target_ids = [avail[0]["id"]]
        for cid in target_ids:
            params, cfg, _ = self._load_checkpoint(cid)
            self._get_score_fn(params, cfg)
        self.startup_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.is_warmed_up = True

    def _load_checkpoint(self, checkpoint_id: str) -> tuple[Any, M.Config, int]:
        if checkpoint_id in self._ckpt_cache:
            return self._ckpt_cache[checkpoint_id]

        registry = self._discover_checkpoints()
        if checkpoint_id not in registry:
            raise ValueError(f"Unknown checkpoint '{checkpoint_id}'. Available: {list(registry.keys())}")
        ckpt_path = Path(registry[checkpoint_id]["path"])
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint file not found on disk: {ckpt_path}")

        payload = None
        for attempt in range(2):
            try:
                with open(ckpt_path, "rb") as fh:
                    payload = pickle.load(fh)
                break
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                raise
        params = payload["params"]
        cfg_dict = payload.get("cfg", {})
        cfg = M.config_from_dict(cfg_dict)
        n_params = int(M.n_params(params))
        self._ckpt_cache[checkpoint_id] = (params, cfg, n_params)
        return params, cfg, n_params

    def _get_score_fn(self, params: Any, cfg: M.Config) -> Any:
        if cfg not in self._score_fn_cache:
            self._score_fn_cache[cfg] = T.make_score(cfg)
        score_fn = self._score_fn_cache[cfg]
        if cfg not in self._warmed_configs:
            dummy = np.zeros((self.CHUNK_SIZE, cfg.W, cfg.n_in), dtype=np.float32)
            out = score_fn(params, dummy)
            out.block_until_ready()
            self._warmed_configs.add(cfg)
        return score_fn

    def _load_preset_ground_truth(self, preset_id: str, cache_dir: Optional[Path]) -> list[dict[str, Any]]:
        if cache_dir is None:
            return []
        ds_dir = Path(cache_dir) / preset_id
        events_path = ds_dir / "events.json"
        if not events_path.exists():
            return []

        events = json.loads(events_path.read_text(encoding="utf-8"))
        seq_event_path = ds_dir / "seq_event.npy"
        seq_ev = np.load(seq_event_path) if seq_event_path.exists() else None

        gt_items: list[dict[str, Any]] = []
        for idx, ev in enumerate(events):
            carriers: list[int] = []
            if seq_ev is not None and idx < seq_ev.shape[1]:
                carriers = [int(s) for s in np.flatnonzero(seq_ev[:, idx])]
            bps = ev.get("breakpoints_orig", [])
            txs = ev.get("boundaries_tx", [])
            for b_i, bp in enumerate(bps):
                gt_items.append(
                    {
                        "gt_bp_orig": int(bp),
                        "gt_boundary_tx": int(txs[b_i]) if b_i < len(txs) else None,
                        "gt_event_id": ev.get("event_id", idx + 1),
                        "event_type": str(ev.get("event_type", "unknown")),
                        "generation": ev.get("generation"),
                        "carrier_seqs": carriers,
                    }
                )
        return sorted(gt_items, key=lambda x: x["gt_bp_orig"])

    @staticmethod
    def _select_informative_and_binarise(
        seqs: list[str],
        null_mode: str,
        strict_min: int = 30,
        max_non_acgt_frac: float = 0.25,
    ) -> tuple[list[int], list[list[str]], list[int], str]:
        """Select parsimony-informative columns and binarise for 4-gamete embedding.

        Uses exact `pipeline.informative_columns` + `pipeline.binarise` whenever at
        least `strict_min` strict ACGT informative columns exist (preserving bit-for-bit
        parity with `pipeline.py`). For real-world FASTAs with scattered gaps or
        ambiguities (`-`, `N`, IUPAC) that leave fewer than `strict_min` gap-free sites,
        falls back to gap-tolerant ACGT informative columns (`<= max_non_acgt_frac`
        non-ACGT) and treats gaps/ambiguities as missing (`'-'`) under `null_mode`.
        """
        kept_strict = P.informative_columns(seqs)
        if len(kept_strict) >= strict_min:
            cols, origin, _kind = P.binarise(seqs, kept_strict, null_mode=null_mode)
            return kept_strict, cols, origin, "strict_acgt"

        acgt = ("A", "C", "G", "T")
        n_seq = len(seqs)
        genome_len = len(seqs[0])
        gap_kept: list[int] = []
        col_major_minor: list[tuple[str, str]] = []

        for c in range(genome_len):
            counts = {b: 0 for b in acgt}
            non_acgt = 0
            for s in seqs:
                ch = s[c]
                if ch in counts:
                    counts[ch] += 1
                else:
                    non_acgt += 1
            if non_acgt / max(n_seq, 1) > max_non_acgt_frac:
                continue
            ordered = sorted(
                ((b, cnt) for b, cnt in counts.items() if cnt > 0),
                key=lambda kv: (-kv[1], kv[0]),
            )
            if len(ordered) < 2 or ordered[1][1] < 2:
                continue
            gap_kept.append(c)
            col_major_minor.append((ordered[0][0], ordered[1][0]))

        if len(gap_kept) <= len(kept_strict):
            cols, origin, _kind = P.binarise(seqs, kept_strict, null_mode=null_mode)
            return kept_strict, cols, origin, "strict_acgt"

        if null_mode not in P.NULL_MODES:
            raise ValueError(null_mode)

        cols: list[list[str]] = []
        origin: list[int] = []
        for c, (major, minor) in zip(gap_kept, col_major_minor):
            raw = [
                "1" if s[c] == major else ("0" if s[c] == minor else "-")
                for s in seqs
            ]
            has_null = "-" in raw
            if null_mode == "drop":
                if not has_null:
                    cols.append(raw)
                    origin.append(c)
            elif null_mode == "zero":
                cols.append(["0" if x == "-" else x for x in raw])
                origin.append(c)
            else:  # duplicate
                if has_null:
                    cols.append(["1" if x == "-" else x for x in raw])
                    origin.append(c)
                    cols.append(["0" if x == "-" else x for x in raw])
                    origin.append(c)
                else:
                    cols.append(raw)
                    origin.append(c)
        return gap_kept, cols, origin, "gap_tolerant"

    @staticmethod
    def _build_circular_windows_freq(haplo: np.ndarray, window_size: int) -> np.ndarray:
        """Compute (M_tx, window_size, 4) normalized gamete frequency windows using bitwise counts.

        When M_tx < window_size, applies linear zero-padding to out-of-range
        offsets to avoid periodic circular wraparound boundary distortion.
        """
        M_tx, S = haplo.shape
        B = haplo.T.astype(np.uint8)  # (S, M_tx) in {0, 1}
        ones = B.sum(axis=0, dtype=np.int32)
        offs = P._offsets(window_size)
        idx = np.arange(M_tx)
        X = np.zeros((M_tx, len(offs), 4), dtype=np.float32)
        inv_s = np.float32(1.0 / max(S, 1))

        if M_tx < window_size:
            # Linear zero-padding for short alignments
            for pos, d in enumerate(offs):
                j = idx + d
                valid = (j >= 0) & (j < M_tx)
                if not np.any(valid):
                    continue
                vi = idx[valid]
                vj = j[valid]
                c11 = np.count_nonzero(B[:, vi] & B[:, vj], axis=0).astype(np.int32)
                oi = ones[vi]
                oj = ones[vj]
                c10 = oi - c11
                c01 = oj - c11
                c00 = S - oi - oj + c11
                X[valid, pos, 0] = c00.astype(np.float32) * inv_s
                X[valid, pos, 1] = c01.astype(np.float32) * inv_s
                X[valid, pos, 2] = c10.astype(np.float32) * inv_s
                X[valid, pos, 3] = c11.astype(np.float32) * inv_s
        else:
            # Standard circular moving window for full-length alignments
            for pos, d in enumerate(offs):
                j = (idx + d) % M_tx
                c11 = np.count_nonzero(B & B[:, j], axis=0).astype(np.int32)
                oj = ones[j]
                c10 = ones - c11
                c01 = oj - c11
                c00 = S - ones - oj + c11
                X[:, pos, 0] = c00.astype(np.float32) * inv_s
                X[:, pos, 1] = c01.astype(np.float32) * inv_s
                X[:, pos, 2] = c10.astype(np.float32) * inv_s
                X[:, pos, 3] = c11.astype(np.float32) * inv_s
        return X

    @staticmethod
    def _fuse_ensemble_logits(member_logits: list[np.ndarray]) -> np.ndarray:
        """Smooth noisy-OR LogMeanExp fusion preserving confident specialist calls."""
        if len(member_logits) == 1:
            return np.asarray(member_logits[0], dtype=np.float64)
        stack = np.stack([np.asarray(m, dtype=np.float64) for m in member_logits], axis=0)
        m_max = np.max(stack, axis=0)
        return m_max + np.log(np.mean(np.exp(stack - m_max), axis=0))

    @staticmethod
    def _align_reverse_logits(sc_fwd: np.ndarray, sc_rev: np.ndarray, win_size: int) -> np.ndarray:
        """Map reverse-complemented column logits back to forward boundary indices `0..M_tx-1`."""
        M_tx = len(sc_fwd)
        if M_tx >= win_size:
            return sc_rev[(M_tx - np.arange(M_tx)) % M_tx]
        aligned = np.empty_like(sc_rev)
        aligned[0] = sc_fwd[0]
        if M_tx > 1:
            aligned[1:] = sc_rev[M_tx - np.arange(1, M_tx)]
        return aligned

    def _score_transformer_with_tta(
        self,
        params: Any,
        cfg: M.Config,
        score_fn: Any,
        haplo: np.ndarray,
        x_fwd: np.ndarray,
        x_rev_cache: dict[int, np.ndarray],
        use_mirror_tta: bool,
        progress_cb: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
    ) -> np.ndarray:
        sc_fwd = self._score_transformer_static(
            params, cfg, score_fn, x_fwd, progress_cb=progress_cb, cancel_check=cancel_check
        )
        if not use_mirror_tta:
            return sc_fwd
        if cfg.W not in x_rev_cache:
            x_rev_cache[cfg.W] = self._build_circular_windows_freq(haplo[::-1].copy(), cfg.W)
        sc_rev = self._score_transformer_static(
            params, cfg, score_fn, x_rev_cache[cfg.W], progress_cb=progress_cb, cancel_check=cancel_check
        )
        return 0.5 * (sc_fwd + self._align_reverse_logits(sc_fwd, sc_rev, cfg.W))

    def _score_transformer_static(
        self,
        params: Any,
        cfg: M.Config,
        score_fn: Any,
        X: np.ndarray,
        progress_cb: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
    ) -> np.ndarray:
        """Score every transformed column using static-shape chunked JAX forward passes."""
        M_tx = X.shape[0]
        scores = np.empty(M_tx, dtype=np.float32)
        buf = np.zeros((self.CHUNK_SIZE, cfg.W, cfg.n_in), dtype=np.float32)
        for s in range(0, M_tx, self.CHUNK_SIZE):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Job cancelled by user")
            e = min(s + self.CHUNK_SIZE, M_tx)
            if progress_cb is not None:
                progress_cb(f"transformer windows {e} of {M_tx}")
            n = e - s
            if n == self.CHUNK_SIZE:
                out = score_fn(params, X[s:e])
            else:
                buf[:n] = X[s:e]
                if n < self.CHUNK_SIZE:
                    buf[n:] = 0.0
                out = score_fn(params, buf)
            out.block_until_ready()
            scores[s:e] = np.asarray(out)[:n]
        return scores

    def run_inference(
        self,
        config: InferenceConfig,
        preset_id: Optional[str] = None,
        fasta_text: Optional[str] = None,
        fasta_filename: Optional[str] = None,
        gt_text: Optional[str] = None,
        gt_filename: Optional[str] = None,
        progress_cb: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
        strict_phase2: bool = False,
    ) -> InferenceResponse:
        if progress_cb is not None:
            progress_cb("parsing")
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Job cancelled by user")

        # Load checkpoint or multi-model ensemble upfront (cached after first load)
        if config.checkpoint == "ckpt_ensemble":
            member_ids = self._resolve_ensemble_members()
            if not member_ids:
                raise ValueError("No member checkpoints available for 'ckpt_ensemble'.")
            member_models = [(mid, *self._load_checkpoint(mid)) for mid in member_ids]
            member_fns = [self._get_score_fn(p_m, c_m) for _, p_m, c_m, _ in member_models]
            cfg = member_models[0][2]
            n_model_params = sum(np_m for _, _, _, np_m in member_models)
            model_desc = f"Ensemble of {len(member_models)} models ({'+'.join(member_ids)})"
        else:
            params, cfg, n_model_params = self._load_checkpoint(config.checkpoint)
            score_fn = self._get_score_fn(params, cfg)
            member_models = [(config.checkpoint, params, cfg, n_model_params)]
            member_fns = [score_fn]
            model_desc = f"{config.checkpoint} ({n_model_params:,} params, d={cfg.d_model}, {cfg.n_layers} layers)"
        if config.use_mirror_tta:
            model_desc += " + Mirror TTA"

        # Resolve alignment input (uploaded FASTA vs preset FASTA file)
        is_known_control = False
        if fasta_text is not None:
            if not fasta_text.strip():
                raise EmptyPayloadError("Empty alignment payload")
            ds_name = (fasta_filename or "uploaded_alignment.fasta").rsplit(".", 1)[0]
            source_type = "upload"
            raw_fasta_str = fasta_text
            preset_cache_dir = None
        else:
            if not preset_id:
                raise EmptyPayloadError("Empty alignment payload")
            if preset_id not in PRESET_REGISTRY:
                raise ValueError(f"Unknown preset alignment ID: '{preset_id}'")
            pid = preset_id
            pmeta = PRESET_REGISTRY[pid]
            fpath = Path(pmeta["fasta_path"])
            if not fpath.exists():
                raise ValueError(f"Preset alignment file not found on disk: {fpath}")
            ds_name = pid
            source_type = "preset"
            raw_fasta_str = fpath.read_text(encoding="utf-8", errors="replace")
            preset_cache_dir = pmeta.get("cache_dir")
            is_known_control = bool(pmeta.get("is_control", False))

        # We start an outer timing recorder once dimensions are known; measure stage 1 directly
        import time
        t_s1_start = time.perf_counter()
        use_strict = strict_phase2 or bool(
            fasta_filename and any(fasta_filename.startswith(pfx) for pfx in ("08", "09"))
        )
        names, seqs, is_rna, gap_status = parse_and_validate_fasta(
            raw_fasta_str, strict_phase2=use_strict
        )
        try:
            _, full_headers = parse_raw_fasta_records(raw_fasta_str)
        except Exception:
            full_headers = {n: n for n in names}
        S = len(seqs)
        L = len(seqs[0])

        t_s1_ms = (time.perf_counter() - t_s1_start) * 1000.0
        gc_pct, ambig_pct = compute_alignment_base_stats(seqs)
        max_pairwise_id = compute_max_pairwise_identity(seqs)

        if cancel_check is not None and cancel_check():
            raise InterruptedError("Job cancelled by user")

        # Stage 2: Informative-site selection (strict ACGT first, gap-tolerant fallback if <30),
        # binarisation, coordinate maps & haplotype matrix
        t_s2_start = time.perf_counter()
        kept, cols, origin, info_mode = self._select_informative_and_binarise(seqs, config.null_policy)
        if len(kept) < 10:
            raise InsufficientInformativeSitesError(found=len(kept), required=10)
        maps = P.build_maps(origin, L)
        M_tx = len(cols)
        if M_tx < 10:
            raise InsufficientInformativeSitesError(found=M_tx, required=10)
        haplo = np.array([[1 if v == "1" else 0 for v in col] for col in cols], dtype=np.uint8)
        t_s2_ms = (time.perf_counter() - t_s2_start) * 1000.0
        t_after_s2 = time.perf_counter()

        recorder = TimingRecorder(
            dataset_name=ds_name,
            checkpoint=config.checkpoint,
            scoring_mode=config.scoring_mode,
            n_sequences=S,
            genome_length_bp=L,
            n_informative_cols=len(kept),
            n_transformed_cols=M_tx,
            window_size=cfg.W,
        )

        # Record already-measured Stages 1 & 2 cleanly via TimingRecorder
        info_desc = (
            "parsimony-informative ACGT sites"
            if info_mode == "strict_acgt"
            else "parsimony-informative sites (gap-tolerant fallback)"
        )
        notes_rna = ", is_rna=True (U->T normalized)" if is_rna else ""
        recorder._stages_raw.append(
            {
                "stage_id": "parse_filter",
                "stage_name": "1. FASTA Parse & Alignment Validation",
                "duration_ms": round(t_s1_ms, 2),
                "items_processed": L,
                "items_unit": "bp",
                "throughput": round(L / max(t_s1_ms / 1000.0, 1e-6), 1),
                "throughput_unit": "bp/s",
                "notes": f"{S} sequences × {L:,} bp parsed (GC={gc_pct}%, non-ACGT={ambig_pct}%{notes_rna})",
            }
        )
        recorder._stages_raw.append(
            {
                "stage_id": "binarise_coords",
                "stage_name": "2. Informative Filter, Binarisation & Coordinate Maps",
                "duration_ms": round(t_s2_ms, 2),
                "items_processed": len(kept),
                "items_unit": "cols",
                "throughput": round(len(kept) / max(t_s2_ms / 1000.0, 1e-6), 1),
                "throughput_unit": "cols/s",
                "notes": f"{len(kept):,} {info_desc} → {M_tx:,} cols ({config.null_policy})",
            }
        )

        # Ground-truth resolution (uploaded GT overrides preset GT)
        if gt_text and gt_text.strip():
            gt_items = parse_ground_truth_payload(gt_text, gt_filename or "", L)
            has_known_gt = True
        elif preset_cache_dir is not None:
            gt_items = self._load_preset_ground_truth(ds_name, Path(preset_cache_dir))
            has_known_gt = is_known_control or len(gt_items) > 0
        else:
            gt_items = []
            has_known_gt = False

        if cancel_check is not None and cancel_check():
            raise InterruptedError("Job cancelled by user")

        # Adaptive bounds & multi-scale window set (computed before Stage 3 so Stage 5 can run concurrently in hybrid_flank mode)
        eff_min_seg = max(6, min(config.min_seg, max(6, int(M_tx * 0.01))))
        eff_L_w = min(config.flank_window, max(16, M_tx // 3))
        if config.scoring_mode == "global_kl":
            eff_min_seg_global = min(max(config.min_seg, 200), max(10, M_tx // 4))
        else:
            eff_min_seg_global = eff_min_seg

        col_density = float(M_tx) / max(float(L), 1.0)
        w_tight = max(24, min(eff_L_w, int(100.0 * col_density)))
        w_short = max(24, min(eff_L_w, int(250.0 * col_density)))
        w_med = max(48, min(eff_L_w, int(500.0 * col_density)))
        scale_windows = sorted(list(set([
            w_short,
            w_med,
            eff_L_w,
        ])))
        scale_windows = [w for w in scale_windows if w >= 16 and w <= eff_L_w]
        if not scale_windows:
            scale_windows = [eff_L_w]
        candidate_windows = sorted(list(set([w_tight] + scale_windows)))
        candidate_windows = [w for w in candidate_windows if w >= 16 and w <= eff_L_w]
        if not candidate_windows:
            candidate_windows = [eff_L_w]

        need_tf = config.scoring_mode in ("hybrid_flank", "standalone_tf")
        need_kl = config.scoring_mode in ("hybrid_flank", "flank_kl", "global_kl")

        def _run_kl_stage() -> tuple[
            np.ndarray,
            dict[int, np.ndarray],
            dict[int, np.ndarray],
            dict[int, np.ndarray],
            dict[int, np.ndarray],
            float,
        ]:
            t_k0 = time.perf_counter()
            rng_local = np.random.default_rng(0)
            if config.scoring_mode == "global_kl":
                z_kl_out, _raw_kl, _null_kl = H.boundary_zscore(
                    haplo, beta=config.beta, min_seg=eff_min_seg_global, kind="kl", rng=rng_local, n_perm=2
                )
                z_map = {w: z_kl_out for w in candidate_windows}
                raw_map = {w: _raw_kl for w in candidate_windows}
                raw_max_map = {w: _raw_kl for w in candidate_windows}
                z_is_max_map = {w: np.zeros_like(_raw_kl, dtype=bool) for w in candidate_windows}
            elif config.scoring_mode in ("hybrid_flank", "flank_kl"):
                z_map = {}
                raw_map = {}
                raw_max_map = {}
                z_is_max_map = {}
                max_z_thr = max(config.z_thresh + 2.0, 8.0)

                def _dual_scan(w_scan: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    z_s, z_m, raw_s, raw_m = H.boundary_zscore_flank_dual(
                        haplo,
                        L_w=w_scan,
                        beta=config.beta,
                        min_seg=eff_min_seg,
                        power=1.0,
                        rng=np.random.default_rng(0),
                        n_perm=3,
                        origin=origin,
                    )
                    if S >= 16:
                        is_max = (z_m >= max_z_thr) & (z_m > z_s)
                        z_fused = np.where(is_max, z_m, z_s)
                        return z_fused, raw_s, raw_m, is_max
                    return z_s, raw_s, raw_m, np.zeros_like(z_s, dtype=bool)

                for w in scale_windows:
                    if cancel_check is not None and cancel_check():
                        raise InterruptedError("Job cancelled by user")
                    z_map[w], raw_map[w], raw_max_map[w], z_is_max_map[w] = _dual_scan(w)
                if w_tight not in z_map:
                    if cancel_check is not None and cancel_check():
                        raise InterruptedError("Job cancelled by user")
                    z_map[w_tight], raw_map[w_tight], raw_max_map[w_tight], z_is_max_map[w_tight] = _dual_scan(w_tight)
                if M_tx >= 64 and 16 < w_tight and 16 not in z_map and (eff_min_seg <= 16 or (16.0 / max(col_density, 1e-6)) >= 35.0):
                    if cancel_check is not None and cancel_check():
                        raise InterruptedError("Job cancelled by user")
                    z_map[16], raw_map[16], raw_max_map[16], z_is_max_map[16] = _dual_scan(16)
                z_kl_out = np.maximum.reduce([z_map[w] for w in scale_windows])
                if w_tight in z_map and w_tight not in scale_windows:
                    z_kl_out = np.maximum(z_kl_out, np.where(z_map[w_tight] >= 12.0, z_map[w_tight], 0.0))
            else:
                z_w, raw_tight, _ = H.boundary_zscore_flank(
                    haplo,
                    L_w=w_tight,
                    beta=config.beta,
                    min_seg=eff_min_seg,
                    power=1.0,
                    rng=np.random.default_rng(0),
                    n_perm=3,
                    origin=origin,
                )
                z_kl_out = np.zeros(M_tx + 1, dtype=np.float64)
                z_map = {w: z_w for w in candidate_windows}
                raw_map = {w: raw_tight for w in candidate_windows}
                raw_max_map = {w: raw_tight for w in candidate_windows}
                z_is_max_map = {w: np.zeros_like(raw_tight, dtype=bool) for w in candidate_windows}
            t_kl_ms = (time.perf_counter() - t_k0) * 1000.0
            return z_kl_out, z_map, raw_map, raw_max_map, z_is_max_map, t_kl_ms

        # Launch Stage 5 concurrently with Stage 3+4 in hybrid_flank mode
        kl_executor: Optional[ThreadPoolExecutor] = None
        kl_future = None
        if config.scoring_mode == "hybrid_flank":
            kl_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gt_kl_parallel")
            kl_future = kl_executor.submit(_run_kl_stage)

        try:
            # Stage 3: Circular Moving-Window Pairwise 4-Gamete Embedding (W=314)
            if need_tf and progress_cb is not None:
                progress_cb(f"transformer windows 0 of {M_tx}")
            with recorder.measure_stage(
                stage_id="pairwise_embed",
                stage_name="3. Circular Window 4-Gamete Pairwise Embedding",
                items_processed=M_tx if need_tf else 0,
                items_unit="windows",
                throughput_unit="win/s",
                notes=(
                    f"Circular gamete frequency tensor ({M_tx:,}, {cfg.W}, 4)"
                    if need_tf
                    else "Skipped (KL-only scoring mode)"
                ),
            ):
                X_embed = self._build_circular_windows_freq(haplo, cfg.W) if need_tf else None

            # Stage 4: Window Transformer Forward Pass (JAX JIT Encoder)
            with recorder.measure_stage(
                stage_id="transformer_fwd",
                stage_name="4. Window Transformer Forward Pass",
                items_processed=M_tx if need_tf else 0,
                items_unit="windows",
                throughput_unit="win/s",
                notes=(
                    model_desc
                    if need_tf
                    else f"Skipped (scoring_mode={config.scoring_mode})"
                ),
            ):
                if need_tf and X_embed is not None:
                    rev_cache: dict[int, np.ndarray] = {}
                    member_scores: list[np.ndarray] = []
                    for (_mid, p_m, c_m, _), fn_m in zip(member_models, member_fns):
                        x_m = X_embed if c_m.W == cfg.W else self._build_circular_windows_freq(haplo, c_m.W)
                        sc_m = self._score_transformer_with_tta(
                            p_m,
                            c_m,
                            fn_m,
                            haplo,
                            x_m,
                            rev_cache,
                            use_mirror_tta=config.use_mirror_tta,
                            progress_cb=progress_cb,
                            cancel_check=cancel_check,
                        )
                        member_scores.append(sc_m)
                    tf_raw = self._fuse_ensemble_logits(member_scores)
                    tf_logits_b = np.concatenate([tf_raw, [tf_raw[0]]]).astype(np.float64)
                else:
                    tf_logits_b = np.zeros(M_tx + 1, dtype=np.float64)
                tf_prob_b = 1.0 / (1.0 + np.exp(-np.clip(tf_logits_b, -10.0, 10.0)))
        finally:
            if kl_future is not None:
                try:
                    z_kl, z_by_w, raw_by_w, raw_max_by_w, z_is_max_by_w, t_kl_ms = kl_future.result()
                finally:
                    if kl_executor is not None:
                        kl_executor.shutdown(wait=False)

        if cancel_check is not None and cancel_check():
            raise InterruptedError("Job cancelled by user")

        if kl_future is None:
            if progress_cb is not None:
                progress_cb("KL scan")
            z_kl, z_by_w, raw_by_w, raw_max_by_w, z_is_max_by_w, t_kl_ms = _run_kl_stage()

        kl_items = (M_tx + 1) if need_kl else 0
        kl_notes = (
            f"Legacy global KL (min_seg={eff_min_seg_global}, beta={config.beta:g})"
            if config.scoring_mode == "global_kl"
            else (
                f"Local flank_kl (windows={candidate_windows}, min_seg={eff_min_seg}, beta={config.beta:g}, 3x col-perm null)"
                if need_kl
                else "Skipped global null scan (standalone_tf mode)"
            )
        )
        recorder._stages_raw.append(
            {
                "stage_id": "flank_kl_null",
                "stage_name": "5. Flanking KL & Permutation Null Calibration",
                "duration_ms": round(t_kl_ms, 2),
                "items_processed": kl_items,
                "items_unit": "boundaries",
                "throughput": round(kl_items / max(t_kl_ms / 1000.0, 1e-6), 1) if kl_items > 0 and t_kl_ms > 0 else None,
                "throughput_unit": "bound/s",
                "notes": kl_notes,
            }
        )

        if cancel_check is not None and cancel_check():
            raise InterruptedError("Job cancelled by user")

        # Stage 6: Peak Picking, Consolidation & 3-Level Carrier/Parent Attribution
        if progress_cb is not None:
            progress_cb("rendering")
        with recorder.measure_stage(
            stage_id="consolidate_3level",
            stage_name="6. Peak Consolidation & 3-Level Attribution",
            items_processed=0,
            items_unit="peaks",
            throughput_unit="peaks/s",
            notes="",
        ) as s6_meta:
            events, raw_peaks, track_profile = self._extract_and_consolidate_events(
                config=config,
                haplo=haplo,
                maps=maps,
                kept=kept,
                L=L,
                seq_names=names,
                z_kl=z_kl,
                tf_logits_b=tf_logits_b,
                tf_prob_b=tf_prob_b,
                candidate_windows=candidate_windows,
                z_by_w=z_by_w,
                raw_by_w=raw_by_w,
                raw_max_by_w=raw_max_by_w,
                z_is_max_by_w=z_is_max_by_w,
                eff_min_seg=eff_min_seg,
                eff_min_seg_global=eff_min_seg_global,
            )
            s6_meta["items_processed"] = max(1, len(raw_peaks))
            s6_meta["notes"] = (
                f"{len(raw_peaks)} raw peak(s) → {len(events)} consolidated event(s) "
                f"(merge_radius={config.merge_radius_bp} bp)"
            )

        wall_total_ms = t_s1_ms + t_s2_ms + (time.perf_counter() - t_after_s2) * 1000.0
        timings = recorder.finalize(wall_total_ms=wall_total_ms)
        self.timing_store.record(timings)

        # Store alignment sequences in bounded LRU cache for nucleotide alignment viewer
        self._store_alignment_cache(timings.run_id, names, seqs, kept, L)

        diagnostics = evaluate_predictions_against_ground_truth(
            consolidated_events=events,
            gt_raw_items=gt_items,
            track_profile=track_profile,
            genome_length_bp=L,
            active_tolerance_bp=config.eval_tolerance_bp,
            is_known_control=is_known_control or (has_known_gt and len(gt_items) == 0),
        )
        if max_pairwise_id < 0.45:
            diagnostics.summary_notes.append(
                f"Low sequence identity (max pairwise identity {max_pairwise_id * 100:.1f}% < 45%). "
                "Alignment exhibits high divergence; review signal attribution."
            )

        left_edge_bp = int(origin[min(eff_min_seg - 1, M_tx - 1)]) + 1
        right_edge_bp = max(1, int(L - int(origin[max(0, M_tx - eff_min_seg)])))
        edge_dead_zone_bp = max(30, left_edge_bp, right_edge_bp)

        warnings_list: list[str] = []
        warnings_list.append(
            f"Edge detection limit: breakpoints within ~{edge_dead_zone_bp} bp of either alignment end "
            f"(columns 1–{left_edge_bp} and {max(1, L - right_edge_bp)}–{L}, requiring at least "
            f"{eff_min_seg} informative sites per flank) cannot be detected."
        )
        if is_rna:
            warnings_list.append("RNA input detected (U->T normalized)")
        if info_mode == "gap_tolerant":
            warnings_list.append("Gap-tolerant fallback engaged (alignment columns contain > 0% and <= 25% gaps)")
        if M_tx < 314:
            warnings_list.append(
                f"Alignment has M_tx={M_tx} < 314 informative sites (linear zero-padding applied; heuristic-only). "
                f"Alignment length ({M_tx} transformed cols) is shorter than model window (W=314); local heuristic prioritized."
            )
        if any(ev.signal_attribution == "flank_kl_heuristic_only" for ev in events):
            warnings_list.append(
                "One or more predicted events rely solely on local flanking KL heuristic "
                "(negative or subthreshold transformer logit)."
            )
        if any(ev.tf_logit < 0.0 for ev in events):
            warnings_list.append(
                "Out-of-distribution signal: one or more predicted events have negative transformer logits (P(bp) < 0.50); "
                "local flanking KL heuristic overrides model uncertainty."
            )
        if max_pairwise_id < 0.45:
            warnings_list.append(
                f"Low sequence identity ({max_pairwise_id * 100:.1f}% < 45%): alignment indicates high divergence or unlinked composition."
            )

        return InferenceResponse(
            warnings=warnings_list,
            engine_mode="live",
            model_param_count=n_model_params,
            config=config,
            alignment=AlignmentMetadata(
                dataset_name=ds_name,
                source_type=source_type,  # type: ignore[arg-type]
                n_sequences=S,
                genome_length_bp=L,
                n_informative_cols=len(kept),
                n_transformed_cols=M_tx,
                window_size=cfg.W,
                null_policy=config.null_policy,
                sequence_names=names,
                full_headers=full_headers,
                gc_content_pct=gc_pct,
                ambiguity_pct=ambig_pct,
                is_rna=is_rna,
                edge_dead_zone_bp=edge_dead_zone_bp,
            ),
            timings=timings,
            consolidated_events=events,
            raw_peaks=raw_peaks,
            diagnostics=diagnostics,
            track_profile=track_profile,
        )

    def _extract_and_consolidate_events(
        self,
        config: InferenceConfig,
        haplo: np.ndarray,
        maps: dict[str, Any],
        kept: list[int],
        L: int,
        seq_names: list[str],
        z_kl: np.ndarray,
        tf_logits_b: np.ndarray,
        tf_prob_b: np.ndarray,
        candidate_windows: list[int],
        z_by_w: dict[int, np.ndarray],
        raw_by_w: dict[int, np.ndarray],
        raw_max_by_w: dict[int, np.ndarray],
        z_is_max_by_w: dict[int, np.ndarray],
        eff_min_seg: int,
        eff_min_seg_global: int,
    ) -> tuple[list[ConsolidatedEvent], list[RawPeakCall], GenomeTrackProfile]:
        """Pick peaks, run Level-2/3 switch attribution, consolidate via report.consolidate_peaks."""
        M_tx, S = haplo.shape
        gate_linkage = np.clip((z_kl - 2.5) / 2.5, 0.0, 1.0)
        hybrid_profile = np.maximum(z_kl, 0.0) + np.where(
            tf_logits_b > 0.5, 7.0 * (tf_logits_b - 0.5), 0.0
        ) * gate_linkage

        w_tight = min(candidate_windows) if candidate_windows else 16

        def _raw_arr_for_window(w_pick: int, sub_idx: np.ndarray) -> np.ndarray:
            peak_local = int(sub_idx[np.argmax(z_by_w[w_pick][sub_idx])])
            use_max = bool(z_is_max_by_w[w_pick][peak_local]) if w_pick in z_is_max_by_w else False
            return raw_max_by_w[w_pick] if (use_max and w_pick in raw_max_by_w) else raw_by_w[w_pick]

        def _refine_kl_peaks(kl_peaks_raw: list[int]) -> list[int]:
            # First identify genuine peaks (filtering out greedy pick_peaks mask cliffs on monotonic slopes)
            genuine_raw: list[int] = []
            for p in kl_peaks_raw:
                p_int = int(p)
                lo = max(eff_min_seg, p_int - 8)
                hi = min(M_tx - eff_min_seg, p_int + 8)
                if lo > hi:
                    continue
                sub_idx = np.arange(lo, hi + 1)
                best_w = candidate_windows[0]
                best_z = -1.0
                for w in candidate_windows:
                    max_z_w = float(np.max(z_by_w[w][sub_idx]))
                    if max_z_w > best_z:
                        best_z = max_z_w
                        best_w = w
                raw_arr = _raw_arr_for_window(best_w, sub_idx)
                best_b = int(sub_idx[np.argmax(raw_arr[sub_idx])])
                tight_max_z = float(np.max(z_by_w[w_tight][sub_idx])) if w_tight in z_by_w else best_z
                if tight_max_z < config.z_thresh:
                    if (best_b == lo and lo > eff_min_seg and raw_arr[lo - 1] > raw_arr[best_b]) or (
                        best_b == hi and hi < (M_tx - eff_min_seg) and raw_arr[hi + 1] > raw_arr[best_b]
                    ):
                        continue
                    if (lo > eff_min_seg and z_kl[lo - 1] > z_kl[best_b] * 1.02) or (
                        hi < (M_tx - eff_min_seg) and z_kl[hi + 1] > z_kl[best_b] * 1.02
                    ):
                        continue
                genuine_raw.append(p_int)

            refined: list[int] = []
            for p_int in genuine_raw:
                lo = max(eff_min_seg, p_int - 8)
                hi = min(M_tx - eff_min_seg, p_int + 8)
                sub_idx = np.arange(lo, hi + 1)
                min_dist_other = min([abs(q - p_int) for q in genuine_raw if q != p_int], default=M_tx)
                valid_ws = [
                    w for w in candidate_windows
                    if w <= (min_dist_other + 4) and float(np.max(z_by_w[w][sub_idx])) >= config.z_thresh
                ]
                if not valid_ws:
                    valid_ws = [w_tight] if (w_tight in z_by_w and float(np.max(z_by_w[w_tight][sub_idx])) >= config.z_thresh) else candidate_windows
                best_w = valid_ws[0]
                best_z = -1.0
                for w in valid_ws:
                    max_z_w = float(np.max(z_by_w[w][sub_idx]))
                    if max_z_w > best_z:
                        best_z = max_z_w
                        best_w = w
                raw_arr = _raw_arr_for_window(best_w, sub_idx)
                best_b = int(sub_idx[np.argmax(raw_arr[sub_idx])])
                refined.append(best_b)
            return sorted(list(set(refined)))

        # Pick candidate peaks depending on scoring mode
        peak_sources: dict[int, str] = {}

        def _rescue_micro_peaks(base_peaks: list[int]) -> list[int]:
            if 16 not in z_by_w:
                return base_peaks
            z16 = z_by_w[16]
            micro_raw = H.pick_peaks(z16, max(14.0, config.z_thresh + 8.0), radius=10, max_peaks=16)
            if not micro_raw:
                return base_peaks
            out_peaks = list(base_peaks)
            existing_origs = [
                int(P.unshrink(P.normalise_boundary(int(b), maps), maps, L)) for b in out_peaks
            ]
            for bm in micro_raw:
                bm_int = int(bm)
                bm_orig = int(P.unshrink(P.normalise_boundary(bm_int, maps), maps, L))
                nearest_idx = min(
                    range(len(out_peaks)),
                    key=lambda idx: abs(existing_origs[idx] - bm_orig),
                    default=None,
                )
                if nearest_idx is None or abs(existing_origs[nearest_idx] - bm_orig) > 35:
                    if len(out_peaks) < 64:
                        out_peaks.append(bm_int)
                        existing_origs.append(bm_orig)
                        peak_sources[bm_int] = "micro_flank"
                elif (
                    abs(existing_origs[nearest_idx] - bm_orig) <= 25
                    and float(z16[bm_int]) >= 18.0
                    and float(z_kl[out_peaks[nearest_idx]]) < 25.0
                ):
                    old_b = out_peaks[nearest_idx]
                    peak_sources.pop(old_b, None)
                    out_peaks[nearest_idx] = bm_int
                    existing_origs[nearest_idx] = bm_orig
                    peak_sources[bm_int] = "micro_flank"
            return sorted(list(set(out_peaks)))

        if config.scoring_mode == "flank_kl":
            kl_peaks_raw = H.pick_peaks(z_kl, config.z_thresh, radius=15, max_peaks=64)
            kl_peaks = _refine_kl_peaks(kl_peaks_raw)
            for b in kl_peaks:
                peak_sources[int(b)] = "flank_kl"
            ordered_peaks = _rescue_micro_peaks(kl_peaks)
            pick_by = "prod"
        elif config.scoring_mode == "global_kl":
            kl_peaks = H.pick_peaks(z_kl, config.z_thresh, radius=20, max_peaks=64)
            for b in kl_peaks:
                peak_sources[int(b)] = "global_kl"
            ordered_peaks = kl_peaks
            pick_by = "score"
        elif config.scoring_mode == "standalone_tf":
            # In standalone_tf mode, the W=314 window scores 4-gamete regions; pin exact switch boundaries
            # inside Transformer-positive regions using tight local haplotype switch z-score
            z_tight = z_by_w.get(w_tight)
            if z_tight is not None and np.any(tf_logits_b >= config.tf_logit_thresh):
                tf_pos_mask = np.zeros(M_tx + 1, dtype=bool)
                pos_indices = np.flatnonzero(tf_logits_b >= config.tf_logit_thresh)
                for idx_p in pos_indices:
                    tf_pos_mask[max(0, int(idx_p) - 10) : min(M_tx + 1, int(idx_p) + 11)] = True
                gated_z = np.where(tf_pos_mask, z_tight, 0.0)
                tf_peaks_raw = H.pick_peaks(gated_z, config.z_thresh, radius=15, max_peaks=64)
                tf_peaks = _refine_kl_peaks(tf_peaks_raw) if tf_peaks_raw else []
                if not tf_peaks:
                    tf_peaks = H.pick_peaks(tf_logits_b, config.tf_logit_thresh, radius=20, max_peaks=64)
            else:
                tf_peaks = H.pick_peaks(tf_logits_b, config.tf_logit_thresh, radius=20, max_peaks=64)
            for b in tf_peaks:
                peak_sources[int(b)] = "transformer"
            ordered_peaks = sorted(list(set(int(b) for b in tf_peaks)))
            pick_by = "score"
        else:
            # hybrid_flank: combine calibrated flank_kl peaks + micro short-tract peaks + non-overlapping confirmed TF peaks
            kl_peaks_raw = H.pick_peaks(z_kl, config.z_thresh, radius=15, max_peaks=64)
            kl_peaks = _refine_kl_peaks(kl_peaks_raw)
            for b in kl_peaks:
                peak_sources[int(b)] = "flank_kl"
            kl_peaks = _rescue_micro_peaks(kl_peaks)
            tf_peaks = H.pick_peaks(tf_logits_b, config.tf_logit_thresh, radius=20, max_peaks=64)
            merged_peaks = list(kl_peaks)
            tf_rescue_thr = max(config.tf_logit_thresh, 1.2)
            for b_tf in tf_peaks:
                b_tf_int = int(b_tf)
                near_kl = [k for k in kl_peaks if abs(int(k) - b_tf_int) <= 20]
                if near_kl:
                    near_k = min(near_kl, key=lambda k: abs(int(k) - b_tf_int))
                    peak_sources[int(near_k)] = "hybrid"
                elif float(tf_logits_b[b_tf_int]) >= tf_rescue_thr and all(abs(b_tf_int - int(b_m)) > 20 for b_m in merged_peaks):
                    lo_pin = max(eff_min_seg, b_tf_int - 10)
                    hi_pin = min(M_tx - eff_min_seg, b_tf_int + 10)
                    z_tight_arr = z_by_w.get(w_tight)
                    if lo_pin <= hi_pin and z_tight_arr is not None:
                        sub_pin = np.arange(lo_pin, hi_pin + 1)
                        b_pin = int(sub_pin[np.argmax(z_tight_arr[sub_pin])])
                        if (
                            float(z_tight_arr[b_pin]) >= config.z_thresh
                            and float(z_kl[b_pin]) >= 5.0
                            and all(abs(b_pin - int(b_m)) > 20 for b_m in merged_peaks)
                            and len(merged_peaks) < 64
                        ):
                            merged_peaks.append(b_pin)
                            peak_sources[b_pin] = "transformer"
            ordered_peaks = sorted(merged_peaks)
            pick_by = "prod"

        # Precompute full-alignment pairwise sequence identity matrix for clade-aware filtering
        full_id = np.mean(haplo[:, :, None] == haplo[:, None, :], axis=0) if M_tx > 0 else np.eye(S, dtype=np.float64)

        raw_for_cons: list[dict[str, Any]] = []
        raw_peak_meta: dict[int, dict[str, Any]] = {}

        for b in ordered_peaks:
            b_int = int(b)
            src = peak_sources.get(b_int, config.scoring_mode)
            if config.scoring_mode == "global_kl":
                detail = H.soft_switch_detail(
                    haplo,
                    b_int,
                    beta=config.beta,
                    min_seg=eff_min_seg_global,
                    kind="kl",
                    top_members=20,
                    member_frac=0.5,
                    top_parents=5,
                )
                dyn_min_sw = 0.0
            else:
                best_detail = []
                best_sw = 0.0
                std_windows = [w for w in candidate_windows if w > w_tight] or candidate_windows
                best_w_used = std_windows[0] if std_windows else eff_min_seg
                for w in std_windows:
                    dt = H.flank_soft_switch_detail(
                        haplo,
                        b_int,
                        L_w=w,
                        beta=config.beta,
                        min_seg=eff_min_seg,
                        kind="kl",
                        top_members=20,
                        member_frac=0.5,
                        top_parents=5,
                    )
                    if dt and dt[0]["switch"] > best_sw:
                        best_sw = dt[0]["switch"]
                        best_detail = dt
                        best_w_used = w

                # When tight/micro window has strong independent z-score (>= 15.0), use it for carrier/parent attribution
                # so adjacent breakpoints ~60-100 bp away do not bleed into the flanking window
                if 16 in z_by_w and (src == "micro_flank" or (float(z_by_w[16][b_int]) >= 18.0 and float(z_kl[b_int]) < 25.0)):
                    dt_micro = H.flank_soft_switch_detail(
                        haplo,
                        b_int,
                        L_w=16,
                        beta=config.beta,
                        min_seg=min(eff_min_seg, 6),
                        kind="kl",
                        top_members=20,
                        member_frac=0.5,
                        top_parents=5,
                    )
                    if dt_micro and dt_micro[0]["switch"] >= max(best_sw, 3.0):
                        best_detail = dt_micro
                        best_sw = dt_micro[0]["switch"]
                        best_w_used = 16
                elif w_tight in z_by_w and float(z_by_w[w_tight][b_int]) >= 15.0:
                    dt_tight = H.flank_soft_switch_detail(
                        haplo,
                        b_int,
                        L_w=w_tight,
                        beta=config.beta,
                        min_seg=eff_min_seg,
                        kind="kl",
                        top_members=20,
                        member_frac=0.5,
                        top_parents=5,
                    )
                    if dt_tight and dt_tight[0]["switch"] >= 0.85 * best_sw and dt_tight[0]["switch"] >= 3.0:
                        best_detail = dt_tight
                        best_w_used = w_tight

                detail = best_detail
                if not detail:
                    continue

                # Filter out reciprocal mirror-donor artifacts from detail:
                filtered_detail = [detail[0]]
                higher_seqs = {int(detail[0]["recombinant"])}
                higher_parents = {int(detail[0]["parent_left"]), int(detail[0]["parent_right"])}
                for cand in detail[1:]:
                    c_seq = int(cand["recombinant"])
                    c_pL = int(cand["parent_left"])
                    c_pR = int(cand["parent_right"])
                    one_side_mirror = (c_pL in higher_seqs) != (c_pR in higher_seqs)
                    ext_l = [x for x in cand.get("left_candidates", []) if int(x["seq"]) not in higher_seqs]
                    ext_r = [x for x in cand.get("right_candidates", []) if int(x["seq"]) not in higher_seqs]
                    ext_same_clade = bool(ext_l and ext_r and int(ext_l[0]["seq"]) == int(ext_r[0]["seq"]))
                    if one_side_mirror or (c_seq in higher_parents and ext_same_clade):
                        continue
                    filtered_detail.append(cand)
                    higher_seqs.add(c_seq)
                    higher_parents.add(c_pL)
                    higher_parents.add(c_pR)
                detail = filtered_detail

                z_val = max(float(z_kl[b_int]), float(z_by_w[16][b_int])) if src == "micro_flank" and 16 in z_by_w else float(z_kl[b_int])
                tf_peak = float(tf_logits_b[b_int])
                lo_tf = max(0, b_int - 20)
                hi_tf = min(M_tx + 1, b_int + 21)
                local_max_tf = float(np.max(tf_logits_b[lo_tf:hi_tf]))
                w_eff = min(best_w_used, b_int, M_tx - b_int)
                b_orig = int(P.unshrink(P.normalise_boundary(b_int, maps), maps, L))

                # Clonal-sibling-aware precision filters (Filters A & B)
                r_primary = detail[0]
                r_idx0 = int(r_primary["recombinant"])
                w_chk_ext = min(best_w_used, b_int, M_tx - b_int, 50)
                clonal_sibs: set[int] = set()
                dc_ext = 0.0
                ext_L_clean = int(r_primary["parent_left"])
                ext_R_clean = int(r_primary["parent_right"])

                def _compute_dc_ext(w_win: int) -> tuple[float, set[int], int, int]:
                    lh_e = haplo[b_int - w_win : b_int]
                    rh_e = haplo[b_int : b_int + w_win]
                    sim_L = np.mean(lh_e == lh_e[:, r_idx0 : r_idx0 + 1], axis=0)
                    sim_R = np.mean(rh_e == rh_e[:, r_idx0 : r_idx0 + 1], axis=0)
                    sibs = {
                        s_i for s_i in range(S)
                        if s_i != r_idx0 and float(sim_L[s_i]) >= 0.995 and float(sim_R[s_i]) >= 0.995
                    }
                    excl_ext = {int(x["recombinant"]) for x in detail} | sibs | {r_idx0}
                    avail_ext = [s_i for s_i in range(S) if s_i not in excl_ext]
                    if not avail_ext:
                        avail_ext = [s_i for s_i in range(S) if s_i != r_idx0]
                    if not avail_ext:
                        return 0.0, sibs, ext_L_clean, ext_R_clean
                    eL = max(avail_ext, key=lambda s_i: float(sim_L[s_i]))
                    eR = max(avail_ext, key=lambda s_i: float(sim_R[s_i]))
                    val = (
                        (float(sim_L[eL]) - float(sim_L[eR]))
                        + (float(sim_R[eR]) - float(sim_R[eL]))
                    )
                    return val, sibs, eL, eR

                if w_chk_ext >= 8:
                    dc_ext_short, clonal_sibs, ext_L_clean, ext_R_clean = _compute_dc_ext(w_chk_ext)
                    dc_ext = dc_ext_short
                    if w_eff > 50:
                        dc_ext_full, _, ext_L_full, ext_R_full = _compute_dc_ext(w_eff)
                        if dc_ext_full > dc_ext:
                            dc_ext = dc_ext_full
                            ext_L_clean, ext_R_clean = ext_L_full, ext_R_full
                    if dc_ext < 0.15 and src != "transformer":
                        continue

                    # Filter B: Mid-tract continuity / jump-over veto when wide flank similarity drops < 0.72
                    if min(float(r_primary.get("left_sim", 1.0)), float(r_primary.get("right_sim", 1.0))) < 0.72:
                        w_loc = min(w_tight, 24, b_int, M_tx - b_int)
                        if w_loc >= 8:
                            lh_loc = haplo[b_int - w_loc : b_int]
                            rh_loc = haplo[b_int : b_int + w_loc]
                            sim_Ll = np.mean(lh_loc == lh_loc[:, r_idx0 : r_idx0 + 1], axis=0)
                            sim_Rl = np.mean(rh_loc == rh_loc[:, r_idx0 : r_idx0 + 1], axis=0)
                            non_sibs = [s_i for s_i in range(S) if s_i != r_idx0 and s_i not in clonal_sibs]
                            max_both = max((min(float(sim_Ll[s_i]), float(sim_Rl[s_i])) for s_i in non_sibs), default=0.0)
                            if max_both >= 0.92:
                                continue

                # Distinct parent check: local detail switch OR external non-carrier parent switch
                distinct_parent_recombs = [r for r in detail if int(r["parent_left"]) != int(r["parent_right"])]
                has_distinct_parents = bool(distinct_parent_recombs) or (ext_L_clean != ext_R_clean and dc_ext >= 0.15)
                if not has_distinct_parents and src != "transformer":
                    continue

                # Compute multi-site contiguous tract evidence for top distinct recombinant or external parent pair
                n_tract_support = 0
                min_side_support = 0
                if distinct_parent_recombs or has_distinct_parents:
                    if distinct_parent_recombs:
                        r_top = distinct_parent_recombs[0]
                        r_idx = int(r_top["recombinant"])
                        pL = int(r_top["parent_left"])
                        pR = int(r_top["parent_right"])
                    else:
                        r_idx, pL, pR = r_idx0, ext_L_clean, ext_R_clean
                    w_chk = min(w_eff, 50)
                    if w_chk > 0:
                        left_H = haplo[max(0, b_int - w_chk) : b_int]
                        left_diff = (left_H[:, pL] != left_H[:, pR])
                        n_left_pL = int(np.sum((left_H[:, r_idx] == left_H[:, pL]) & left_diff))
                        right_H = haplo[b_int : min(M_tx, b_int + w_chk)]
                        right_diff = (right_H[:, pL] != right_H[:, pR])
                        n_right_pR = int(np.sum((right_H[:, r_idx] == right_H[:, pR]) & right_diff))
                        n_tract_support = n_left_pL + n_right_pR
                        min_side_support = min(n_left_pL, n_right_pR)

                # Calibrated soft-switch evidence thresholding for orphan donors:
                # min_sw = 0.35 for z >= 15.0, 0.50 for z >= 10.0, 0.70 for z >= 8.0, 1.0 otherwise.
                if z_val >= 15.0:
                    dyn_min_sw = 0.35
                elif z_val >= 10.0:
                    dyn_min_sw = 0.50
                elif z_val >= 8.0:
                    dyn_min_sw = 0.70
                else:
                    dyn_min_sw = 1.0

                if config.scoring_mode == "hybrid_flank":
                    if src == "transformer":
                        # Transformer-pinned peaks require strong local TF support, tight-window switch, linkage & tract evidence
                        z_tight_ok = (b_int < len(z_by_w[w_tight]) and float(z_by_w[w_tight][b_int]) >= config.z_thresh) if w_tight in z_by_w else True
                        if (
                            local_max_tf < max(config.tf_logit_thresh, 1.2)
                            or not z_tight_ok
                            or z_kl[b_int] < 5.0
                            or best_sw < 1.0
                            or dc_ext < 0.20
                            or not has_distinct_parents
                            or n_tract_support < 3
                            or min_side_support < 1
                        ):
                            continue
                    else:
                        # Negative logit veto:
                        # If Transformer assigns tf_peak <= -1.0 or local_max_tf <= 0.0,
                        # reject unless corroborated by contiguous multi-site tract evidence (>= 3 supporting sites, min side >= 1, best_sw >= 1.5).
                        if tf_peak <= -1.0 or local_max_tf <= 0.0:
                            has_tract_corroboration = (
                                has_distinct_parents
                                and n_tract_support >= 3
                                and min_side_support >= 1
                                and best_sw >= 1.5
                            )
                            if not has_tract_corroboration:
                                continue

                        # Short / low-polymorphism alignments (M_tx < 314):
                        # Transformer operates with zero-padding (heuristic-only); require robust heuristic evidence.
                        if M_tx < 314:
                            if z_val < 10.5 or best_sw < 2.0:
                                continue
                        else:
                            # Marginal KL peaks (z < 8.0) on full-length alignments:
                            # Require local transformer logit >= config.tf_logit_thresh or best_sw >= 1.0.
                            if z_val < 8.0:
                                if local_max_tf < config.tf_logit_thresh and best_sw < 1.0:
                                    continue

                        # Truncated flanking windows (w_eff < 48 or b_orig <= 50 or b_orig >= L - 50):
                        # Require best_sw >= 1.5 or local_max_tf >= 0.5, and if z_val < 7.0 require tight-window confirmation.
                        if w_eff < 48 or b_orig <= 50 or b_orig >= (L - 50):
                            if best_sw < 1.5 and local_max_tf < 0.5:
                                continue
                            if z_val < 7.0 and w_tight in z_by_w and float(z_by_w[w_tight][b_int]) < config.z_thresh:
                                continue

                elif config.scoring_mode == "flank_kl":
                    # Truncated flanking windows and short alignments under flank_kl mode
                    if w_eff < 48 or b_orig <= 50 or b_orig >= (L - 50):
                        if best_sw < 1.5:
                            continue
                        if z_val < 7.0 and w_tight in z_by_w and float(z_by_w[w_tight][b_int]) < config.z_thresh:
                            continue
                    if M_tx < 314:
                        if z_val < 10.5 or best_sw < 2.0:
                            continue

                if best_sw < dyn_min_sw:
                    continue

            if not detail:
                continue

            b_orig = int(P.unshrink(P.normalise_boundary(b_int, maps), maps, L))
            if config.scoring_mode == "standalone_tf":
                sc_val = float(tf_logits_b[b_int])
            elif config.scoring_mode == "hybrid_flank":
                sc_val = float(hybrid_profile[b_int]) if src == "transformer" else float(z_kl[b_int])
            else:
                sc_val = float(z_kl[b_int])

            eff_z_val = max(float(z_kl[b_int]), float(z_by_w[16][b_int])) if src == "micro_flank" and 16 in z_by_w else float(z_kl[b_int])
            raw_for_cons.append(
                {
                    "boundary_tx": b_int,
                    "score": float(hybrid_profile[b_int]) if src == "transformer" else eff_z_val,
                    "breakpoint_orig": b_orig,
                    "n_switch": len(detail),
                    "recombinants": detail,
                }
            )
            raw_peak_meta[b_int] = {
                "source": src,
                "z_kl": eff_z_val,
                "tf_logit": float(tf_logits_b[b_int]),
                "tf_prob": float(tf_prob_b[b_int]),
                "hybrid": float(hybrid_profile[b_int]),
            }

        # Filter C: Clade-aware shoulder echo suppression within 220 bp (respecting opposite-switch boundaries)
        if len(raw_for_cons) >= 2:
            def _ext_pair(rp: dict[str, Any]) -> tuple[int, int, int, set[int]]:
                r_l = rp.get("recombinants", [])
                r0 = r_l[0]
                cset = {int(x["recombinant"]) for x in r_l}
                r_idx = int(r0["recombinant"])
                sibs = {s_i for s_i in range(S) if full_id[r_idx, s_i] >= 0.995}
                excl = cset | sibs
                el = [x for x in r0.get("left_candidates", []) if int(x["seq"]) not in excl] or r0.get("left_candidates", [])
                er = [x for x in r0.get("right_candidates", []) if int(x["seq"]) not in excl] or r0.get("right_candidates", [])
                pL = int(el[0]["seq"]) if el else int(r0["parent_left"])
                pR = int(er[0]["seq"]) if er else int(r0["parent_right"])
                return r_idx, pL, pR, cset

            peak_info = [_ext_pair(rp) for rp in raw_for_cons]
            filtered_raw: list[dict[str, Any]] = []
            for idx_p, rp in enumerate(raw_for_cons):
                r_i, pLi, pRi, cset_i = peak_info[idx_p]
                r0 = rp["recombinants"][0]
                bp_i = int(rp["breakpoint_orig"])
                sc_i = float(rp["score"]) * float(r0.get("switch", 1.0))
                is_shoulder_echo = False
                for idx_q, rq in enumerate(raw_for_cons):
                    if idx_q == idx_p:
                        continue
                    r_q, pLq, pRq, cset_q = peak_info[idx_q]
                    rq0 = rq["recombinants"][0]
                    bp_q = int(rq["breakpoint_orig"])
                    sc_q = float(rq["score"]) * float(rq0.get("switch", 1.0))
                    if abs(bp_i - bp_q) > 220 or sc_q <= sc_i:
                        continue
                    same_carrier = bool(cset_i & cset_q) or float(full_id[r_i, r_q]) >= 0.98
                    same_dir = float(full_id[pLi, pLq]) >= 0.92 and float(full_id[pRi, pRq]) >= 0.92
                    if not (same_carrier and same_dir):
                        continue
                    lo_bp, hi_bp = min(bp_i, bp_q), max(bp_i, bp_q)
                    has_opp_between = any(
                        lo_bp < int(rm["breakpoint_orig"]) < hi_bp
                        and (bool(cset_i & peak_info[idx_m][3]) or float(full_id[r_i, peak_info[idx_m][0]]) >= 0.98)
                        and float(full_id[pLi, peak_info[idx_m][2]]) >= 0.92
                        and float(full_id[pRi, peak_info[idx_m][1]]) >= 0.92
                        for idx_m, rm in enumerate(raw_for_cons)
                    )
                    if not has_opp_between:
                        is_shoulder_echo = True
                        break
                if not is_shoulder_echo:
                    filtered_raw.append(rp)
            raw_for_cons = filtered_raw

        cons_list = REP.consolidate_peaks(
            raw_for_cons,
            merge_radius_bp=config.merge_radius_bp,
            pick_by=pick_by,
        )

        # Phase 3 Step 5: Enforce paired-breakpoint carrier rule on internal tracts
        # When an alignment has paired breakpoints (at least one sequence carries >= 2 events),
        # remove secondary single-breakpoint leak candidates from events whose primary carrier is paired.
        if len(cons_list) >= 2:
            seq_ev_counts: dict[int, int] = {}
            for ev in cons_list:
                for r in ev.get("recombinants", []):
                    s_id = int(r["recombinant"])
                    seq_ev_counts[s_id] = seq_ev_counts.get(s_id, 0) + 1
            if any(cnt >= 2 for cnt in seq_ev_counts.values()):
                for ev in cons_list:
                    r_list = ev.get("recombinants", [])
                    if len(r_list) > 1 and seq_ev_counts.get(int(r_list[0]["recombinant"]), 0) >= 2:
                        kept_r = [r_list[0]]
                        for r_sec in r_list[1:]:
                            s_sec = int(r_sec["recombinant"])
                            if seq_ev_counts.get(s_sec, 0) >= 2:
                                kept_r.append(r_sec)
                        ev["recombinants"] = kept_r
                        ev["all_carrier_seqs"] = sorted(list({int(r["recombinant"]) for r in kept_r}))

        # Map each raw peak's orig coordinate to its consolidated event_id
        orig_to_event_id: dict[int, int] = {}
        for ev in cons_list:
            for bp_c in ev.get("constituent_breakpoints_orig", []):
                orig_to_event_id[int(bp_c)] = int(ev["event_id"])

        def sname(idx: int) -> str:
            return seq_names[idx] if 0 <= idx < len(seq_names) else f"Seq_{idx}"

        def format_parent_with_ties(ext_cands: list[dict[str, Any]], fallback_idx: int) -> str:
            if not ext_cands:
                return sname(fallback_idx)
            c0 = ext_cands[0]
            name0 = sname(int(c0["seq"]))
            if len(ext_cands) >= 2:
                c1 = ext_cands[1]
                sim0 = float(c0["sim"])
                sim1 = float(c1["sim"])
                margin = 0.01 if sim0 >= 0.999 else 0.05
                if sim0 >= 0.80 and (sim0 - sim1) <= margin:
                    name1 = sname(int(c1["seq"]))
                    if name1 != name0:
                        return f"{name0} or {name1}"
            return name0

        consolidated_events: list[ConsolidatedEvent] = []
        for ev in cons_list:
            b_tx = int(ev.get("boundary_tx", 0))
            p_meta = raw_peak_meta.get(
                b_tx,
                {
                    "z_kl": float(z_kl[b_tx]) if 0 <= b_tx < len(z_kl) else 0.0,
                    "tf_logit": float(tf_logits_b[b_tx]) if 0 <= b_tx < len(tf_logits_b) else 0.0,
                    "tf_prob": float(tf_prob_b[b_tx]) if 0 <= b_tx < len(tf_prob_b) else 0.5,
                    "hybrid": float(hybrid_profile[b_tx]) if 0 <= b_tx < len(hybrid_profile) else 0.0,
                },
            )
            carrier_set = set(int(s) for s in ev.get("all_carrier_seqs", []))
            recombs_out: list[RecombinantCall] = []
            for r in ev.get("recombinants", []):
                r_idx = int(r["recombinant"])
                pL = int(r["parent_left"])
                pR = int(r["parent_right"])
                l_cands = r.get("left_candidates", [])
                r_cands = r.get("right_candidates", [])
                sibs_r = {s_i for s_i in range(S) if s_i != r_idx and float(full_id[r_idx, s_i]) >= 0.995}
                ext_mask = carrier_set | sibs_r
                ext_l_cands = [c for c in l_cands if int(c["seq"]) not in ext_mask]
                if not ext_l_cands:
                    ext_l_cands = [c for c in l_cands if int(c["seq"]) not in carrier_set]
                ext_r_cands = [c for c in r_cands if int(c["seq"]) not in ext_mask]
                if not ext_r_cands:
                    ext_r_cands = [c for c in r_cands if int(c["seq"]) not in carrier_set]
                ext_L = int(ext_l_cands[0]["seq"]) if ext_l_cands else pL
                ext_R = int(ext_r_cands[0]["seq"]) if ext_r_cands else pR
                ext_L_name = format_parent_with_ties(ext_l_cands, ext_L)
                ext_R_name = format_parent_with_ties(ext_r_cands, ext_R)

                recombs_out.append(
                    RecombinantCall(
                        seq_idx=r_idx,
                        seq_name=sname(r_idx),
                        p_switch=round(float(r.get("switch", 0.0)), 4),
                        is_recombinant=True,
                        parent_left=pL,
                        parent_left_name=sname(pL),
                        parent_right=pR,
                        parent_right_name=sname(pR),
                        ext_parent_left=ext_L,
                        ext_parent_left_name=ext_L_name,
                        ext_parent_right=ext_R,
                        ext_parent_right_name=ext_R_name,
                        left_sim=round(float(r.get("left_sim", 0.0)), 4),
                        right_sim=round(float(r.get("right_sim", 0.0)), 4),
                        left_candidates=[
                            ParentalCandidate(
                                seq_idx=int(c["seq"]),
                                seq_name=sname(int(c["seq"])),
                                similarity=round(float(c["sim"]), 4),
                                is_external_to_event=(int(c["seq"]) not in carrier_set),
                            )
                            for c in l_cands
                        ],
                        right_candidates=[
                            ParentalCandidate(
                                seq_idx=int(c["seq"]),
                                seq_name=sname(int(c["seq"])),
                                similarity=round(float(c["sim"]), 4),
                                is_external_to_event=(int(c["seq"]) not in carrier_set),
                            )
                            for c in r_cands
                        ],
                    )
                )

            # Signal attribution & diagnostic notes (PROJECT.md interface contract)
            src = p_meta.get("source", config.scoring_mode)
            z_val = float(p_meta["z_kl"])
            tf_log = float(p_meta["tf_logit"])

            if config.scoring_mode == "standalone_tf" or src == "transformer":
                if z_val >= config.z_thresh:
                    sig_attr = "transformer_corroborated"
                    attr_note = f"Transformer peak corroborated by flanking KL (logit={tf_log:.2f}, z={z_val:.2f})"
                else:
                    sig_attr = "standalone_tf"
                    attr_note = f"Transformer prediction without flanking KL peak (logit={tf_log:.2f})"
            elif config.scoring_mode == "flank_kl" or src in ("flank_kl", "micro_flank", "hybrid"):
                if tf_log >= config.tf_logit_thresh and M_tx >= 314:
                    if src == "hybrid":
                        sig_attr = "hybrid_flank"
                        attr_note = f"Hybrid prediction combining flanking KL (z={z_val:.2f}) and Transformer (logit={tf_log:.2f})"
                    else:
                        sig_attr = "transformer_corroborated"
                        attr_note = f"Flanking KL peak corroborated by Transformer (z={z_val:.2f}, logit={tf_log:.2f})"
                elif M_tx < 314:
                    sig_attr = "flank_kl_heuristic_only"
                    attr_note = f"Prediction relies solely on local flanking KL heuristic (alignment M_tx={M_tx} < 314 window)"
                elif tf_log < 0.0:
                    sig_attr = "flank_kl_heuristic_only"
                    attr_note = f"Prediction relies solely on local flanking KL heuristic (transformer logit={tf_log:.2f} < 0.0)"
                else:
                    sig_attr = "flank_kl_heuristic_only"
                    attr_note = f"Flanking KL peak (z={z_val:.2f}) with subthreshold transformer response (logit={tf_log:.2f})"
            else:
                sig_attr = "hybrid_flank"
                attr_note = f"Hybrid flank detection (z={z_val:.2f}, logit={tf_log:.2f})"

            consolidated_events.append(
                ConsolidatedEvent(
                    event_id=int(ev["event_id"]),
                    most_plausible_bp_orig=int(ev["most_plausible_breakpoint_orig"]),
                    span_orig_start=int(ev["breakpoint_span_orig"][0]),
                    span_orig_end=int(ev["breakpoint_span_orig"][1]),
                    n_peaks_consolidated=int(ev["n_peaks_consolidated"]),
                    constituent_bps_orig=[int(x) for x in ev.get("constituent_breakpoints_orig", [])],
                    score_z=round(float(p_meta["z_kl"]), 4),
                    tf_logit=round(float(p_meta["tf_logit"]), 4),
                    tf_prob=round(float(p_meta["tf_prob"]), 4),
                    hybrid_score=round(float(p_meta["hybrid"]), 4),
                    boundary_tx=b_tx,
                    signal_attribution=sig_attr,
                    attribution_note=attr_note,
                    n_switch_recombinants=len(recombs_out),
                    recombinant_sequences=[r.seq_idx for r in recombs_out],
                    recombinants=recombs_out,
                )
            )

        assign_ancestral_event_ids(consolidated_events)

        raw_peaks_out: list[RawPeakCall] = []
        for rp in raw_for_cons:
            b_tx = int(rp["boundary_tx"])
            bp_orig = int(rp["breakpoint_orig"])
            p_meta = raw_peak_meta[b_tx]
            raw_peaks_out.append(
                RawPeakCall(
                    boundary_tx=b_tx,
                    breakpoint_orig=bp_orig,
                    score_z=round(float(p_meta["z_kl"]), 4),
                    tf_logit=round(float(p_meta["tf_logit"]), 4),
                    tf_prob=round(float(p_meta["tf_prob"]), 4),
                    n_switch=int(rp["n_switch"]),
                    source=str(p_meta["source"]),
                    consolidated_event_id=orig_to_event_id.get(bp_orig),
                )
            )

        # Build binned continuous genome profile in original coordinates for plotting
        track_profile = self._build_track_profile(
            maps=maps,
            kept=kept,
            L=L,
            z_kl=z_kl,
            tf_logits_b=tf_logits_b,
            tf_prob_b=tf_prob_b,
            hybrid_profile=hybrid_profile,
            threshold_z=config.z_thresh,
            threshold_tf_logit=config.tf_logit_thresh,
        )
        return consolidated_events, raw_peaks_out, track_profile

    @staticmethod
    def _build_track_profile(
        maps: dict[str, Any],
        kept: list[int],
        L: int,
        z_kl: np.ndarray,
        tf_logits_b: np.ndarray,
        tf_prob_b: np.ndarray,
        hybrid_profile: np.ndarray,
        threshold_z: float,
        threshold_tf_logit: float,
        max_bins: int = 400,
    ) -> GenomeTrackProfile:
        """Downsample boundary score arrays onto original bp bins taking local maxima per bin."""
        tx2o = np.asarray(maps["tx_to_orig"], dtype=np.int64)
        M_tx = len(tx2o)
        b_orig = np.empty(M_tx + 1, dtype=np.int64)
        b_orig[:M_tx] = tx2o
        b_orig[M_tx] = L
        kept_arr = np.asarray(kept, dtype=np.int64)

        n_bins = min(max_bins, max(120, L // 10))
        edges = np.linspace(0, L, n_bins + 1)
        positions: list[int] = []
        z_bins: list[float] = []
        tf_prob_bins: list[float] = []
        tf_logit_bins: list[float] = []
        hyb_bins: list[float] = []
        dens_bins: list[float] = []

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mid = int(round(0.5 * (lo + hi)))
            positions.append(mid)
            bin_span = max(hi - lo, 1.0)
            info_cnt = int(np.sum((kept_arr >= lo) & (kept_arr < hi))) if len(kept_arr) else 0
            dens_bins.append(round(min(1.0, info_cnt / bin_span), 3))

            mask = (b_orig >= lo) & (b_orig < hi) if i < n_bins - 1 else (b_orig >= lo) & (b_orig <= hi)
            if not np.any(mask):
                idx = int(np.argmin(np.abs(b_orig - mid)))
                z_bins.append(round(float(z_kl[idx]), 3))
                tf_prob_bins.append(round(float(tf_prob_b[idx]), 4))
                tf_logit_bins.append(round(float(tf_logits_b[idx]), 3))
                hyb_bins.append(round(float(hybrid_profile[idx]), 3))
            else:
                z_bins.append(round(float(np.max(z_kl[mask])), 3))
                tf_prob_bins.append(round(float(np.max(tf_prob_b[mask])), 4))
                tf_logit_bins.append(round(float(np.max(tf_logits_b[mask])), 3))
                hyb_bins.append(round(float(np.max(hybrid_profile[mask])), 3))

        return GenomeTrackProfile(
            positions_orig_bp=positions,
            flank_kl_z=z_bins,
            tf_prob=tf_prob_bins,
            tf_logit=tf_logit_bins,
            hybrid_score=hyb_bins,
            informative_density=dens_bins,
            threshold_z=threshold_z,
            threshold_tf_logit=threshold_tf_logit,
        )
