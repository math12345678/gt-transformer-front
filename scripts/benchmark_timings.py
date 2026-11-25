"""
Phase 4 Timing Benchmark Script (`scripts/benchmark_timings.py`)

Runs:
  - 01_main_8seq_1500bp.fasta (8 x 1,500 bp)
  - 03_null_6seq_1200bp.fasta (6 x 1,200 bp)
  - 07_scale_60seq_3000bp.fasta (60 x 3,000 bp)
  - d005_L3536_s96_r001.fas (96 x 3,536 bp preset)
3 times each (after model warmup) and writes `timings.csv` with columns:
  file,n_seqs,length,run,parse_s,kl_s,transformer_s,render_s,total_s,median_total_s
"""
from __future__ import annotations

import csv
import shutil
import statistics
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = WEB_DIR / "tests" / "fixtures"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app  # noqa: E402
from schemas import InferenceConfig  # noqa: E402


def main() -> None:
    suite_dir = FIXTURES_DIR
    targets = [
        ("01_main_8seq_1500bp.fasta", suite_dir / "01_main_8seq_1500bp.fasta", None),
        ("03_null_6seq_1200bp.fasta", suite_dir / "03_null_6seq_1200bp.fasta", None),
        ("07_SCALE_60seq_3000bp.fasta", suite_dir / "07_SCALE_60seq_3000bp.fasta", None),
        ("d005_L3536_s96_r001.fas", None, "d005_L3536_s96_r001"),
    ]

    cfg = InferenceConfig(checkpoint="ckpt_N10000", scoring_mode="hybrid_flank")
    all_rows: list[dict] = []

    for label, fpath, preset_id in targets:
        run_records = []
        raw_text = fpath.read_text(encoding="utf-8") if fpath is not None else None
        for r_idx in range(1, 4):
            if preset_id is not None:
                res = web_app.engine.run_inference(cfg, preset_id=preset_id)
            else:
                res = web_app.engine.run_inference(
                    cfg,
                    fasta_text=raw_text,
                    fasta_filename=label,
                    strict_phase2=True,
                )
            tim = res.timings
            sec = tim.stage_seconds
            run_records.append({
                "file": label,
                "n_seqs": res.alignment.n_sequences,
                "length": res.alignment.genome_length_bp,
                "run": r_idx,
                "parse_s": round(sec.get("parse_s", 0.0), 4),
                "kl_s": round(sec.get("kl_s", 0.0), 4),
                "transformer_s": round(sec.get("transformer_s", 0.0), 4),
                "render_s": round(sec.get("render_s", 0.0), 4),
                "total_s": round(tim.total_ms / 1000.0, 4),
                "summary_line": tim.stage_summary_line,
            })

        med_total = round(statistics.median(r["total_s"] for r in run_records), 4)
        for r in run_records:
            r["median_total_s"] = med_total
            all_rows.append(r)
        print(
            f"{label}: S={run_records[0]['n_seqs']}, L={run_records[0]['length']} bp | "
            f"median_total={med_total:.3f} s | {run_records[-1]['summary_line']}"
        )

    fieldnames = [
        "file",
        "n_seqs",
        "length",
        "run",
        "parse_s",
        "kl_s",
        "transformer_s",
        "render_s",
        "total_s",
        "median_total_s",
    ]
    out_paths = [WEB_DIR / "timings.csv"]
    for out_csv in out_paths:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"Wrote timings.csv ({len(all_rows)} rows) to {out_paths[0]}")


if __name__ == "__main__":
    main()
