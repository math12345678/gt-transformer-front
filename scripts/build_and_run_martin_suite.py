"""
Build the martin-web-test/ benchmark & validation suite (01 through 09h)
and execute the full end-to-end evaluation through the web job API to produce
RESULTS_v2.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

WEB_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = WEB_DIR / "tests" / "fixtures"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from services.fasta_utils import parse_raw_fasta_records  # noqa: E402


def mutate_seq(seq: str, rate: float, rng: np.random.Generator) -> str:
    arr = list(seq)
    bases = ["A", "C", "G", "T"]
    n_mut = int(round(len(arr) * rate))
    positions = rng.choice(len(arr), size=n_mut, replace=False)
    for pos in positions:
        cur = arr[pos]
        choices = [b for b in bases if b != cur]
        arr[pos] = choices[int(rng.integers(0, 3))]
    return "".join(arr)


def wrap60(seq: str) -> str:
    return "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60))


def build_martin_suite(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_01_src = FIXTURES_DIR / "01_main_8seq_1500bp.fasta"
    if not main_01_src.exists():
        raise FileNotFoundError(f"Missing {main_01_src}")

    raw_01 = main_01_src.read_text(encoding="utf-8")
    (out_dir / "01_main_8seq_1500bp.fasta").write_text(raw_01, encoding="utf-8")
    (out_dir / "02_CLEAN_two_recombinants.fasta").write_text(raw_01, encoding="utf-8")

    records_01, _, _ = parse_raw_fasta_records(raw_01)
    seqs_01 = {name: seq for name, seq in records_01}
    p1 = seqs_01["P1"]
    p2 = seqs_01["P2"]
    p3 = seqs_01["P3"]
    p4 = seqs_01["P4"]
    p5 = seqs_01["P5"]
    p6 = seqs_01["P6"]

    # 03: Null 6 seqs x 1200 bp (pure clonal tree, 0 recombinations)
    rng03 = np.random.default_rng(202603)
    anc03 = "".join(rng03.choice(["A", "C", "G", "T"], size=1200))
    bA = mutate_seq(anc03, 0.05, rng03)
    bB = mutate_seq(anc03, 0.05, rng03)
    s03 = [
        ("Clonal_A1", mutate_seq(bA, 0.02, rng03)),
        ("Clonal_A2", mutate_seq(bA, 0.02, rng03)),
        ("Clonal_A3", mutate_seq(bA, 0.02, rng03)),
        ("Clonal_B1", mutate_seq(bB, 0.02, rng03)),
        ("Clonal_B2", mutate_seq(bB, 0.02, rng03)),
        ("Clonal_B3", mutate_seq(bB, 0.02, rng03)),
    ]
    txt03 = "\n".join(f">{k}\n{wrap60(v)}" for k, v in s03) + "\n"
    (out_dir / "03_CLEAN_no_recombination_null.fasta").write_text(txt03, encoding="utf-8")
    (out_dir / "03_null_6seq_1200bp.fasta").write_text(txt03, encoding="utf-8")

    # 04: Short tract 60 bp (701..760 in R1: P1 -> P2 -> P1)
    rng04 = np.random.default_rng(202604)
    r1_04 = mutate_seq(p1[:700] + p2[700:760] + p1[760:], 0.002, rng04)
    s04 = [
        ("P1_parent_alpha", p1),
        ("P2_parent_beta", p2),
        ("P3_parent_gamma", p3),
        ("P4_parent_delta", p4),
        ("R1_short_701_760", r1_04),
        ("P5_sister_of_P1", p5),
        ("P6_sister_of_P4", p6),
        ("P7_sister_of_P2", mutate_seq(p2, 0.01, rng04)),
    ]
    txt04 = "\n".join(f">{k}\n{wrap60(v)}" for k, v in s04) + "\n"
    (out_dir / "04_EDGE_short_tract_60bp.fasta").write_text(txt04, encoding="utf-8")

    # 05: Boundary near start (bp=26 and bp=401 in R1: P2[0:25] + P1[25:400] + P2[400:])
    rng05 = np.random.default_rng(202605)
    r1_05 = mutate_seq(p2[:25] + p1[25:400] + p2[400:], 0.002, rng05)
    s05 = [
        ("P1_parent_alpha", p1),
        ("P2_parent_beta", p2),
        ("P3_parent_gamma", p3),
        ("P4_parent_delta", p4),
        ("R1_edge_bp26_401", r1_05),
        ("P5_sister_of_P1", p5),
        ("P6_sister_of_P4", p6),
        ("P7_sister_of_P2", mutate_seq(p2, 0.01, rng05)),
    ]
    txt05 = "\n".join(f">{k}\n{wrap60(v)}" for k, v in s05) + "\n"
    (out_dir / "05_EDGE_boundary_near_start.fasta").write_text(txt05, encoding="utf-8")

    # 06: Overlapping events (bp=301, 601, 901)
    rng06 = np.random.default_rng(202606)
    r1_06 = mutate_seq(p1[:300] + p2[300:600] + p3[600:900] + p1[900:], 0.002, rng06)
    s06 = [
        ("P1_parent_alpha", p1),
        ("P2_parent_beta", p2),
        ("P3_parent_gamma", p3),
        ("P4_parent_delta", p4),
        ("R1_overlap_301_601_901", r1_06),
        ("P5_sister_of_P1", p5),
        ("P6_sister_of_P4", p6),
        ("P7_sister_of_P2", mutate_seq(p2, 0.01, rng06)),
    ]
    txt06 = "\n".join(f">{k}\n{wrap60(v)}" for k, v in s06) + "\n"
    (out_dir / "06_EDGE_overlapping_events.fasta").write_text(txt06, encoding="utf-8")

    # 07: Scale 60 seqs x 3000 bp (planted recombinants at 801, 1501, 1801, 2401)
    rng07 = np.random.default_rng(202607)
    anc07 = "".join(rng07.choice(["A", "C", "G", "T"], size=3000))
    clade_centers = [mutate_seq(anc07, 0.08, rng07) for _ in range(6)]
    s07 = []
    for c_idx, center in enumerate(clade_centers):
        for m_idx in range(9):
            s07.append((f"Clade{c_idx+1}_seq{m_idx+1}", mutate_seq(center, 0.012, rng07)))
    # Add 6 recombinant sequences across two mosaic patterns
    pA = s07[0][1]
    pB = s07[9][1]
    pC = s07[18][1]
    pD = s07[27][1]
    for r_i in range(3):
        rec1 = mutate_seq(pA[:800] + pB[800:1500] + pA[1500:], 0.003, rng07)
        s07.append((f"R1_clade1x2_rep{r_i+1}", rec1))
    for r_i in range(3):
        rec2 = mutate_seq(pC[:1800] + pD[1800:2400] + pC[2400:], 0.003, rng07)
        s07.append((f"R2_clade3x4_rep{r_i+1}", rec2))
    txt07 = "\n".join(f">{k}\n{wrap60(v)}" for k, v in s07) + "\n"
    (out_dir / "07_SCALE_60seq_3000bp.fasta").write_text(txt07, encoding="utf-8")
    (out_dir / "07_scale_60seq_3000bp.fasta").write_text(txt07, encoding="utf-8")

    # 08a: lowercase + wrapped at 60 cols
    txt08a = "\n".join(f">{name}\n{wrap60(seq.lower())}" for name, seq in records_01) + "\n"
    (out_dir / "08a_FMT_lowercase_wrapped60.fasta").write_text(txt08a, encoding="utf-8")

    # 08b: CRLF line endings
    txt08b = raw_01.replace("\r\n", "\n").replace("\n", "\r\n")
    (out_dir / "08b_FMT_crlf_line_endings.fasta").write_bytes(txt08b.encode("utf-8"))

    # 08c: spaces and digits inside sequence lines
    lines_08c = []
    for name, seq in records_01:
        lines_08c.append(f">{name}")
        for pos in range(0, len(seq), 60):
            chunk = seq[pos : pos + 60]
            spaced = " ".join(chunk[j : j + 10] for j in range(0, len(chunk), 10))
            lines_08c.append(f"  {pos + 1:4d} {spaced} {min(len(seq), pos + 60)}")
    (out_dir / "08c_FMT_spaces_and_digits_in_seq.fasta").write_text("\n".join(lines_08c) + "\n", encoding="utf-8")

    # 08d: special characters & long descriptions in headers
    lines_08d = []
    for idx, (name, seq) in enumerate(records_01):
        full_hdr = f"{name} |isolate=ZA-{idx+101}/2026|host=Homo_sapiens|country=South_Africa:Cape_Town [complete_cds]"
        lines_08d.append(f">{full_hdr}\n{wrap60(seq)}")
    (out_dir / "08d_FMT_special_chars_in_headers.fasta").write_text("\n".join(lines_08d) + "\n", encoding="utf-8")

    # 08e: IUPAC ambiguity characters (N, R, Y, W, S, K, M) and gaps (-)
    lines_08e = []
    iupac_codes = ["N", "R", "Y", "W", "S", "K", "M", "-"]
    for idx, (name, seq) in enumerate(records_01):
        arr = list(seq)
        arr[10 + idx] = iupac_codes[idx % len(iupac_codes)]
        arr[20 + idx] = "-"
        lines_08e.append(f">{name}\n{wrap60(''.join(arr))}")
    (out_dir / "08e_FMT_iupac_ambiguity_and_gaps.fasta").write_text("\n".join(lines_08e) + "\n", encoding="utf-8")

    # 09a: Unequal lengths
    lines_09a = []
    for idx, (name, seq) in enumerate(records_01):
        s_mod = seq[:1450] if idx == 1 else seq
        lines_09a.append(f">{name}\n{wrap60(s_mod)}")
    (out_dir / "09a_BAD_unequal_lengths.fasta").write_text("\n".join(lines_09a) + "\n", encoding="utf-8")

    # 09b: Only 2 sequences
    lines_09b = [f">{name}\n{wrap60(seq)}" for name, seq in records_01[:2]]
    (out_dir / "09b_BAD_only_2_sequences.fasta").write_text("\n".join(lines_09b) + "\n", encoding="utf-8")

    # 09c: Duplicate headers
    lines_09c = []
    for idx, (name, seq) in enumerate(records_01):
        dup_name = records_01[0][0] if idx == 2 else name
        lines_09c.append(f">{dup_name}\n{wrap60(seq)}")
    (out_dir / "09c_BAD_duplicate_headers.fasta").write_text("\n".join(lines_09c) + "\n", encoding="utf-8")

    # 09d: Invalid character 'J' at column 25 of sequence 2
    lines_09d = []
    for idx, (name, seq) in enumerate(records_01):
        if idx == 1:
            seq = seq[:24] + "J" + seq[25:]
        lines_09d.append(f">{name}\n{wrap60(seq)}")
    (out_dir / "09d_BAD_invalid_characters.fasta").write_text("\n".join(lines_09d) + "\n", encoding="utf-8")

    # 09e: Empty file
    (out_dir / "09e_BAD_empty_file.fasta").write_text("", encoding="utf-8")

    # 09f: Not a FASTA (no '>' header)
    (out_dir / "09f_BAD_not_a_fasta.fasta").write_text(
        "This is a plain text document with no FASTA headers.\nACGTACGTACGTACGT\n",
        encoding="utf-8",
    )

    # 09g: Header with empty sequence
    lines_09g = [
        f">{records_01[0][0]}\n{wrap60(records_01[0][1])}",
        ">Empty_Seq_Record\n   \n",
    ] + [f">{name}\n{wrap60(seq)}" for name, seq in records_01[2:]]
    (out_dir / "09g_BAD_header_with_empty_seq.fasta").write_text("\n".join(lines_09g) + "\n", encoding="utf-8")

    # 09h: All gaps
    lines_09h = [f">GapSeq_{i+1}\n{'-' * 500}" for i in range(6)]
    (out_dir / "09h_BAD_all_gaps.fasta").write_text("\n".join(lines_09h) + "\n", encoding="utf-8")

    truth = {
        "01_main_8seq_1500bp.fasta": {
            "breakpoints": [401, 901, 1001, 1351],
            "events": [
                {"carrier": "R1_rec_P1xP2_bp400_900", "start": 401, "end": 901, "parents": ["P1_parent_alpha", "P2_parent_beta"]},
                {"carrier": "R2_rec_P3xP4_bp1000_1350", "start": 1001, "end": 1351, "parents": ["P3_parent_gamma", "P4_parent_delta"]},
            ],
        },
        "03_CLEAN_no_recombination_null.fasta": {"breakpoints": [], "events": []},
        "04_EDGE_short_tract_60bp.fasta": {"breakpoints": [701, 761]},
        "05_EDGE_boundary_near_start.fasta": {"breakpoints": [26, 401]},
        "06_EDGE_overlapping_events.fasta": {"breakpoints": [301, 601, 901]},
        "07_SCALE_60seq_3000bp.fasta": {"breakpoints": [801, 1501, 1801, 2401]},
    }
    (out_dir / "truth_breakpoints.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def submit_and_wait_job(client: TestClient, fasta_path: Path, scoring_mode: str = "hybrid_flank", checkpoint: str = "ckpt_N10000") -> tuple[int, dict]:
    with fasta_path.open("rb") as fh:
        res = client.post(
            "/api/jobs",
            data={
                "checkpoint": checkpoint,
                "scoring_mode": scoring_mode,
            },
            files={"fasta_file": (fasta_path.name, fh, "text/plain")},
        )
    if res.status_code != 200:
        return res.status_code, res.json()
    job_id = res.json()["job_id"]
    for _ in range(120):
        poll = client.get(f"/api/jobs/{job_id}").json()
        if poll["status"] in ("done", "error", "cancelled"):
            return 200, poll
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} timed out")


def main() -> None:
    suite_dir = FIXTURES_DIR
    client = TestClient(web_app.app)

    test_files = [
        "01_main_8seq_1500bp.fasta",
        "02_CLEAN_two_recombinants.fasta",
        "03_CLEAN_no_recombination_null.fasta",
        "04_EDGE_short_tract_60bp.fasta",
        "05_EDGE_boundary_near_start.fasta",
        "06_EDGE_overlapping_events.fasta",
        "07_SCALE_60seq_3000bp.fasta",
        "08a_FMT_lowercase_wrapped60.fasta",
        "08b_FMT_crlf_line_endings.fasta",
        "08c_FMT_spaces_and_digits_in_seq.fasta",
        "08d_FMT_special_chars_in_headers.fasta",
        "08e_FMT_iupac_ambiguity_and_gaps.fasta",
        "09a_BAD_unequal_lengths.fasta",
        "09b_BAD_only_2_sequences.fasta",
        "09c_BAD_duplicate_headers.fasta",
        "09d_BAD_invalid_characters.fasta",
        "09e_BAD_empty_file.fasta",
        "09f_BAD_not_a_fasta.fasta",
        "09g_BAD_header_with_empty_seq.fasta",
        "09h_BAD_all_gaps.fasta",
    ]

    summary_rows = []
    for fname in test_files:
        fpath = suite_dir / fname
        code, body = submit_and_wait_job(client, fpath, scoring_mode="hybrid_flank", checkpoint="ckpt_N10000")
        if code != 200:
            summary_rows.append({
                "file": fname,
                "http": code,
                "status": "REJECTED",
                "detail": body.get("detail", ""),
            })
        elif body.get("status") == "error":
            summary_rows.append({
                "file": fname,
                "http": 422,
                "status": "REJECTED (worker)",
                "detail": body.get("error", ""),
            })
        else:
            res = body["result"]
            evs = res["consolidated_events"]
            bps = [e["most_plausible_bp_orig"] for e in evs]
            recs = [
                f"{e['recombinants'][0]['seq_name']} ({e['recombinants'][0]['ext_parent_left_name']}->{e['recombinants'][0]['ext_parent_right_name']})"
                if e.get("recombinants") else "none"
                for e in evs
            ]
            summary_rows.append({
                "file": fname,
                "http": 200,
                "status": "OK",
                "n_events": len(evs),
                "bps": bps,
                "carriers_parents": recs,
                "timing": res["timings"]["stage_summary_line"],
            })

    print(json.dumps(summary_rows, indent=2))
    (suite_dir / "suite_run_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
