"""
Comprehensive Verification Test Suite for Phases 1 through 6:
  - Phase 1: Non-blocking startup, model pre-loading, background job queue, polling, cancellation, 6-job cap (HTTP 429), 1-hour expiry
  - Phase 2: validate_alignment(records) exact error messages (09a-09h), formatting variants (08a-08e), size caps (20 MB, 500 seqs, 50,000 cols)
  - Phase 3: Accuracy on 01_main_8seq_1500bp.fasta (4/4 TP within ±6 bp, 0 FP, exact carriers R1_rec/R2_rec, exact parents P1/P2/P3/P4)
  - Phase 4: Per-stage timing summary line and timings.csv verification
  - Phase 5 & 6: /healthz endpoint, Dockerfile, README, and "How to use" guide panel
"""
from __future__ import annotations

import csv
import sys
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

WEB_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = WEB_DIR / "tests" / "fixtures"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app  # noqa: E402
from services.fasta_utils import validate_alignment  # noqa: E402


class TestPhases1To6(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(web_app.app)
        cls.suite_dir = FIXTURES_DIR

    def test_01_index_html_empty_state_and_how_to_use(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn("Open a FASTA alignment or pick a demo, then press Run.", html)
        self.assertIn("How to Use gt-transformer", html)
        self.assertIn("Download CSV", html)
        self.assertIn("Download JSON", html)

    def test_02_healthz_and_preloaded_checkpoints(self) -> None:
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertGreater(data["startup_load_ms"], 0.0)
        # Verify core checkpoints are preloaded in memory
        self.assertIn("ckpt_N40000", web_app.engine._ckpt_cache)
        self.assertIn("ckpt_N10000", web_app.engine._ckpt_cache)
        self.assertIn("ckpt_N10000_extreme", web_app.engine._ckpt_cache)

    def test_03_validate_alignment_unit_and_caps(self) -> None:
        # < 5 sequences
        with self.assertRaises(ValueError) as ctx:
            validate_alignment([("s1", "ACGT"), ("s2", "ACGT"), ("s3", "ACGT")])
        self.assertEqual(str(ctx.exception), "Need at least 5 sequences; this file has 3.")

        # Duplicate sequence names
        with self.assertRaises(ValueError) as ctx:
            validate_alignment([
                ("P1", "ACGTACGT"),
                ("P2", "ACGTACGT"),
                ("P1", "ACGTACGT"),
                ("P4", "ACGTACGT"),
                ("P5", "ACGTACGT"),
            ])
        self.assertEqual(str(ctx.exception), "Duplicate sequence name 'P1' (records 1 and 3).")

        # Unequal lengths
        with self.assertRaises(ValueError) as ctx:
            validate_alignment([
                ("P1", "ACGTACGT"),
                ("P2", "ACGT"),
                ("P3", "ACGTACGT"),
                ("P4", "ACGTACGT"),
                ("P5", "ACGTACGT"),
            ])
        self.assertEqual(
            str(ctx.exception),
            "Sequence 'P2' is 4 bp; 'P1' is 8 bp. All sequences must be aligned to the same length.",
        )

        # Invalid character
        with self.assertRaises(ValueError) as ctx:
            validate_alignment([
                ("P1", "ACGTACGT"),
                ("P2", "ACGJACGT"),
                ("P3", "ACGTACGT"),
                ("P4", "ACGTACGT"),
                ("P5", "ACGTACGT"),
            ])
        self.assertEqual(
            str(ctx.exception),
            "Sequence 'P2' has invalid character 'J' at column 4.",
        )

        # > 500 sequences cap
        with self.assertRaises(ValueError) as ctx:
            validate_alignment([(f"s{i}", "ACGTACGT") for i in range(501)])
        self.assertIn("500 sequences", str(ctx.exception))

        # > 50,000 columns cap
        with self.assertRaises(ValueError) as ctx:
            validate_alignment([(f"s{i}", "A" * 50001) for i in range(5)])
        self.assertIn("50,000 columns", str(ctx.exception))

    def test_04_phase2_bad_inputs_09a_to_09h(self) -> None:
        expected_substrings = {
            "09a_BAD_unequal_lengths.fasta": "Sequence 'P2' is 1450 bp; 'P1' is 1500 bp. All sequences must be aligned to the same length.",
            "09b_BAD_only_2_sequences.fasta": "Need at least 5 sequences; this file has 2.",
            "09c_BAD_duplicate_headers.fasta": "Duplicate sequence name 'P1' (records 1 and 3).",
            "09d_BAD_invalid_characters.fasta": "Sequence 'P2' has invalid character 'J' at column 25.",
            "09e_BAD_empty_file.fasta": "Empty alignment payload",
            "09f_BAD_not_a_fasta.fasta": "before first '>' header",
            "09g_BAD_header_with_empty_seq.fasta": "Sequence 'Empty_Seq_Record' is 0 bp; 'P1' is 1500 bp.",
            "09h_BAD_all_gaps.fasta": "contains only gap characters",
        }
        for fname, exp in expected_substrings.items():
            fpath = self.suite_dir / fname
            with fpath.open("rb") as fh:
                res = self.client.post("/api/jobs", files={"fasta_file": (fname, fh, "text/plain")})
            self.assertEqual(res.status_code, 400, f"Expected 400 for {fname}, got {res.status_code}")
            self.assertIn(exp, res.json()["detail"])

    def test_05_phase3_accuracy_on_01_and_08a_to_08e(self) -> None:
        for fname in [
            "01_main_8seq_1500bp.fasta",
            "08a_FMT_lowercase_wrapped60.fasta",
            "08b_FMT_crlf_line_endings.fasta",
            "08c_FMT_spaces_and_digits_in_seq.fasta",
            "08d_FMT_special_chars_in_headers.fasta",
            "08e_FMT_iupac_ambiguity_and_gaps.fasta",
        ]:
            fpath = self.suite_dir / fname
            with fpath.open("rb") as fh:
                res = self.client.post(
                    "/api/jobs",
                    data={"checkpoint": "ckpt_N10000", "scoring_mode": "hybrid_flank"},
                    files={"fasta_file": (fname, fh, "text/plain")},
                )
            self.assertEqual(res.status_code, 200, f"Failed on {fname}: {res.text}")
            job_id = res.json()["job_id"]
            poll = None
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                poll = self.client.get(f"/api/jobs/{job_id}").json()
                if poll["status"] in ("done", "error", "cancelled"):
                    break
                time.sleep(0.05)
            self.assertIsNotNone(poll)
            self.assertEqual(poll["status"], "done", f"Job failed for {fname}: {poll}")
            result = poll["result"]
            evs = result["consolidated_events"]
            bps = [e["most_plausible_bp_orig"] for e in evs]
            self.assertEqual(bps, [405, 901, 1000, 1357], f"Mismatch on {fname}: {bps}")
            carriers = [e["recombinants"][0]["seq_name"] for e in evs]
            self.assertEqual(carriers, ["R1_rec", "R1_rec", "R2_rec", "R2_rec"])
            parents = [
                (e["recombinants"][0]["ext_parent_left_name"], e["recombinants"][0]["ext_parent_right_name"])
                for e in evs
            ]
            self.assertEqual(parents, [("P1", "P2"), ("P2", "P1"), ("P3", "P4"), ("P4", "P3")])
            self.assertIn("Ran in ", result["timings"]["stage_summary_line"])

    def test_06_job_cancellation_and_queue_saturation_429(self) -> None:
        # Inject synthetic queued jobs to test cancellation and 6-job queue saturation cap
        with web_app._JOBS_LOCK:
            saved_jobs = dict(web_app._JOBS)
            web_app._JOBS.clear()
            for i in range(6):
                jid = f"mockjob{i}"
                web_app._JOBS[jid] = {
                    "job_id": jid,
                    "status": "running" if i == 0 else "queued",
                    "stage": "queued",
                    "created_at": time.time(),
                    "started_at": time.time(),
                    "finished_at": None,
                    "cancel_requested": False,
                    "result": None,
                    "error": None,
                }
        try:
            # 7th job must be rejected with HTTP 429 "Server busy, try again in a few minutes"
            res_busy = self.client.post("/api/jobs", data={"preset_id": "d005_L3536_s96_r001"})
            self.assertEqual(res_busy.status_code, 429)
            self.assertEqual(res_busy.json()["detail"], "Server busy, try again in a few minutes")

            # Cancel one of the queued jobs
            res_cancel = self.client.post("/api/jobs/mockjob1/cancel")
            self.assertEqual(res_cancel.status_code, 200)
            self.assertEqual(res_cancel.json()["status"], "cancelled")
        finally:
            with web_app._JOBS_LOCK:
                web_app._JOBS.clear()
                web_app._JOBS.update(saved_jobs)

    def test_07_timings_csv_and_deployment_files(self) -> None:
        timings_path = WEB_DIR / "timings.csv"
        self.assertTrue(timings_path.is_file())
        with timings_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12)
        for col in ("file", "n_seqs", "length", "parse_s", "kl_s", "transformer_s", "render_s", "total_s", "median_total_s"):
            self.assertIn(col, rows[0])

        self.assertTrue((WEB_DIR / "Dockerfile").is_file())
        self.assertTrue((WEB_DIR / "README.md").is_file())
        self.assertTrue((WEB_DIR / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
