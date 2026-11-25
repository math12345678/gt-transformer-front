"""
Regression tests for Security Headers, Private Accounts (local + Google OAuth),
Private Run Isolation & SQLite Job Fallback, Admin Data Collection/Exports,
AI Chat Routes, and New Frontend Controls (Zoom, HTML Report, Run Comparison).

Uses an isolated TemporaryDirectory SQLite database so web/recomb_app.db is untouched.
"""
from __future__ import annotations

import io
import os
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import jax
import numpy as np
from fastapi.testclient import TestClient

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app  # noqa: E402
from gt_core import model as M  # noqa: E402
from routers import auth as auth_router  # noqa: E402
from schemas import ConsolidatedEvent, RecombinantCall  # noqa: E402
from services import chat as chat_mod  # noqa: E402
from services import db as db_mod  # noqa: E402
from services.inference_adapter import GTTransformerInferenceEngine, assign_ancestral_event_ids  # noqa: E402
from services import self_train as self_train_mod  # noqa: E402


def _make_sample_result(run_id: str, dataset_name: str = "sample_align") -> dict:
    return {
        "alignment": {
            "dataset_name": dataset_name,
            "source_type": "upload",
            "n_sequences": 6,
            "genome_length_bp": 1500,
            "n_informative_cols": 120,
            "n_transformed_cols": 120,
            "window_size": 314,
            "gc_content_pct": 48.2,
            "edge_dead_zone_bp": 30,
        },
        "config": {
            "checkpoint": "ckpt_N10000",
            "scoring_mode": "hybrid_flank",
        },
        "timings": {
            "run_id": run_id,
            "total_ms": 42.5,
            "throughput_bp_per_sec": 35294.1,
            "stage_summary_line": "load=2ms | infer=20ms | kl=18ms | post=2ms",
        },
        "consolidated_events": [
            {
                "event_id": 1,
                "most_plausible_bp_orig": 600,
                "span_orig_start": 590,
                "span_orig_end": 610,
                "score_z": 9.4,
                "tf_prob": 0.91,
                "tf_logit": 2.31,
                "signal_attribution": "both_strong",
                "match_status": "UNLABELLED",
                "recombinant_sequences": ["Rec1"],
                "recombinants": [
                    {
                        "seq_idx": 0,
                        "seq_name": "Rec1",
                        "ext_parent_left": "P1",
                        "ext_parent_right": "P2",
                    }
                ],
            }
        ],
        "raw_peaks": [],
        "diagnostics": {"has_ground_truth": False, "tolerance_metrics": []},
        "warnings": [],
    }


class TestAccountsChatAndSelfTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_db_path = db_mod.DB_PATH
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="recomb_web_test_")
        db_mod.DB_PATH = Path(cls._tmpdir.name) / "test_recomb_app.db"
        db_mod.init_db()
        cls.client = TestClient(web_app.app)

    @classmethod
    def tearDownClass(cls) -> None:
        db_mod.DB_PATH = cls._orig_db_path
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self.client.cookies.clear()
        with auth_router._RATE_LOCK:
            auth_router._AUTH_ATTEMPTS.clear()

    def test_01_security_headers_present(self) -> None:
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(res.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")

    def test_02_local_auth_lifecycle_and_rate_limit(self) -> None:
        # Unauthenticated /me returns user=None
        me0 = self.client.get("/api/auth/me")
        self.assertEqual(me0.status_code, 200)
        self.assertIsNone(me0.json()["user"])

        # Register user
        reg = self.client.post("/api/auth/register", json={"username": "student_a", "password": "secretpw"})
        self.assertEqual(reg.status_code, 200)
        self.assertEqual(reg.json()["user"]["username"], "student_a")
        self.assertEqual(reg.json()["user"]["role"], "user")

        # Duplicate username rejected with 400
        dup = self.client.post("/api/auth/register", json={"username": "student_a", "password": "secretpw"})
        self.assertEqual(dup.status_code, 400)

        # Session cookie active
        me1 = self.client.get("/api/auth/me")
        self.assertEqual(me1.json()["user"]["username"], "student_a")

        # Logout clears session
        out = self.client.post("/api/auth/logout")
        self.assertEqual(out.status_code, 200)
        self.assertIsNone(self.client.get("/api/auth/me").json()["user"])

        # Bad password fails with 401, good password logs in
        bad = self.client.post("/api/auth/login", json={"username": "student_a", "password": "wrong"})
        self.assertEqual(bad.status_code, 401)
        good = self.client.post("/api/auth/login", json={"username": "student_a", "password": "secretpw"})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(good.json()["user"]["username"], "student_a")

        # Rate limit enforcement after 20 attempts from same client
        with auth_router._RATE_LOCK:
            auth_router._AUTH_ATTEMPTS.clear()
        for _ in range(20):
            r = self.client.post("/api/auth/login", json={"username": "student_a", "password": "wrong"})
            self.assertEqual(r.status_code, 401)
        limited = self.client.post("/api/auth/login", json={"username": "student_a", "password": "secretpw"})
        self.assertEqual(limited.status_code, 429)

    def test_03_google_oauth_state_validation_and_upsert(self) -> None:
        st = self.client.get("/api/auth/google/status")
        self.assertEqual(st.status_code, 200)
        self.assertIn("configured", st.json())

        # Invalid OAuth state returns 400
        cb = self.client.get("/api/auth/google/callback?code=dummy&state=tampered")
        self.assertEqual(cb.status_code, 400)
        self.assertIn("Invalid OAuth state", cb.json()["detail"])

        # Direct upsert creates/updates Google account and session works
        u1 = db_mod.upsert_google_user("bio.student@uct.ac.za", "sub-uct-99", "Bio Student")
        self.assertEqual(u1["auth_provider"], "google")
        self.assertEqual(u1["email"], "bio.student@uct.ac.za")
        u2 = db_mod.upsert_google_user("bio.student@uct.ac.za", "sub-uct-99", "Bio Student")
        self.assertEqual(u1["id"], u2["id"])

        tok = db_mod.create_session(u1["id"])
        self.client.cookies.set("recomb_session", tok)
        me = self.client.get("/api/auth/me")
        self.assertEqual(me.json()["user"]["auth_provider"], "google")

    def test_04_private_run_isolation_delete_and_job_sqlite_fallback(self) -> None:
        alice = self.client.post("/api/auth/register", json={"username": "alice_priv", "password": "pw11"}).json()["user"]
        self.client.cookies.clear()
        bob = self.client.post("/api/auth/register", json={"username": "bob_priv", "password": "pw22"}).json()["user"]

        fasta_txt = ">Rec1\nACGTACGT\n>P1\nACGTACGT\n>P2\nTGCATGCA\n>S4\nACGTACGT\n>S5\nTGCATGCA\n"
        db_mod.save_run("run_alice_1", alice, "upload", fasta_txt, _make_sample_result("run_alice_1", "alice_hiv"))

        # Currently signed in as Bob: Bob cannot see, read, delete, or fetch Alice's run via /api/jobs fallback
        bob_list = self.client.get("/api/runs").json()["runs"]
        self.assertEqual(len(bob_list), 0)
        self.assertEqual(self.client.get("/api/runs/run_alice_1").status_code, 403)
        self.assertEqual(self.client.delete("/api/runs/run_alice_1").status_code, 403)
        self.assertEqual(self.client.get("/api/jobs/run_alice_1").status_code, 403)

        # Unauthenticated job fallback for nonexistent/unowned job returns 404
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/jobs/run_alice_1").status_code, 404)

        # Log in as Alice: she sees her run, job fallback returns done+result, and delete removes it
        self.client.post("/api/auth/login", json={"username": "alice_priv", "password": "pw11"})
        alice_list = self.client.get("/api/runs").json()["runs"]
        self.assertEqual([r["id"] for r in alice_list], ["run_alice_1"])

        job_fb = self.client.get("/api/jobs/run_alice_1")
        self.assertEqual(job_fb.status_code, 200)
        self.assertEqual(job_fb.json()["status"], "done")
        self.assertEqual(job_fb.json()["result"]["alignment"]["dataset_name"], "alice_hiv")

        deleted = self.client.delete("/api/runs/run_alice_1")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/runs/run_alice_1").status_code, 404)

    def test_05_admin_overview_and_exports(self) -> None:
        # Regular user denied admin endpoints
        user = self.client.post("/api/auth/register", json={"username": "regular_u", "password": "pw33"}).json()["user"]
        self.assertEqual(self.client.get("/api/admin/overview").status_code, 403)
        self.assertEqual(self.client.get("/api/admin/export.csv").status_code, 403)
        self.assertEqual(self.client.get("/api/admin/export.json").status_code, 403)

        fasta_txt = ">Seq1\nAAAAACCCCC\n>Seq2\nCCCCCAAAAA\n>Seq3\nAAAAACCCCC\n>Seq4\nTTTTTGGGGG\n>Seq5\nGGGGTTTTTT\n"
        db_mod.save_run("run_collect_1", user, "upload", fasta_txt, _make_sample_result("run_collect_1", "user_flu"))
        db_mod.record_retrain_event(
            run_id="run_collect_1",
            dataset_name="user_flu",
            status="updated",
            loss_before=0.51,
            loss_after=0.39,
            steps=3,
            n_positive=2,
            n_negative=8,
            notes="Adaptive update OK",
        )

        # Log in as seeded admin
        self.client.cookies.clear()
        admin_user = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
        admin_pass = os.environ.get("ADMIN_PASSWORD", "admin-recomb-2026").strip() or "admin-recomb-2026"
        adm_login = self.client.post("/api/auth/login", json={"username": admin_user, "password": admin_pass})
        self.assertEqual(adm_login.status_code, 200)
        self.assertEqual(adm_login.json()["user"]["role"], "admin")

        ov = self.client.get("/api/admin/overview")
        self.assertEqual(ov.status_code, 200)
        ov_data = ov.json()
        self.assertTrue(any(r["id"] == "run_collect_1" for r in ov_data["runs"]))
        self.assertGreaterEqual(ov_data["stats"]["retrain_updates"], 1)

        csv_res = self.client.get("/api/admin/export.csv")
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn("run_collect_1", csv_res.text)
        self.assertIn("user_flu", csv_res.text)

        json_res = self.client.get("/api/admin/export.json")
        self.assertEqual(json_res.status_code, 200)
        self.assertTrue(any(r["id"] == "run_collect_1" for r in json_res.json()["runs"]))

        fasta_res = self.client.get("/api/admin/runs/run_collect_1/fasta")
        self.assertEqual(fasta_res.status_code, 200)
        self.assertIn(">Seq1", fasta_res.text)

    def test_06_chat_status_and_empty_question_rejected(self) -> None:
        st = self.client.get("/api/chat/status")
        self.assertEqual(st.status_code, 200)
        self.assertIn("configured", st.json())
        self.assertIn("provider", st.json())
        self.assertIn("model", st.json())

        empty_q = self.client.post("/api/chat", json={"question": "   "})
        self.assertEqual(empty_q.status_code, 400)

    def test_07_index_html_includes_zoom_html_report_and_compare_controls(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        for token in (
            "Open a FASTA alignment or pick a demo, then press Run.",
            "How to Use gt-transformer",
            "Download CSV",
            "Download JSON",
            "btnZoomRegionPlot",
            "btnResetPlotZoom",
            "btnExportHtmlReport",
            "Download HTML Report (Printable)",
            "btnCompareRuns",
            "compareModal",
            "btnClearAiChat",
            "btnExportCompleteZip",
            "/api/admin/export.zip",
            "btnInspectRegion",
            "regionModal",
            "batchModal",
            "btnAdminResetAdaptive",
            "tab-simplot",
            "simplotCanvas",
            "btnToggleBaseColors",
            "eventSearchInput",
            "optMirrorTta",
        ):
            self.assertIn(token, html)

    def test_08_alignment_window_sqlite_hydration_and_privacy(self) -> None:
        owner = self.client.post("/api/auth/register", json={"username": "hydr_owner", "password": "pw88"}).json()["user"]
        self.client.cookies.clear()
        self.client.post("/api/auth/register", json={"username": "hydr_other", "password": "pw99"})

        fasta_60 = "\n".join([
            ">Rec1",
            ("ACGT" * 7 + "AT" + "TGCA" * 7 + "TG")[:60],
            ">P1_a",
            ("ACGT" * 15)[:60],
            ">P1_b",
            ("ACGT" * 15)[:60],
            ">P2_a",
            ("TGCA" * 15)[:60],
            ">P2_b",
            ("TGCA" * 15)[:60],
            ">Out",
            ("ACGT" * 15)[:60],
        ])
        res_payload = _make_sample_result("run_hydrate_test", "hydrate_ds")
        res_payload["alignment"]["genome_length_bp"] = 60
        db_mod.save_run("job_hydrate_1", owner, "upload", fasta_60, res_payload)

        with web_app.engine._alignment_cache_lock:
            web_app.engine._alignment_cache.pop("run_hydrate_test", None)

        # Signed in as another user -> 403
        self.assertEqual(self.client.get("/api/alignment/window?run_id=run_hydrate_test&start=0&width=40").status_code, 403)

        # Unauthenticated -> 403
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/alignment/window?run_id=run_hydrate_test&start=0&width=40").status_code, 403)

        # Owner gets 200 hydrated from SQLite
        self.client.post("/api/auth/login", json={"username": "hydr_owner", "password": "pw88"})
        ok = self.client.get("/api/alignment/window?run_id=run_hydrate_test&start=0&width=40")
        self.assertEqual(ok.status_code, 200)
        data = ok.json()
        self.assertEqual(data["genome_length_bp"], 60)
        self.assertEqual(len(data["sequences"]), 6)
        self.assertEqual(len(data["sequences"][0]["slice"]), 40)

    def test_09_self_train_atomic_update_and_skip_short(self) -> None:
        orig_dir = self_train_mod.ADAPTIVE_DIR
        orig_ckpt = self_train_mod.ADAPTIVE_CKPT
        with tempfile.TemporaryDirectory(prefix="recomb_selftrain_") as td:
            self_train_mod.ADAPTIVE_DIR = Path(td) / "ckpt_user_adaptive"
            self_train_mod.ADAPTIVE_CKPT = self_train_mod.ADAPTIVE_DIR / "params.pkl"
            try:
                # 1. Short alignment (10 <= M_tx < 314) passes validation and skips cleanly
                s_a = "ACGT" * 5
                s_b = "TGCA" * 5
                short_fasta = f">S1\n{s_a}\n>S2\n{s_b}\n>S3\n{s_a}\n>S4\n{s_b}\n>S5\n{s_a}\n"
                self_train_mod.retrain_on_user_input(
                    web_app.engine,
                    "run_st_short",
                    short_fasta,
                    _make_sample_result("run_st_short", "short_ds"),
                )
                ov_short = [r for r in db_mod.admin_overview()["retrain_log"] if r["run_id"] == "run_st_short"]
                self.assertEqual(len(ov_short), 1)
                self.assertEqual(ov_short[0]["status"], "skipped")

                # 2. Informative alignment (>= 314 cols) updates adaptive checkpoint atomically with metadata
                p1 = ("AC" * 250)  # 500 bp
                p2 = ("GT" * 250)  # 500 bp
                rec = p1[:250] + p2[250:]
                long_fasta = "\n".join([
                    ">Rec1", rec,
                    ">P1_a", p1,
                    ">P1_b", p1,
                    ">P2_a", p2,
                    ">P2_b", p2,
                    ">P3", p1,
                ])
                long_res = _make_sample_result("run_st_long", "long_ds")
                long_res["consolidated_events"][0]["boundary_tx"] = 250
                long_res["consolidated_events"][0]["score_z"] = 9.5
                long_res["consolidated_events"][0]["tf_prob"] = 0.90
                self_train_mod.retrain_on_user_input(web_app.engine, "run_st_long", long_fasta, long_res)

                self.assertTrue(self_train_mod.ADAPTIVE_CKPT.exists())
                with open(self_train_mod.ADAPTIVE_CKPT, "rb") as fh:
                    ckpt_payload = pickle.load(fh)
                self.assertEqual(ckpt_payload["meta"]["updates"], 1)
                self.assertEqual(ckpt_payload["meta"]["last_run_id"], "run_st_long")

                ov_logs = [r for r in db_mod.admin_overview()["retrain_log"] if r["run_id"] == "run_st_long"]
                self.assertEqual(len(ov_logs), 1)
                self.assertEqual(ov_logs[0]["status"], "updated")
                self.assertLessEqual(ov_logs[0]["loss_after"], ov_logs[0]["loss_before"] * 1.05)
            finally:
                self_train_mod.ADAPTIVE_DIR = orig_dir
                self_train_mod.ADAPTIVE_CKPT = orig_ckpt
                web_app.engine._ckpt_cache.pop("ckpt_user_adaptive", None)

    def test_10_admin_zip_export_and_delete_run(self) -> None:
        user = self.client.post("/api/auth/register", json={"username": "zip_user", "password": "pwzip"}).json()["user"]
        fasta_txt = ">SeqA\nACGTACGTACGT\n>SeqB\nTGCATGCATGCA\n>SeqC\nACGTACGTACGT\n>SeqD\nTGCATGCATGCA\n>SeqE\nACGTACGTACGT\n"
        db_mod.save_run("run_zip_1", user, "upload", fasta_txt, _make_sample_result("run_zip_1", "zip_ds"))

        # Non-admin denied
        self.assertEqual(self.client.get("/api/admin/export.zip").status_code, 403)
        self.assertEqual(self.client.delete("/api/admin/runs/run_zip_1").status_code, 403)

        # Admin exports ZIP & deletes run
        self.client.cookies.clear()
        admin_user = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
        admin_pass = os.environ.get("ADMIN_PASSWORD", "admin-recomb-2026").strip() or "admin-recomb-2026"
        self.assertEqual(
            self.client.post("/api/auth/login", json={"username": admin_user, "password": admin_pass}).status_code,
            200,
        )

        zip_res = self.client.get("/api/admin/export.zip")
        self.assertEqual(zip_res.status_code, 200)
        self.assertEqual(zip_res.headers.get("content-type"), "application/zip")
        with zipfile.ZipFile(io.BytesIO(zip_res.content), "r") as zf:
            names = set(zf.namelist())
            self.assertIn("summary.csv", names)
            self.assertIn("overview.json", names)
            fasta_entries = [n for n in names if n.startswith("fastas/") and n.endswith(".fasta")]
            self.assertTrue(any("run_zip_1" in n for n in fasta_entries))
            self.assertIn("run_zip_1", zf.read("summary.csv").decode("utf-8"))

        del_res = self.client.delete("/api/admin/runs/run_zip_1")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "deleted")
        self.assertEqual(self.client.get("/api/admin/runs/run_zip_1/result").status_code, 404)

    def test_11_ancestral_event_grouping_python_parity(self) -> None:
        def _mk(eid: int, bp: int, carrier: str, p_l: str, p_r: str) -> ConsolidatedEvent:
            return ConsolidatedEvent(
                event_id=eid,
                most_plausible_bp_orig=bp,
                span_orig_start=max(1, bp - 10),
                span_orig_end=bp + 10,
                n_peaks_consolidated=1,
                score_z=8.5,
                tf_logit=2.0,
                tf_prob=0.88,
                hybrid_score=10.5,
                boundary_tx=bp,
                n_switch_recombinants=1,
                recombinant_sequences=[0],
                recombinants=[
                    RecombinantCall(
                        seq_idx=0,
                        seq_name=carrier,
                        p_switch=0.9,
                        parent_left=1,
                        parent_left_name=p_l,
                        parent_right=2,
                        parent_right_name=p_r,
                        ext_parent_left=1,
                        ext_parent_left_name=p_l,
                        ext_parent_right=2,
                        ext_parent_right_name=p_r,
                        left_sim=0.95,
                        right_sim=0.94,
                    )
                ],
            )

        events = [
            _mk(1, 500, "RecA", "P1", "P2"),
            _mk(2, 550, "RecB", "P1 or P3", "P2"),
            _mk(3, 520, "RecC", "P4", "P5"),
            _mk(4, 1200, "RecD", "P1", "P2"),
        ]
        assign_ancestral_event_ids(events)
        self.assertEqual(events[0].ancestral_event_id, 1)
        self.assertEqual(events[1].ancestral_event_id, 1)
        self.assertEqual(events[2].ancestral_event_id, 2)
        self.assertEqual(events[3].ancestral_event_id, 3)

    def test_12_region_stats_pairwise_identity_and_privacy(self) -> None:
        owner = self.client.post("/api/auth/register", json={"username": "reg_owner", "password": "pwreg1"}).json()["user"]
        self.client.cookies.clear()
        self.client.post("/api/auth/register", json={"username": "reg_other", "password": "pwreg2"})

        p1 = "AC" * 60
        p2 = "GT" * 60
        rec = p1[:60] + p2[60:]
        fasta_120 = "\n".join([
            ">Rec1", rec,
            ">P1_a", p1,
            ">P1_b", p1,
            ">P2_a", p2,
            ">P2_b", p2,
            ">Out", p1,
        ])
        payload = _make_sample_result("run_reg_stats", "reg_ds")
        payload["alignment"]["genome_length_bp"] = 120
        db_mod.save_run("job_reg_1", owner, "upload", fasta_120, payload)

        with web_app.engine._alignment_cache_lock:
            web_app.engine._alignment_cache.pop("run_reg_stats", None)

        url = "/api/alignment/region-stats?run_id=run_reg_stats&start_bp=61&end_bp=120&ref_seq_idx=0"
        self.assertEqual(self.client.get(url).status_code, 403)
        self.client.cookies.clear()
        self.assertEqual(self.client.get(url).status_code, 403)

        self.client.post("/api/auth/login", json={"username": "reg_owner", "password": "pwreg1"})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        stats = res.json()
        self.assertEqual(stats["length_bp"], 60)
        by_name = {r["seq_name"]: r for r in stats["pairwise"]}
        self.assertEqual(by_name["P2_a"]["inside_pct"], 100.0)
        self.assertEqual(by_name["P2_a"]["outside_pct"], 0.0)
        self.assertEqual(by_name["P2_a"]["delta_pct"], 100.0)
        self.assertEqual(by_name["P1_a"]["inside_pct"], 0.0)
        self.assertEqual(by_name["P1_a"]["outside_pct"], 100.0)
        self.assertEqual(by_name["P1_a"]["delta_pct"], -100.0)

    def test_13_admin_adaptive_status_and_reset(self) -> None:
        orig_dir = self_train_mod.ADAPTIVE_DIR
        orig_ckpt = self_train_mod.ADAPTIVE_CKPT
        with tempfile.TemporaryDirectory(prefix="recomb_adp_reset_") as td:
            self_train_mod.ADAPTIVE_DIR = Path(td) / "ckpt_user_adaptive"
            self_train_mod.ADAPTIVE_CKPT = self_train_mod.ADAPTIVE_DIR / "params.pkl"
            try:
                self.client.post("/api/auth/register", json={"username": "nonadmin_adp", "password": "pw44"})
                self.assertEqual(self.client.get("/api/admin/adaptive/status").status_code, 403)
                self.assertEqual(self.client.post("/api/admin/adaptive/reset").status_code, 403)

                self.client.cookies.clear()
                admin_user = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
                admin_pass = os.environ.get("ADMIN_PASSWORD", "admin-recomb-2026").strip() or "admin-recomb-2026"
                self.assertEqual(
                    self.client.post("/api/auth/login", json={"username": admin_user, "password": admin_pass}).status_code,
                    200,
                )

                st0 = self.client.get("/api/admin/adaptive/status")
                self.assertEqual(st0.status_code, 200)
                self.assertFalse(st0.json()["exists"])

                rst = self.client.post("/api/admin/adaptive/reset")
                self.assertEqual(rst.status_code, 200)
                rdata = rst.json()
                self.assertTrue(rdata["exists"])
                self.assertEqual(rdata["updates"], 0)
                self.assertTrue(self_train_mod.ADAPTIVE_CKPT.exists())

                reset_logs = [r for r in db_mod.admin_overview()["retrain_log"] if r["status"] == "reset"]
                self.assertGreaterEqual(len(reset_logs), 1)
            finally:
                self_train_mod.ADAPTIVE_DIR = orig_dir
                self_train_mod.ADAPTIVE_CKPT = orig_ckpt
                web_app.engine._ckpt_cache.pop("ckpt_user_adaptive", None)

    def test_14_simplot_sliding_window_and_privacy(self) -> None:
        owner = self.client.post("/api/auth/register", json={"username": "sim_owner", "password": "simpw1"}).json()["user"]
        self.client.cookies.clear()
        self.client.post("/api/auth/register", json={"username": "sim_other", "password": "simpw2"})

        left_half = "AC" * 50   # 100 bp
        right_half = "GT" * 50  # 100 bp
        p1_seq = "AC" * 100     # 200 bp
        p2_seq = "GT" * 100     # 200 bp
        rec_seq = left_half + right_half
        fasta_txt = "\n".join([
            ">Rec1", rec_seq,
            ">P1_a", p1_seq,
            ">P1_b", p1_seq,
            ">P2_a", p2_seq,
            ">P2_b", p2_seq,
            ">Out", p1_seq,
        ])
        run_id = "run_simplot_1"
        db_mod.save_run(run_id, owner, "upload", fasta_txt, _make_sample_result(run_id, "simplot_ds"))
        web_app.engine._alignment_cache.pop(run_id, None)

        url = f"/api/alignment/simplot?run_id={run_id}&query_seq_idx=0&window_bp=40&step_bp=20"
        self.assertEqual(self.client.get(url).status_code, 403)

        self.client.cookies.clear()
        self.assertEqual(self.client.get(url).status_code, 403)

        self.assertEqual(
            self.client.post("/api/auth/login", json={"username": "sim_owner", "password": "simpw1"}).status_code,
            200,
        )
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["run_id"], run_id)
        self.assertEqual(data["genome_length_bp"], 200)
        self.assertEqual(data["query_seq_name"], "Rec1")
        pos = data["positions_bp"]
        self.assertGreaterEqual(len(pos), 5)
        by_name = {s["seq_name"]: s["identity_pct"] for s in data["series"]}
        self.assertIn("P1_a", by_name)
        self.assertIn("P2_a", by_name)

        left_idxs = [i for i, bp in enumerate(pos) if bp <= 80]
        right_idxs = [i for i, bp in enumerate(pos) if bp >= 120]
        self.assertTrue(left_idxs)
        self.assertTrue(right_idxs)
        for i in left_idxs:
            self.assertAlmostEqual(by_name["P1_a"][i], 100.0, places=3)
            self.assertAlmostEqual(by_name["P2_a"][i], 0.0, places=3)
        for i in right_idxs:
            self.assertAlmostEqual(by_name["P1_a"][i], 0.0, places=3)
            self.assertAlmostEqual(by_name["P2_a"][i], 100.0, places=3)

    def test_15_python_bitwise_window_features_exact(self) -> None:
        rng = np.random.default_rng(2026)
        w = 32
        s = 15
        offs = list(range(-(w // 2), w // 2))
        for m_tx in (24, 48):
            haplo = rng.integers(0, 2, size=(m_tx, s), dtype=np.uint8)
            got = GTTransformerInferenceEngine._build_circular_windows_freq(haplo, w)
            ref = np.zeros((m_tx, w, 4), dtype=np.float32)
            for i in range(m_tx):
                for pos, d in enumerate(offs):
                    j = i + d
                    if m_tx < w:
                        if j < 0 or j >= m_tx:
                            continue
                    else:
                        j %= m_tx
                    code = 2 * haplo[i] + haplo[j]
                    for g in range(4):
                        ref[i, pos, g] = float(np.sum(code == g)) / float(s)
            self.assertEqual(got.dtype, np.float32)
            self.assertEqual(got.shape, (m_tx, w, 4))
            np.testing.assert_allclose(got, ref, atol=1e-7, rtol=0.0)

    def test_16_v2_architecture_invariants_and_forward(self) -> None:
        # Row 0: 3-gamete complete linkage (p11=0 -> fg=0, |D'|=1.0); Row 1: 4-gamete recombinant (fg=1, |D'|<1.0)
        x_pair = np.zeros((1, 2, 4), dtype=np.float32)
        x_pair[0, 0] = [0.20, 0.40, 0.40, 0.00]
        x_pair[0, 1] = [0.25, 0.25, 0.25, 0.25]
        feats = np.asarray(M.enrich_gamete_features(x_pair))
        self.assertEqual(feats.shape, (1, 2, 10))
        self.assertAlmostEqual(float(feats[0, 0, 4]), 0.0, places=5)  # fg == 0
        self.assertAlmostEqual(float(feats[0, 0, 8]), 1.0, places=4)  # |D'| == 1
        self.assertAlmostEqual(float(feats[0, 1, 4]), 1.0, places=5)  # fg == 1
        self.assertAlmostEqual(float(feats[0, 1, 7]), 0.0, places=4)  # r2 == 0
        self.assertAlmostEqual(float(feats[0, 1, 8]), 0.0, places=4)  # |D'| == 0

        cfg_compat = M.config_from_dict({"arch_version": 1, "unknown_future_key": 123})
        self.assertEqual(cfg_compat.arch_version, 1)

        rng = np.random.default_rng(7)
        x_rand = rng.random((4, 314, 4), dtype=np.float32)
        x_rand = x_rand / np.maximum(x_rand.sum(axis=-1, keepdims=True), 1e-6)
        for arch, exp_params in ((1, 570881), (2, 671361)):
            cfg = M.Config(arch_version=arch)
            params = M.init_params(jax.random.PRNGKey(0), cfg)
            self.assertEqual(M.n_params(params), exp_params)
            logits = np.asarray(M.forward(params, x_rand, cfg))
            self.assertEqual(logits.shape, (4,))
            self.assertTrue(np.all(np.isfinite(logits)))

    def test_17_mirror_tta_and_checkpoint_ensemble(self) -> None:
        presets_res = self.client.get("/api/presets")
        self.assertEqual(presets_res.status_code, 200)
        ckpt_ids = [c["id"] for c in presets_res.json()["checkpoints"]]
        self.assertIn("ckpt_ensemble", ckpt_ids)

        p1 = "ACGT" * 90  # 360 bp
        p2 = "TGCA" * 90  # 360 bp
        rec = p1[:180] + p2[180:]
        fasta = "\n".join([
            ">Rec1", rec,
            ">P1_a", p1,
            ">P1_b", p1,
            ">P2_a", p2,
            ">P2_b", p2,
            ">Out", p1,
        ])
        res = self.client.post(
            "/api/infer",
            data={
                "checkpoint": "ckpt_ensemble",
                "scoring_mode": "hybrid_flank",
                "use_mirror_tta": "true",
                "pasted_fasta": fasta,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        self.assertTrue(payload["config"]["use_mirror_tta"])
        self.assertEqual(payload["config"]["checkpoint"], "ckpt_ensemble")
        stage4 = next(s for s in payload["timings"]["stages"] if s["stage_id"] == "transformer_fwd")
        self.assertIn("Ensemble", stage4["notes"])
        self.assertIn("Mirror TTA", stage4["notes"])
        evs = payload["consolidated_events"]
        self.assertTrue(any(abs(ev["most_plausible_bp_orig"] - 181) <= 30 for ev in evs), evs)

    def test_18_chat_structured_run_context_summary(self) -> None:
        prev_ant = os.environ.pop("ANTHROPIC_MODEL", None)
        try:
            self.assertEqual(chat_mod.get_active_model("anthropic"), "claude-haiku-4-5-20251001")
        finally:
            if prev_ant is not None:
                os.environ["ANTHROPIC_MODEL"] = prev_ant
        giant_track = list(range(5000))
        ctx = {
            "alignment": {
                "dataset_name": "hiv_test",
                "source_type": "upload",
                "n_sequences": 8,
                "genome_length_bp": 1500,
                "n_informative_cols": 140,
                "n_transformed_cols": 140,
                "gc_content_pct": 47.5,
                "edge_dead_zone_bp": 30,
            },
            "config": {
                "checkpoint": "ckpt_ensemble",
                "scoring_mode": "hybrid_flank",
                "use_mirror_tta": True,
                "z_thresh": 6.0,
                "tf_logit_thresh": 0.8,
            },
            "timings": {"total_ms": 1234.5},
            "warnings": ["Edge limit note"],
            "track_profile": {
                "positions_orig_bp": giant_track,
                "flank_kl_z": giant_track,
                "tf_prob": giant_track,
            },
            "consolidated_events": [
                {
                    "event_id": 1,
                    "ancestral_event_id": 1,
                    "most_plausible_bp_orig": 405,
                    "span_orig_start": 399,
                    "span_orig_end": 411,
                    "score_z": 12.4,
                    "tf_prob": 0.91,
                    "tf_logit": 2.31,
                    "signal_attribution": "hybrid_flank",
                    "match_status": "TP",
                    "recombinants": [
                        {
                            "seq_name": "R1_rec",
                            "p_switch": 18.2,
                            "ext_parent_left_name": "P1",
                            "ext_parent_right_name": "P2",
                            "left_sim": 0.99,
                            "right_sim": 0.98,
                        }
                    ],
                }
            ],
            "diagnostics": {
                "has_ground_truth": True,
                "tolerance_metrics": [
                    {"tolerance_bp": 30, "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "median_error_bp": 4.0}
                ],
                "ground_truth_items": [],
                "summary_notes": ["Exact match"],
            },
        }
        prompt = chat_mod._build_system_prompt(ctx)
        self.assertIn("Ancestral #1", prompt)
        self.assertIn("R1_rec", prompt)
        self.assertIn("MirrorTTA: True", prompt)
        self.assertNotIn("4999", prompt)

    def test_19_rare_carrier_max_kl_and_precision_filters(self) -> None:
        from gt_core import haldec as H
        from schemas import InferenceConfig

        # 1a. Rare 1-of-60 carrier amidst diverse clade polymorphism: z_max > z_sum
        rng = np.random.default_rng(7)
        m_div, s_div = 600, 60
        cA = (rng.random(m_div) < 0.5).astype(np.uint8)
        cB = (rng.random(m_div) < 0.5).astype(np.uint8)
        hap_div = np.zeros((m_div, s_div), dtype=np.uint8)
        hap_div[:300, 0] = cA[:300]
        hap_div[300:, 0] = cB[300:]
        for seq_i in range(1, 30):
            row = cA.copy()
            row[rng.integers(0, m_div, size=80)] ^= 1
            hap_div[:, seq_i] = row
        for seq_i in range(30, s_div):
            row = cB.copy()
            row[rng.integers(0, m_div, size=80)] ^= 1
            hap_div[:, seq_i] = row
        z_sum_d, z_max_d, _, _ = H.boundary_zscore_flank_dual(hap_div, L_w=120, min_seg=20, beta=20.0, power=1.0, n_perm=3)
        self.assertGreater(float(np.max(z_max_d[280:321])), float(np.max(z_sum_d[280:321])))

        # 1b. End-to-end 1-of-20 carrier inference detects exact breakpoint at bp 301
        p1 = "ACGT" * 150  # 600 bp
        p2 = "TGCA" * 150  # 600 bp
        rec_rare = p1[:300] + p2[300:]
        fasta_rare = "\n".join([">Rec_rare", rec_rare] + [f">P1_{i}\n{p1}" for i in range(9)] + [f">P2_{i}\n{p2}" for i in range(10)])
        res_rare = web_app.engine.run_inference(
            InferenceConfig(checkpoint="ckpt_N10000", scoring_mode="flank_kl"),
            fasta_text=fasta_rare,
            fasta_filename="rare20.fasta",
        )
        bps_rare = [e.most_plausible_bp_orig for e in res_rare.consolidated_events]
        self.assertTrue(any(abs(bp - 301) <= 30 for bp in bps_rare), bps_rare)

        # 2. Short 60bp recombinant tract: opposite-direction consolidation guard + end-to-end 04_EDGE
        from pathlib import Path
        from gt_core import report as REP

        raw_opp = [
            {
                "boundary_tx": 100,
                "breakpoint_orig": 704,
                "score": 20.4,
                "recombinants": [{"recombinant": 0, "parent_left": 1, "parent_right": 2, "switch": 2.1}],
            },
            {
                "boundary_tx": 120,
                "breakpoint_orig": 761,
                "score": 20.8,
                "recombinants": [{"recombinant": 0, "parent_left": 2, "parent_right": 1, "switch": 2.2}],
            },
        ]
        cons_opp = REP.consolidate_peaks(raw_opp, merge_radius_bp=75)
        self.assertEqual(len(cons_opp), 2, cons_opp)
        self.assertEqual(sorted(e["most_plausible_breakpoint_orig"] for e in cons_opp), [704, 761])

        raw_same = [
            raw_opp[0],
            {
                "boundary_tx": 120,
                "breakpoint_orig": 761,
                "score": 20.8,
                "recombinants": [{"recombinant": 0, "parent_left": 1, "parent_right": 2, "switch": 2.2}],
            },
        ]
        cons_same = REP.consolidate_peaks(raw_same, merge_radius_bp=75)
        self.assertEqual(len(cons_same), 1, cons_same)

        edge60 = Path(__file__).resolve().parents[1] / "fixtures" / "04_EDGE_short_tract_60bp.fasta"
        if edge60.exists():
            res_short = web_app.engine.run_inference(
                InferenceConfig(checkpoint="ckpt_N10000", scoring_mode="flank_kl"),
                fasta_text=edge60.read_text(encoding="utf-8"),
                fasta_filename=edge60.name,
            )
            bps_short = [e.most_plausible_bp_orig for e in res_short.consolidated_events]
            self.assertEqual(len(bps_short), 2, bps_short)
            self.assertTrue(any(abs(bp - 701) <= 25 for bp in bps_short), bps_short)
            self.assertTrue(any(abs(bp - 761) <= 25 for bp in bps_short), bps_short)

    def test_20_logmeanexp_ensemble_calibration(self) -> None:
        fused = GTTransformerInferenceEngine._fuse_ensemble_logits([
            np.array([2.5, -1.0, 100.0], dtype=np.float32),
            np.array([-2.0, -1.2, -100.0], dtype=np.float32),
        ])
        self.assertEqual(fused.shape, (3,))
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertGreater(float(fused[0]), 1.75)
        self.assertLess(float(fused[1]), -1.0)
        self.assertGreater(float(fused[2]), 99.0)

    def test_21_short_window_dual_kl_and_v2_checkpoint(self) -> None:
        from gt_core import haldec as H
        from schemas import InferenceConfig

        # Regression test for Bug 2A: L_w < min_seg must produce non-zero dual KL peaks
        p1 = "ACGT" * 50  # 200 bp
        p2 = "TGCA" * 50  # 200 bp
        rec = p1[:100] + p2[100:]
        seqs = [rec] + [p1] * 5 + [p2] * 6
        hap = np.array([[1 if ch in ("A", "G") else 0 for ch in s] for s in seqs], dtype=np.uint8).T
        score_sum, score_max = H.flank_kl_score_dual(hap, L_w=16, min_seg=50, beta=20.0, power=1.0)
        self.assertGreater(float(score_sum[100]), 0.1)
        self.assertGreater(float(score_max[100]), 0.5)

        ckpts = {c["id"] for c in web_app.engine.list_checkpoints()}
        if "ckpt_N10000_v2" in ckpts:
            p_a = "ACGT" * 90
            p_b = "TGCA" * 90
            rec_a = p_a[:180] + p_b[180:]
            fasta = "\n".join([
                ">Rec1", rec_a,
                ">P1a", p_a,
                ">P1b", p_a,
                ">P2a", p_b,
                ">P2b", p_b,
                ">Out", p_a,
            ])
            res = web_app.engine.run_inference(
                InferenceConfig(checkpoint="ckpt_N10000_v2", scoring_mode="hybrid_flank"),
                fasta_text=fasta,
                fasta_filename="v2_smoke.fasta",
            )
            bps = [e.most_plausible_bp_orig for e in res.consolidated_events]
            self.assertTrue(any(abs(bp - 181) <= 30 for bp in bps), bps)

            self.assertIn("ckpt_N40000", ckpts)
            res_40k = web_app.engine.run_inference(
                InferenceConfig(checkpoint="ckpt_N40000", scoring_mode="hybrid_flank"),
                fasta_text=fasta,
                fasta_filename="n40k_smoke.fasta",
            )
            bps_40k = [e.most_plausible_bp_orig for e in res_40k.consolidated_events]
            self.assertTrue(any(abs(bp - 181) <= 30 for bp in bps_40k), bps_40k)

        bench_res = self.client.get("/api/benchmarks")
        self.assertEqual(bench_res.status_code, 200)
        bdata = bench_res.json()
        self.assertIn("N=40,000 (28K In-Range)", bdata["scaling_table"]["headers"])
        self.assertGreater(float(bdata["eval_40k_summary"]["val_auprc_at_save"]), 0.83)


if __name__ == "__main__":
    unittest.main()
