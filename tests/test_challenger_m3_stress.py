"""
Empirical Adversarial Stress Testing Harness for Milestone 3 Gate Verification.

Tests all mandated edge cases and boundaries against BOTH:
1. In-process FastAPI TestClient(web_app.app)
2. A live HTTP server started by the test on an available local port
"""
from __future__ import annotations

import io
import json
import socket
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient
import uvicorn

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app
from services.fasta_utils import (
    AlignmentValidationError,
    AllGapSequenceError,
    BinaryPayloadError,
    EmptyPayloadError,
    InsufficientInformativeSitesError,
    InsufficientSequencesError,
    UnequalLengthError,
    count_parsimony_informative_sites,
    parse_and_validate_fasta,
)

SAMPLE_DIR = WEB_DIR / "sample_data"
LIVE_SERVER_URL = "http://127.0.0.1:8050/api/infer"


def live_http_post(payload_dict: dict | None = None, raw_body: bytes | None = None, content_type: str = "application/json") -> tuple[int, dict, str]:
    """Execute live HTTP POST against the running server."""
    if raw_body is not None:
        data = raw_body
    else:
        data = json.dumps(payload_dict or {}).encode("utf-8")
    
    headers = {"Content-Type": content_type}
    req = urllib.request.Request(LIVE_SERVER_URL, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body_str = resp.read().decode("utf-8")
            ct = resp.headers.get("content-type", "")
            try:
                return resp.status, json.loads(body_str), ct
            except Exception:
                return resp.status, {"raw": body_str}, ct
    except urllib.error.HTTPError as exc:
        body_str = exc.read().decode("utf-8")
        ct = exc.headers.get("content-type", "")
        try:
            return exc.code, json.loads(body_str), ct
        except Exception:
            return exc.code, {"raw": body_str}, ct


class TestChallengerM3Stress(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(web_app.app)
        web_app.engine.warmup("ckpt_N10000_extreme")

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        global LIVE_SERVER_URL
        LIVE_SERVER_URL = f"http://127.0.0.1:{port}/api/infer"
        config = uvicorn.Config(web_app.app, host="127.0.0.1", port=port, log_level="error")
        cls._server = uvicorn.Server(config)
        cls._thread = threading.Thread(target=cls._server.run, daemon=True)
        cls._thread.start()

        health_url = f"http://127.0.0.1:{port}/healthz"
        for _ in range(100):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Live test server did not start")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.should_exit = True
        cls._thread.join(timeout=10)
        # uvicorn shutdown closes the app-wide executors; restore them so other
        # test modules can keep submitting jobs in the same process.
        web_app._JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gt_infer_worker")
        web_app._RETRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gt_retrain_worker")

    # =========================================================================
    # BATTERY 1: Malformed Inputs & HTTP Status Codes (Client & Live HTTP)
    # =========================================================================

    def test_01_empty_string_json_payload(self) -> None:
        # TestClient
        res = self.client.post("/api/infer", json={"fasta_text": ""})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

        # Live HTTP
        code, body, ct = live_http_post({"fasta_text": ""})
        self.assertEqual(code, 400)
        self.assertEqual(body, {"detail": "Empty alignment payload"})
        self.assertIn("application/json", ct)

    def test_02_whitespace_json_payload(self) -> None:
        ws = "   \n\t\r  \n  \t  \n"
        res = self.client.post("/api/infer", json={"pasted_fasta": ws})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

        code, body, ct = live_http_post({"pasted_fasta": ws})
        self.assertEqual(code, 400)
        self.assertEqual(body, {"detail": "Empty alignment payload"})
        self.assertIn("application/json", ct)

    def test_03_empty_body_json(self) -> None:
        res = self.client.post("/api/infer", json={})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

        code, body, ct = live_http_post({})
        self.assertEqual(code, 400)
        self.assertEqual(body, {"detail": "Empty alignment payload"})
        self.assertIn("application/json", ct)

    def test_04_empty_file_upload(self) -> None:
        files = {"fasta_file": ("empty.fasta", io.BytesIO(b""), "text/plain")}
        res = self.client.post("/api/infer", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

    def test_05_binary_blob_null_bytes_in_file(self) -> None:
        content = b">s1\nACGT\x00ACGT\n>s2\nACGTACGT\n>s3\nACGTACGT\n>s4\nACGTACGT\n"
        files = {"fasta_file": ("binary.fasta", io.BytesIO(content), "application/octet-stream")}
        res = self.client.post("/api/infer", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

    def test_06_binary_blob_null_bytes_in_json(self) -> None:
        payload = {"fasta_text": ">s1\nACGT\x00ACGT\n>s2\nACGTACGT\n>s3\nACGTACGT\n>s4\nACGTACGT\n"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

        code, body, ct = live_http_post(payload)
        self.assertEqual(code, 400)
        self.assertEqual(body, {"detail": "Binary or non-text alignment content detected"})
        self.assertIn("application/json", ct)

    def test_07_binary_blob_gzip_magic_header(self) -> None:
        gzip_blob = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xffFAKEGZIP"
        files = {"fasta_file": ("archive.tar.gz", io.BytesIO(gzip_blob), "application/octet-stream")}
        res = self.client.post("/api/infer", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

    def test_08_binary_control_chars_rejected(self) -> None:
        # Non-printable control char \x07 (bell)
        payload = {"fasta_text": ">s1\nACGT\x07ACGT\n>s2\nACGTACGT\n>s3\nACGTACGT\n>s4\nACGTACGT\n"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

    def test_09_unaligned_sequence_headers_unequal_lengths(self) -> None:
        payload = {"fasta_text": ">s1\nACGTACGTAC\n>s2\nACGTACGT\n>s3\nACGTACGTAC\n>s4\nACGTACGTAC\n"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Alignment sequences must have identical lengths"})

        code, body, ct = live_http_post(payload)
        self.assertEqual(code, 422)
        self.assertEqual(body, {"detail": "Alignment sequences must have identical lengths"})
        self.assertIn("application/json", ct)

    def test_10_unaligned_single_base_difference(self) -> None:
        payload = {"fasta_text": ">s1\n" + "A" * 100 + "\n>s2\n" + "A" * 101 + "\n>s3\n" + "A" * 100 + "\n>s4\n" + "A" * 100}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Alignment sequences must have identical lengths"})

    def test_11_panel_size_under_4_crf07(self) -> None:
        crf_path = SAMPLE_DIR / "CRF_07_test.fasta"
        raw = crf_path.read_text(encoding="utf-8")
        payload = {"fasta_text": raw, "fasta_filename": "CRF_07_test.fasta"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Alignment requires at least 4 sequences for recombination detection"})

        code, body, ct = live_http_post(payload)
        self.assertEqual(code, 422)
        self.assertEqual(body, {"detail": "Alignment requires at least 4 sequences for recombination detection"})
        self.assertIn("application/json", ct)

    def test_12_panel_size_under_4_synthetic_1_2_3_seqs(self) -> None:
        for n_seq in [1, 2, 3]:
            fasta = "\n".join(f">seq_{i}\n" + "ACGT" * 25 for i in range(n_seq))
            res = self.client.post("/api/infer", json={"fasta_text": fasta})
            self.assertEqual(res.status_code, 422, f"Failed for n_seq={n_seq}")
            self.assertEqual(res.json(), {"detail": "Alignment requires at least 4 sequences for recombination detection"})

    def test_13_informative_sites_under_10_sarbecovirus_5(self) -> None:
        sar_path = SAMPLE_DIR / "sarbecovirus_5_aligned.fasta"
        raw = sar_path.read_text(encoding="utf-8")
        payload = {"fasta_text": raw, "fasta_filename": "sarbecovirus_5_aligned.fasta"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.json(),
            {"detail": "Insufficient informative polymorphic sites (found 9, required >= 10; only 9 parsimony-informative sites)"},
        )

        code, body, ct = live_http_post(payload)
        self.assertEqual(code, 422)
        self.assertEqual(
            body,
            {"detail": "Insufficient informative polymorphic sites (found 9, required >= 10; only 9 parsimony-informative sites)"},
        )
        self.assertIn("application/json", ct)

    def test_14_all_gap_sequence_pure_dashes(self) -> None:
        payload = {"fasta_text": ">s1\nACGTACGTAC\n>seq_gap\n----------\n>s3\nACGTACGTAC\n>s4\nACGTACGTAC\n"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Sequence 'seq_gap' contains only gap characters"})

        code, body, ct = live_http_post(payload)
        self.assertEqual(code, 422)
        self.assertEqual(body, {"detail": "Sequence 'seq_gap' contains only gap characters"})
        self.assertIn("application/json", ct)

    def test_15_all_gap_sequence_mixed_gap_chars(self) -> None:
        payload = {"fasta_text": ">s1\nACGTACGTAC\n>s2\n-.~.-.~.-.\n>s3\nACGTACGTAC\n>s4\nACGTACGTAC\n"}
        res = self.client.post("/api/infer", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Sequence 's2' contains only gap characters"})

        code, body, ct = live_http_post(payload)
        self.assertEqual(code, 422)
        self.assertEqual(body, {"detail": "Sequence 's2' contains only gap characters"})
        self.assertIn("application/json", ct)

    # =========================================================================
    # BATTERY 2: Boundary Conditions (10 vs 9 sites, 24% vs 26% gaps)
    # =========================================================================

    def test_16_boundary_exactly_10_informative_sites_returns_200(self) -> None:
        # Construct alignment with exactly 10 parsimony-informative sites
        s1 = "A" * 10 + "A" * 90
        s2 = "A" * 10 + "A" * 90
        s3 = "T" * 10 + "A" * 90
        s4 = "T" * 10 + "A" * 90
        fasta = f">s1\n{s1}\n>s2\n{s2}\n>s3\n{s3}\n>s4\n{s4}\n"

        # Verify parsimony count is exactly 10
        cnt, mode = count_parsimony_informative_sites([s1, s2, s3, s4])
        self.assertEqual(cnt, 10)
        self.assertEqual(mode, "strict_acgt")

        # In-process TestClient
        res = self.client.post("/api/infer", json={"fasta_text": fasta})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["alignment"]["n_sequences"], 4)
        self.assertEqual(data["alignment"]["n_informative_cols"], 10)
        self.assertIn("consolidated_events", data)

        # Live HTTP
        code, body, ct = live_http_post({"fasta_text": fasta})
        self.assertEqual(code, 200)
        self.assertEqual(body["alignment"]["n_informative_cols"], 10)
        self.assertIn("application/json", ct)

    def test_17_boundary_exactly_9_informative_sites_returns_422(self) -> None:
        # Construct alignment with exactly 9 parsimony-informative sites
        s1 = "A" * 9 + "A" * 91
        s2 = "A" * 9 + "A" * 91
        s3 = "T" * 9 + "A" * 91
        s4 = "T" * 9 + "A" * 91
        fasta = f">s1\n{s1}\n>s2\n{s2}\n>s3\n{s3}\n>s4\n{s4}\n"

        # Verify parsimony count is exactly 9
        cnt, mode = count_parsimony_informative_sites([s1, s2, s3, s4])
        self.assertEqual(cnt, 9)
        self.assertEqual(mode, "strict_acgt")

        # In-process TestClient
        res = self.client.post("/api/infer", json={"fasta_text": fasta})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.json(),
            {"detail": "Insufficient informative polymorphic sites (found 9, required >= 10; only 9 parsimony-informative sites)"},
        )

        # Live HTTP
        code, body, ct = live_http_post({"fasta_text": fasta})
        self.assertEqual(code, 422)
        self.assertEqual(
            body,
            {"detail": "Insufficient informative polymorphic sites (found 9, required >= 10; only 9 parsimony-informative sites)"},
        )
        self.assertIn("application/json", ct)

    def test_18_boundary_24_percent_gaps_returns_200(self) -> None:
        # Construct alignment with 50 sequences, 15 polymorphic columns, 24% gaps (12/50)
        S = 50
        L = 100
        lines = []
        raw_seqs = []
        for i in range(S):
            row = []
            for _c in range(15):
                if i < 12:  # 12 / 50 = 24.0% gaps
                    row.append("-")
                elif i < 12 + 19:
                    row.append("A")
                else:
                    row.append("T")
            row.extend(["A"] * 85)
            seq_str = "".join(row)
            raw_seqs.append(seq_str)
            lines.append(f">s{i:02d}\n{seq_str}")
        fasta = "\n".join(lines)

        # Verify parsimony count is 15 in gap_tolerant mode
        cnt, mode = count_parsimony_informative_sites(raw_seqs)
        self.assertEqual(cnt, 15)
        self.assertEqual(mode, "gap_tolerant")

        # In-process TestClient
        res = self.client.post("/api/infer", json={"fasta_text": fasta})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["alignment"]["n_sequences"], 50)
        self.assertTrue(any("Gap-tolerant fallback engaged" in w for w in data["warnings"]))
        self.assertIn("gap-tolerant fallback", data["timings"]["stages"][1]["notes"])

        # Live HTTP
        code, body, ct = live_http_post({"fasta_text": fasta})
        self.assertEqual(code, 200)
        self.assertTrue(any("Gap-tolerant fallback engaged" in w for w in body["warnings"]))
        self.assertIn("application/json", ct)

    def test_19_boundary_26_percent_gaps_returns_422(self) -> None:
        # Construct alignment with 50 sequences, 15 polymorphic columns, 26% gaps (13/50)
        S = 50
        L = 100
        lines = []
        raw_seqs = []
        for i in range(S):
            row = []
            for _c in range(15):
                if i < 13:  # 13 / 50 = 26.0% gaps (> 25% max)
                    row.append("-")
                elif i < 13 + 18:
                    row.append("A")
                else:
                    row.append("T")
            row.extend(["A"] * 85)
            seq_str = "".join(row)
            raw_seqs.append(seq_str)
            lines.append(f">s{i:02d}\n{seq_str}")
        fasta = "\n".join(lines)

        # Verify parsimony count is 0 because gap fraction exceeds max_gap_frac=0.25
        cnt, mode = count_parsimony_informative_sites(raw_seqs)
        self.assertEqual(cnt, 0)
        self.assertEqual(mode, "strict_acgt")

        # In-process TestClient
        res = self.client.post("/api/infer", json={"fasta_text": fasta})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.json(),
            {"detail": "Insufficient informative polymorphic sites (found 0, required >= 10; only 0 parsimony-informative sites)"},
        )

        # Live HTTP
        code, body, ct = live_http_post({"fasta_text": fasta})
        self.assertEqual(code, 422)
        self.assertEqual(
            body,
            {"detail": "Insufficient informative polymorphic sites (found 0, required >= 10; only 0 parsimony-informative sites)"},
        )
        self.assertIn("application/json", ct)

    def test_20_boundary_exact_25_percent_gaps_returns_200(self) -> None:
        # Boundary test: exactly 25.0% gaps (10 / 40 = 0.25)
        S = 40
        lines = []
        raw_seqs = []
        for i in range(S):
            row = []
            for _c in range(15):
                if i < 10:  # 10 / 40 = exactly 25.0%
                    row.append("-")
                elif i < 10 + 15:
                    row.append("A")
                else:
                    row.append("T")
            row.extend(["A"] * 85)
            seq_str = "".join(row)
            raw_seqs.append(seq_str)
            lines.append(f">s{i:02d}\n{seq_str}")
        fasta = "\n".join(lines)

        cnt, mode = count_parsimony_informative_sites(raw_seqs)
        self.assertEqual(cnt, 15)
        self.assertEqual(mode, "gap_tolerant")

        res = self.client.post("/api/infer", json={"fasta_text": fasta})
        self.assertEqual(res.status_code, 200)

        code, body, _ct = live_http_post({"fasta_text": fasta})
        self.assertEqual(code, 200)

    # =========================================================================
    # BATTERY 3: Zero Unhandled 500 HTML Tracebacks Invariant
    # =========================================================================

    def test_21_no_unhandled_500_html_traceback_on_arbitrary_garbage(self) -> None:
        garbage_payloads = [
            "completely unformatted non-fasta text",
            ">>>malformed>>>headers<<<",
            ">seq1\nACGT\n>seq2\n",
            "\n\n\n   \t\t\n",
            ">s1\n" + "Z" * 100 + "\n>s2\n" + "X" * 100 + "\n>s3\n" + "J" * 100 + "\n>s4\n" + "Q" * 100,
            ">s1\n12345\n>s2\n67890\n>s3\n12345\n>s4\n67890",
            '{"not": "even fasta"}',
        ]
        for p in garbage_payloads:
            # Client
            res = self.client.post("/api/infer", json={"fasta_text": p})
            self.assertIn(res.status_code, (400, 422), f"Payload triggered status {res.status_code}: {p[:30]}")
            self.assertIn("application/json", res.headers.get("content-type", ""))
            body = res.json()
            self.assertIn("detail", body)
            self.assertIsInstance(body["detail"], str)
            self.assertNotIn("Traceback (most recent call last)", body["detail"])
            self.assertNotIn("<!DOCTYPE html>", res.text)

            # Live HTTP
            code, live_body, ct = live_http_post({"fasta_text": p})
            self.assertIn(code, (400, 422))
            self.assertIn("application/json", ct)
            self.assertIn("detail", live_body)
            self.assertNotIn("Traceback (most recent call last)", str(live_body))


if __name__ == "__main__":
    unittest.main()
