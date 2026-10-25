"""
Phase 2 Route and End-to-End Functional Test Suite.
Tests all GET/POST endpoints, CSV batch processing, template downloads, and role permissions for all 6 tools.
"""
import io
import os
import sys
import unittest
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import create_app
from app.services import seif_service, ons_service, dhie_service, roua_service, moez_service, asma_service

class TestPhase2Routes(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

    def login(self, username="doctor"):
        if username == "doctor":
            self.client.post("/login", data={"email": "doctor@diacare.hospital", "password": "doctor123"})
        else:
            self.client.post("/login", data={"email": "reception@diacare.hospital", "password": "reception123"})

    # ── 1. Seif: Readmission Screening ────────────────────────────────────────
    def test_seif_get_and_post_single(self):
        self.login("doctor")
        # GET
        resp = self.client.get("/readmission")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Readmission Risk Screening", resp.data)

        # POST valid
        data = seif_service.DEFAULTS.copy()
        resp = self.client.post("/readmission", data=data)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Estimated Readmission Risk", resp.data)
        self.assertIn(b"Key Divergence Drivers", resp.data)

        # POST invalid
        bad_data = dict(data, num_medications=999)
        resp = self.client.post("/readmission", data=bad_data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn(b"Please review the following inputs", resp.data)

    def test_seif_batch_and_downloads(self):
        self.login("doctor")
        # Template
        resp = self.client.get("/readmission/batch/template")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "text/csv; charset=utf-8")

        # Upload CSV batch
        csv_buf = seif_service.generate_template_csv()
        resp = self.client.post("/readmission/batch", data={"file": (csv_buf, "test_batch.csv")},
                                content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Patients Evaluated", resp.data)
        self.assertIn(b"Prioritized Clinical Outreach List", resp.data)

        # Download scored CSV
        resp_dl = self.client.get("/readmission/batch/download")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertIn(b"risk", resp_dl.data)

    # ── 2. Ons: HbA1c & Treatment Decision ────────────────────────────────────
    def test_ons_get_and_post_single(self):
        self.login("doctor")
        # GET
        resp = self.client.get("/hba1c")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"HbA1c & Treatment Decision", resp.data)

        # POST valid
        data = ons_service.DEFAULTS.copy()
        resp = self.client.post("/hba1c", data=data)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Estimated Readmission Risk & Glycemic Status", resp.data)

        # POST invalid
        bad_data = dict(data, time_in_hospital=99)
        resp = self.client.post("/hba1c", data=bad_data)
        self.assertEqual(resp.status_code, 400)

    def test_ons_batch_and_downloads(self):
        self.login("doctor")
        resp = self.client.get("/hba1c/batch/template")
        self.assertEqual(resp.status_code, 200)

        csv_buf = ons_service.generate_template_csv()
        resp = self.client.post("/hba1c/batch", data={"file": (csv_buf, "test_batch.csv")},
                                content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Patients Evaluated", resp.data)

        resp_dl = self.client.get("/hba1c/batch/download")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertIn(b"risk", resp_dl.data)

    # ── 3. Dhie: Discharge Destination ────────────────────────────────────────
    def test_dhie_get_and_post_single(self):
        self.login("reception")
        # Reception has access!
        resp = self.client.get("/discharge")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Discharge Destination Prediction", resp.data)

        # POST valid home
        data = dhie_service.EXAMPLES["home"].copy()
        resp = self.client.post("/discharge", data=data)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Predicted Discharge Destination", resp.data)
        self.assertIn(b"Standard Discharge Pathway", resp.data)

        # POST valid transfer candidate
        data_tr = dhie_service.EXAMPLES["transfer"].copy()
        resp = self.client.post("/discharge", data=data_tr)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Transfer Facility Alert Flagged", resp.data)

    def test_dhie_batch_and_downloads(self):
        self.login("reception")
        resp = self.client.get("/discharge/batch/template")
        self.assertEqual(resp.status_code, 200)

        csv_buf = dhie_service.generate_template_csv()
        resp = self.client.post("/discharge/batch", data={"file": (csv_buf, "test_batch.csv")},
                                content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Transfer Alerts Flagged", resp.data)

        resp_dl = self.client.get("/discharge/batch/download")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertIn(b"prediction", resp_dl.data)

    # ── 4. Roua: Medication Load Assessment ───────────────────────────────────
    def test_roua_get_and_post_single(self):
        # Both reception and doctor can access!
        self.login("reception")
        resp = self.client.get("/medication")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Medication Load & Polypharmacy Assessment", resp.data)

        # POST valid
        data = roua_service.EXAMPLES["low"].copy()
        resp = self.client.post("/medication", data=data)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Estimated Medication Load Tier", resp.data)
        self.assertIn(b"Low Load", resp.data)

    def test_roua_batch_and_downloads(self):
        self.login("doctor")
        resp = self.client.get("/medication/batch/template")
        self.assertEqual(resp.status_code, 200)

        csv_buf = roua_service.generate_template_csv()
        resp = self.client.post("/medication/batch", data={"file": (csv_buf, "test_batch.csv")},
                                content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"High Polypharmacy", resp.data)

        resp_dl = self.client.get("/medication/batch/download")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertIn(b"predicted_class", resp_dl.data)

    # ── 5. Moez: Patient Profiles ─────────────────────────────────────────────
    def test_moez_info_and_placement(self):
        self.login("doctor")
        # Overview page
        resp = self.client.get("/profiles")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"The Two Clinical Patient Profiles", resp.data)
        self.assertIn(b"Profile 0", resp.data)
        self.assertIn(b"Profile 1", resp.data)

        # Placement GET
        resp = self.client.get("/profiles/patient")
        self.assertEqual(resp.status_code, 200)

        # Placement POST Profile 0
        data0 = moez_service.EXAMPLES["profile0"].copy()
        resp0 = self.client.post("/profiles/patient", data=data0)
        self.assertEqual(resp0.status_code, 200)
        self.assertIn(b"Assigned Patient Archetype", resp0.data)
        self.assertIn(b"Profile 0", resp0.data)

        # Placement POST Profile 1
        data1 = moez_service.EXAMPLES["profile1"].copy()
        resp1 = self.client.post("/profiles/patient", data=data1)
        self.assertEqual(resp1.status_code, 200)
        self.assertIn(b"Profile 1", resp1.data)

    def test_moez_reception_forbidden(self):
        self.login("reception")
        resp = self.client.get("/profiles")
        self.assertEqual(resp.status_code, 403)
        resp = self.client.get("/profiles/patient")
        self.assertEqual(resp.status_code, 403)

    # ── 6. Asma: Length of Stay ───────────────────────────────────────────────
    def test_asma_single_and_versions(self):
        self.login("reception")
        # Admission version
        resp = self.client.get("/los?version=admission")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Length of Stay Prediction", resp.data)

        data_adm = asma_service.EXAMPLES["short"]["admission"].copy()
        resp_adm = self.client.post("/los", data=dict(data_adm, version="admission"))
        self.assertEqual(resp_adm.status_code, 200)
        self.assertIn(b"Short stay", resp_adm.data)

        # After-stay version
        resp_stay = self.client.get("/los?version=after_stay")
        self.assertEqual(resp_stay.status_code, 200)

        data_stay = asma_service.EXAMPLES["long"]["after_stay"].copy()
        resp_post = self.client.post("/los", data=dict(data_stay, version="after_stay"))
        self.assertEqual(resp_post.status_code, 200)
        self.assertIn(b"Long stay", resp_post.data)

    def test_asma_batch_and_downloads(self):
        self.login("reception")
        resp = self.client.get("/los/batch/template?version=admission")
        self.assertEqual(resp.status_code, 200)

        csv_buf = asma_service.generate_template_csv(version="admission")
        resp = self.client.post("/los/batch", data={"file": (csv_buf, "test_batch.csv"), "version": "admission"},
                                content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Predicted Long Stays", resp.data)

        resp_dl = self.client.get("/los/batch/download")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertIn(b"prob_long_stay", resp_dl.data)

if __name__ == "__main__":
    unittest.main()
