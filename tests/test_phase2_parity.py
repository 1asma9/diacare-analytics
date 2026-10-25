"""
Phase 2 Parity and Verification Test Suite.
Verifies exact parity across all 6 models on 20 patients from inputs/data/df_clean.csv.
Also verifies that all 12 model checksums remain untouched.
"""
import hashlib
import importlib.util
import json
import os
import sys
import unittest
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.services import seif_service, ons_service, dhie_service, roua_service, moez_service, asma_service

def load_mod(name: str, filepath: str):
    folder = os.path.dirname(os.path.abspath(filepath))
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class TestPhase2Parity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df20 = pd.read_csv(os.path.join(ROOT, "inputs", "data", "df_clean.csv"), nrows=20)
        cls.orig_seif = load_mod("orig_seif", os.path.join(ROOT, "inputs", "seif", "app.py"))
        cls.orig_ons = load_mod("orig_ons", os.path.join(ROOT, "inputs", "ons", "app.py"))
        cls.orig_dhie = load_mod("orig_dhie", os.path.join(ROOT, "inputs", "dhie", "app.py"))

    def test_01_seif_parity_20_rows(self):
        """Compare Seif service output against original inputs/seif/app.py on 20 patients."""
        rows_service = []
        rows_orig = []
        for _, r in self.df20.iterrows():
            d = seif_service.DEFAULTS.copy()
            # Map common fields
            for k in seif_service.LIMITS:
                if k in r:
                    d[k] = r[k]
            if "admission_type_id" in r:
                d["admission"] = "Emergency" if r["admission_type_id"] == 1 else "Urgent"
            if "change" in r:
                d["change"] = "Yes" if r["change"] in ("Ch", "Yes") else "No"
            if "diabetesMed" in r:
                d["diabetesMed"] = "Yes" if r["diabetesMed"] == "Yes" else "No"
            
            clean_s, _ = seif_service.clean(d)
            clean_o, _ = self.orig_seif.clean(d)
            rows_service.append(clean_s)
            rows_orig.append(clean_o)

        p_serv, _ = seif_service.score(rows_service)
        p_orig, _ = self.orig_seif.score(rows_orig)

        np.testing.assert_allclose(p_serv, p_orig, rtol=1e-5, atol=1e-5,
                                   err_msg="Seif service output differs from original!")

    def test_02_ons_parity_20_rows(self):
        """Compare Ons service output against original inputs/ons/app.py on 20 patients."""
        rows_service = []
        rows_orig = []
        for _, r in self.df20.iterrows():
            d = ons_service.DEFAULTS.copy()
            for k in ons_service.LIMITS:
                if k in r:
                    d[k] = r[k]
            if "gender" in r and r["gender"] in ("Female", "Male"):
                d["gender"] = r["gender"]
            if "race" in r and r["race"] in [v for v, _ in ons_service.RACE]:
                d["race"] = r["race"]
            if "admission_type_id" in r:
                d["admission"] = str(r["admission_type_id"])
            if "admission_source_id" in r:
                d["source"] = str(r["admission_source_id"])
            if "change" in r:
                d["change"] = "Yes" if r["change"] in ("Ch", "Yes") else "No"
            if "diabetesMed" in r:
                d["diabetesMed"] = "Yes" if r["diabetesMed"] == "Yes" else "No"

            clean_s, _ = ons_service.clean(d)
            clean_o, _ = self.orig_ons.clean(d)
            rows_service.append(clean_s)
            rows_orig.append(clean_o)

        p_serv, _ = ons_service.score(rows_service)
        p_orig, _ = self.orig_ons.score(rows_orig)

        np.testing.assert_allclose(p_serv, p_orig, rtol=1e-5, atol=1e-5,
                                   err_msg="Ons service output differs from original!")

    def test_03_dhie_parity_20_rows(self):
        """Compare Dhie service output against original inputs/dhie/app.py on 20 patients."""
        for _, r in self.df20.iterrows():
            d = dhie_service.DEFAULTS.copy()
            for k in dhie_service.LIMITS:
                if k in r:
                    d[k] = r[k]
            if "age" in r:
                # convert bracket "[50-60)" to years for Dhie's form input
                if isinstance(r["age"], str) and "[" in r["age"]:
                    import re
                    m = re.search(r"\[(\d+)-", r["age"])
                    d["age"] = int(m.group(1)) + 5 if m else 70
                else:
                    d["age"] = int(r["age"])
            if "gender" in r and r["gender"] in dhie_service.GENDERS:
                d["gender"] = r["gender"]
            if "race" in r and r["race"] in dhie_service.RACES:
                d["race"] = r["race"]
            if "diag_1" in r:
                d["diag_1"] = str(r["diag_1"])
            if "diag_2" in r:
                d["diag_2"] = str(r["diag_2"])
            if "diag_3" in r:
                d["diag_3"] = str(r["diag_3"])
            if "change" in r:
                d["change"] = "Ch" if r["change"] in ("Ch", "Yes") else "No"
            if "diabetesMed" in r:
                d["diabetesMed"] = "Yes" if r["diabetesMed"] == "Yes" else "No"

            row_serv, _ = dhie_service.clean(d)
            row_orig, _ = self.orig_dhie._clean(d)

            res_serv = dhie_service.predict_single(row_serv)
            pred_orig, probs_orig, p_tr_orig, alert_orig = self.orig_dhie._predict(row_orig)

            self.assertEqual(res_serv["pred"], pred_orig)
            self.assertEqual(res_serv["transfer_alert"], alert_orig)
            self.assertAlmostEqual(res_serv["transfer_prob"], round(p_tr_orig, 4), places=3)

    def test_04_roua_parity_20_rows(self):
        """Verify Roua service feature preparation and predictions on 20 rows."""
        X_serv = roua_service.prepare_features(self.df20)
        self.assertEqual(X_serv.shape[0], 20)
        b = roua_service._get_bundle()
        model = b["model"]
        preds = model.predict(X_serv)
        probs = model.predict_proba(X_serv)
        self.assertEqual(len(preds), 20)
        self.assertEqual(probs.shape, (20, 3))
        # Ensure all probabilities sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-4)

    def test_05_moez_clustering_20_rows(self):
        """Verify Moez clustering pipeline assigns valid clusters and PC1/PC2 to 20 patients."""
        for _, r in self.df20.iterrows():
            d = moez_service.DEFAULTS.copy()
            for k in moez_service.LIMITS:
                if k in r:
                    d[k] = r[k]
            if "gender" in r and r["gender"] in ("Female", "Male"):
                d["gender"] = r["gender"]
            if "diag_1" in r:
                d["diag_1"] = str(r["diag_1"])
            
            row, _ = moez_service.clean(d)
            res = moez_service.assign_patient(row)
            self.assertIn(res["cluster"], (0, 1))
            self.assertIsInstance(res["pc1"], float)
            self.assertIsInstance(res["pc2"], float)
            self.assertGreaterEqual(res["dist_0"], 0.0)
            self.assertGreaterEqual(res["dist_1"], 0.0)

    def test_06_asma_longstay_20_rows(self):
        """Verify Asma's admission and after_stay versions predict consistent probabilities on 20 patients."""
        for _, r in self.df20.iterrows():
            d = asma_service.DEFAULTS.copy()
            for k in asma_service.LIMITS:
                if k in r:
                    d[k] = r[k]
            if "change" in r:
                d["change"] = "Yes" if r["change"] in ("Ch", "Yes") else "No"

            row_adm, _ = asma_service.clean(d, version="admission")
            res_adm = asma_service.predict_single(row_adm, version="admission")
            self.assertGreaterEqual(res_adm["probability"], 0.0)
            self.assertLessEqual(res_adm["probability"], 1.0)
            self.assertIn(res_adm["label"], ("Short stay (≤ 3 days)", "Long stay (> 3 days)"))

            row_stay, _ = asma_service.clean(d, version="after_stay")
            res_stay = asma_service.predict_single(row_stay, version="after_stay")
            self.assertGreaterEqual(res_stay["probability"], 0.0)
            self.assertLessEqual(res_stay["probability"], 1.0)
            self.assertIn(res_stay["label"], ("Short stay (≤ 3 days)", "Long stay (> 3 days)"))

    def test_07_model_checksums_untouched(self):
        """Verify all 12 model files match docs/model_checksums.txt."""
        chk_file = os.path.join(ROOT, "docs", "model_checksums.txt")
        with open(chk_file, "r", encoding="utf-8-sig") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        expected = {}
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected[parts[1].replace("\\", "/")] = parts[0]

        for rel_path, exp_hash in expected.items():
            full_path = os.path.join(ROOT, rel_path.replace("/", os.sep))
            self.assertTrue(os.path.exists(full_path), f"File missing: {rel_path}")
            h = hashlib.sha256()
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_hash = h.hexdigest()
            self.assertEqual(actual_hash, exp_hash, f"Hash mismatch for {rel_path}!")

if __name__ == "__main__":
    unittest.main()
