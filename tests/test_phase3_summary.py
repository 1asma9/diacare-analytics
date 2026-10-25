"""
Phase 3 — Unified Patient Summary route tests.
Tests:
  1. GET /patient-summary → 200 for both roles
  2. POST /patient-summary (doctor) → 200, all 6 result sections present
  3. POST /patient-summary (reception) → 200, only 3 result sections present
  4. POST /patient-summary (doctor, high-risk form) → combined rules include at least one danger alert
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


BASE_FORM = {
    'age': '7', 'age_num': '65', 'gender': 'Female', 'race': 'Caucasian',
    'payer_code': 'MC', 'medical_specialty': 'InternalMedicine',
    'time_in_hospital': '4', 'num_lab_procedures': '43', 'num_procedures': '1',
    'num_medications': '15', 'number_outpatient': '0', 'number_emergency': '0',
    'number_inpatient': '0', 'number_diagnoses': '8',
    'n_meds_used': '1', 'n_meds_changed': '0', 'n_med_changes': '0',
    'diag_1': '428', 'diag_2': '250', 'diag_3': 'NA',
    'diag_1_grp': 'Circulatory',
    'admission_type_id': '1', 'admission_source_id': '7',
    'discharge_disposition_id': '1',
    'A1Cresult': 'None', 'max_glu_serum': 'None',
    'insulin': 'No', 'metformin': 'No',
    'change': 'No', 'diabetesMed': 'Yes',
    'a1c_tested': 'No', 'glu_tested': 'No',
    'sejours_anterieurs': '0',
    'discharge_seif': 'Home', 'admission_seif': 'Emergency',
}

HIGH_RISK_FORM = {
    'age': '9', 'age_num': '82', 'gender': 'Male', 'race': 'Caucasian',
    'payer_code': 'MC', 'medical_specialty': 'Cardiology',
    'time_in_hospital': '9', 'num_lab_procedures': '68', 'num_procedures': '3',
    'num_medications': '22', 'number_outpatient': '2', 'number_emergency': '2',
    'number_inpatient': '3', 'number_diagnoses': '11',
    'n_meds_used': '3', 'n_meds_changed': '2', 'n_med_changes': '2',
    'diag_1': '428', 'diag_2': '428', 'diag_3': '250',
    'diag_1_grp': 'Circulatory',
    'admission_type_id': '1', 'admission_source_id': '7',
    'discharge_disposition_id': '3',
    'A1Cresult': '>8', 'max_glu_serum': '>300',
    'insulin': 'Up', 'metformin': 'No',
    'change': 'Yes', 'diabetesMed': 'Yes',
    'a1c_tested': 'No', 'glu_tested': 'No',
    'sejours_anterieurs': '3',
    'discharge_seif': 'Skilled nursing facility', 'admission_seif': 'Emergency',
}


class TestPhase3Routes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()

    def _login(self, role='doctor'):
        email = 'doctor@diacare.hospital' if role == 'doctor' else 'reception@diacare.hospital'
        pwd   = 'doctor123' if role == 'doctor' else 'reception123'
        self.client.post('/login', data={'email': email, 'password': pwd}, follow_redirects=True)

    def _logout(self):
        self.client.get('/logout')

    # ── Test 1: GET for both roles ────────────────────────────────────────────
    def test_01_get_patient_summary_doctor(self):
        """Doctor can GET /patient-summary and /patient-report."""
        self._login('doctor')
        for url in ['/patient-summary', '/patient-report']:
            resp = self.client.get(url)
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(b'Unified Patient Report' in resp.data or b'Unified Patient Summary' in resp.data)
        self._logout()

    def test_02_get_patient_summary_reception(self):
        """Reception can GET /patient-summary and /patient-report."""
        self._login('reception')
        for url in ['/patient-summary', '/patient-report']:
            resp = self.client.get(url)
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(b'Unified Patient Report' in resp.data or b'Unified Patient Summary' in resp.data)
        self._logout()

    # ── Test 2: POST doctor — all 6 models present ───────────────────────────
    def test_03_post_doctor_all_models(self):
        """POST as doctor returns results for all 6 models."""
        self._login('doctor')
        resp = self.client.post('/patient-summary', data=BASE_FORM, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        html = resp.data
        # Check all model result sections appear
        self.assertIn(b"Length of Stay", html, "LOS result section missing")
        self.assertIn(b"Discharge Destination", html, "Discharge result section missing")
        self.assertIn(b"Medication Load", html, "Medication result section missing")
        self.assertIn(b"Readmission Risk", html, "Readmission result section missing")
        self.assertIn(b"HbA1c", html, "HbA1c result section missing")
        self.assertIn(b"Patient Profile", html, "Profile result section missing")
        self.assertIn(b"Integrated Clinical Guidance", html, "Combined rules section missing")
        self._logout()

    # ── Test 3: POST reception — only 3 models ───────────────────────────────
    def test_04_post_reception_three_models(self):
        """POST as reception returns results for 3 models only (no doctor-only content)."""
        self._login('reception')
        resp = self.client.post('/patient-summary', data=BASE_FORM, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        html = resp.data
        self.assertIn(b"Length of Stay", html)
        self.assertIn(b"Discharge Destination", html)
        self.assertIn(b"Medication Load", html)
        # Verify service: doctor-only models are None for reception
        with self.app.app_context():
            import app.services.summary_service as svc
            results = svc.run_all(BASE_FORM, role='reception')
            self.assertIsNotNone(results['los'])
            self.assertIsNotNone(results['discharge'])
            self.assertIsNotNone(results['medication'])
            self.assertIsNone(results['readmission'])
            self.assertIsNone(results['hba1c'])
            self.assertIsNone(results['profile'])
        self._logout()

    # ── Test 4: High-risk patient generates danger alerts ────────────────────
    def test_05_high_risk_combined_rules(self):
        """High-risk patient triggers danger-level combined rules."""
        self._login('doctor')
        resp = self.client.post('/patient-summary', data=HIGH_RISK_FORM, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        html = resp.data
        # At least one danger alert
        self.assertIn(b'alert-danger', html, "No danger alerts for high-risk patient")
        self.assertIn(b'Integrated Clinical Guidance', html)
        self._logout()

    # ── Test 5: Summary service directly ─────────────────────────────────────
    def test_06_summary_service_direct(self):
        """Direct call to summary_service.run_all with doctor role succeeds for all 6 models."""
        with self.app.app_context():
            import app.services.summary_service as svc
            results = svc.run_all(BASE_FORM, role='doctor')
            self.assertEqual(results['errors'], {})
            self.assertIsNotNone(results['los'])
            self.assertIsNotNone(results['discharge'])
            self.assertIsNotNone(results['medication'])
            self.assertIsNotNone(results['readmission'])
            self.assertIsNotNone(results['hba1c'])
            self.assertIsNotNone(results['profile'])
            self.assertIsInstance(results['combined_rules'], list)
            self.assertGreater(len(results['combined_rules']), 0)

    def test_07_summary_service_reception_role(self):
        """Direct call to summary_service.run_all with reception role returns None for doctor models."""
        with self.app.app_context():
            import app.services.summary_service as svc
            results = svc.run_all(BASE_FORM, role='reception')
            self.assertIsNotNone(results['los'])
            self.assertIsNotNone(results['discharge'])
            self.assertIsNotNone(results['medication'])
            self.assertIsNone(results['readmission'])
            self.assertIsNone(results['hba1c'])
            self.assertIsNone(results['profile'])


if __name__ == '__main__':
    unittest.main()
