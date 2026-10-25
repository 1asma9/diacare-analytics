"""
Phase 1 Foundation Automated Verification Tests (updated for refactor)
Tests authentication, role-based access control, sitemap routes, error pages.
"""

import sys
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.config import TestingConfig


def get_test_client():
    app = create_app(TestingConfig)
    return app.test_client()


def test_unauthenticated_redirects():
    """All protected routes must redirect unauthenticated users to /login."""
    client = get_test_client()
    protected_routes = [
        "/dashboard",
        "/patient-summary",
        "/los",
        "/los/batch",
        "/discharge",
        "/discharge/batch",
        "/readmission",
        "/readmission/batch",
        "/hba1c",
        "/hba1c/batch",
        "/medication",
        "/medication/batch",
        "/profiles",
        "/profiles/patient",
    ]
    for route in protected_routes:
        resp = client.get(route, follow_redirects=False)
        assert resp.status_code == 302, f"Expected 302 for unauthenticated {route}, got {resp.status_code}"
        assert "/login" in resp.headers["Location"], f"Expected redirect to /login for {route}"
    print("PASS: test_unauthenticated_redirects (14 routes verified)")


def test_public_pages_accessible():
    """Login and terms pages are accessible without authentication."""
    client = get_test_client()
    public_routes = ["/login", "/terms"]
    for route in public_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Expected 200 for {route}, got {resp.status_code}"
    print("PASS: test_public_pages_accessible (2 routes verified)")


def test_privacy_page_removed():
    """Privacy page has been removed — /privacy must return 404."""
    client = get_test_client()
    resp = client.get("/privacy")
    assert resp.status_code == 404, f"Expected 404 for /privacy (removed), got {resp.status_code}"
    print("PASS: test_privacy_page_removed")


def test_login_validation():
    """Valid and invalid credentials behave correctly."""
    client = get_test_client()

    # Invalid login
    resp = client.post("/login", data={"email": "doctor@diacare.hospital", "password": "wrongpassword"})
    assert resp.status_code == 200
    assert b"Invalid credentials" in resp.data

    # Valid Doctor login
    resp = client.post("/login", data={"email": "doctor@diacare.hospital", "password": "doctor123"}, follow_redirects=False)
    assert resp.status_code == 302
    assert "/dashboard" in resp.headers["Location"]

    # Valid Reception login
    resp = client.post("/login", data={"email": "reception@diacare.hospital", "password": "reception123"}, follow_redirects=False)
    assert resp.status_code == 302
    assert "/dashboard" in resp.headers["Location"]
    print("PASS: test_login_validation")


def test_doctor_full_access():
    """Doctor receives HTTP 200 on all routes."""
    client = get_test_client()
    client.post("/login", data={"email": "doctor@diacare.hospital", "password": "doctor123"})

    all_routes = [
        "/dashboard",
        "/patient-summary",
        "/los",
        "/los/batch",
        "/discharge",
        "/discharge/batch",
        "/readmission",
        "/readmission/batch",
        "/hba1c",
        "/hba1c/batch",
        "/medication",
        "/medication/batch",
        "/profiles",
        "/profiles/patient",
        "/terms",
    ]
    for route in all_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Doctor should have access to {route}, got {resp.status_code}"
    print("PASS: test_doctor_full_access (15 routes verified)")


def test_reception_role_enforcement():
    """Reception has access to admission and medication tools; physician-only tools return 403."""
    client = get_test_client()
    client.post("/login", data={"email": "reception@diacare.hospital", "password": "reception123"})

    allowed_routes = [
        "/dashboard",
        "/patient-summary",
        "/los",
        "/los/batch",
        "/discharge",
        "/discharge/batch",
        "/medication",
        "/medication/batch",
        "/terms",
    ]
    for route in allowed_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Reception should have access to {route}, got {resp.status_code}"

    forbidden_routes = [
        "/readmission", "/readmission/batch",
        "/hba1c", "/hba1c/batch",
        "/profiles", "/profiles/patient",
    ]
    for route in forbidden_routes:
        resp = client.get(route)
        assert resp.status_code == 403, f"Reception must get 403 for {route}, got {resp.status_code}"
        assert b"403" in resp.data
    print("PASS: test_reception_role_enforcement (9 allowed + 6 forbidden routes verified)")


def test_logout():
    """Logout clears session; next protected access redirects to login."""
    client = get_test_client()
    client.post("/login", data={"email": "doctor@diacare.hospital", "password": "doctor123"})
    resp = client.get("/dashboard")
    assert resp.status_code == 200

    logout_resp = client.get("/logout", follow_redirects=False)
    assert logout_resp.status_code == 302
    assert "/login" in logout_resp.headers["Location"]

    dash_resp = client.get("/dashboard", follow_redirects=False)
    assert dash_resp.status_code == 302
    assert "/login" in dash_resp.headers["Location"]
    print("PASS: test_logout")


def test_404_handler():
    """Non-existent routes return 404 with custom template."""
    client = get_test_client()
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert b"Page Not Found" in resp.data or b"404" in resp.data
    print("PASS: test_404_handler")


def test_no_esprit_badge_in_login():
    """Login page must not contain ESPRIT 4CCE19 badge."""
    client = get_test_client()
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"4CCE19" not in resp.data, "ESPRIT 4CCE19 badge found in login page — must be removed"
    print("PASS: test_no_esprit_badge_in_login")


def test_model_checksums_integrity():
    """All 12 model files must match baseline SHA-256 checksums."""
    lines = open("docs/model_checksums.txt", encoding="utf-8-sig").read().strip().splitlines()
    assert len(lines) == 12, f"Expected 12 model files, found {len(lines)}"
    for line in lines:
        if not line.strip():
            continue
        expected, relpath = line.strip().split(maxsplit=1)
        with open(relpath, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == expected, f"Model checksum mismatch for {relpath}!"
    print("PASS: test_model_checksums_integrity (12/12 models verified identical)")


if __name__ == "__main__":
    test_unauthenticated_redirects()
    test_public_pages_accessible()
    test_privacy_page_removed()
    test_login_validation()
    test_doctor_full_access()
    test_reception_role_enforcement()
    test_logout()
    test_404_handler()
    test_no_esprit_badge_in_login()
    test_model_checksums_integrity()
    print("\nALL PHASE 1 AUTOMATED TESTS PASSED SUCCESSFULLY!")
