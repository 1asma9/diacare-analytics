import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
INPUTS_DIR = BASE_DIR / "inputs"
DATA_DIR = INPUTS_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "diacare-analytics-secret-key-2026")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    UPLOAD_MAX_ROWS = 5000
    
    # Session security settings
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    
    # Two demo accounts with hashed passwords
    # Doctor sees everything; Reception sees dashboard, summary, and admission & planning only.
    DEMO_USERS = {
        "doctor@diacare.hospital": {
            "email": "doctor@diacare.hospital",
            "name": "Dr. Sarah Connor",
            "role": "doctor",
            "role_title": "Attending Physician",
            "department": "Endocrinology & Internal Medicine",
            "password_hash": "scrypt:32768:8:1$cnR7McjhH553mBC0$872b750aaa199f0a7f0d024a87d4d71eb2be168f963a20bb97dc49fdf8617175be60359fb4e2910bfed313064e8606bbfc75e149e490ebe1dc810fbd85814d4f"
        },
        "reception@diacare.hospital": {
            "email": "reception@diacare.hospital",
            "name": "Alex Martinez",
            "role": "reception",
            "role_title": "Admission & Reception Staff",
            "department": "Patient Admissions & Discharge Bureau",
            "password_hash": "scrypt:32768:8:1$Acwm1ZcbnporvFD7$4c899a765fa121af8038a5ef1c58820e6c090a0a0700eacbf43d877c11d1be43d8daaece5888ecd9bb317138ac9aa8f4ea4583d7b025b2a461ac013ac24fccfa"
        }
    }

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    WTF_CSRF_ENABLED = False
