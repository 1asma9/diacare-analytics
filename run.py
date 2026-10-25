"""
run.py — DiaCare Analytics entry point.
Loads environment variables from .env (if present) before starting the app.
"""
import os
from pathlib import Path

# Load .env if present (never required; app works without it)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass   # python-dotenv not installed — .env is silently ignored

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
