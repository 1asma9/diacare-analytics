"""
fda_service.py — Phase 6: openFDA Drug Label Reference Panel
=============================================================
Server-side service for querying the openFDA drug/label.json endpoint.

Rules:
- Only sends the drug NAME to the API; never patient values, session data or form fields.
- API key is read from the environment variable OPENFDA_API_KEY (never hardcoded).
- In-memory 24-hour cache per drug name (case-insensitive, normalised).
- Per-session rate-limiting: max SESSION_LIMIT lookups per hour.
- Falls back to a dated snapshot file if the live call fails.
- All network errors are caught; the rest of the app never depends on this module.

Confirmed API details (openFDA, verified 2026-10-25):
  Endpoint : GET https://api.fda.gov/drug/label.json
  Auth     : optional query param api_key= (free key from open.fda.gov)
  Rate     : 240 req/min, 1 000 req/day per IP without key
             240 req/min, 120 000 req/day per key
  Timeout  : 5 seconds (configured below)
  Strategy : search openfda.generic_name first, then openfda.brand_name
  Limit    : 1 result
"""

import os
import re
import json
import time
import hashlib
import logging
import pathlib
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

FDA_LABEL_URL  = "https://api.fda.gov/drug/label.json"
TIMEOUT_SEC    = 5          # seconds before giving up on the live call
CACHE_TTL_SEC  = 86_400     # 24 hours
SESSION_LIMIT  = 30         # max lookups per session per hour
SESSION_WINDOW = 3_600      # 1 hour in seconds
NAME_MAX_LEN   = 60
NAME_RE        = re.compile(r"^[A-Za-z0-9 \-]+$")

# Sections to display (in order); do NOT include dosage_and_administration or similar
SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "drug_interactions",
    "warnings",
]
SECTION_LABELS = {
    "boxed_warning":       "Boxed Warning",
    "indications_and_usage": "Indications & Usage",
    "drug_interactions":   "Drug Interactions",
    "warnings":            "Warnings",
}
SECTION_TRIM = 500          # characters per section before "Show more"

# Snapshot file (written by scripts/fetch_fda_snapshot.py)
SNAPSHOT_PATH = pathlib.Path(__file__).resolve().parents[2] / "app" / "static" / "fda_snapshot.json"

# ── In-memory cache: {normalised_name: {"ts": float, "payload": dict}} ────────
_cache: dict = {}

# Chip drugs derived from the dataset drug columns (20 drugs)
CHIP_DRUGS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


# ── Input validation ─────────────────────────────────────────────────────────

def validate_name(raw: str) -> tuple[Optional[str], Optional[str]]:
    """
    Validate and normalise a drug name.
    Returns (clean_name, error_message) — one of them is None.
    """
    name = raw.strip() if raw else ""
    if not name:
        return None, "Please enter a drug name."
    if len(name) > NAME_MAX_LEN:
        return None, f"Drug name must be at most {NAME_MAX_LEN} characters."
    if not NAME_RE.match(name):
        return None, "Drug name may only contain letters, digits, spaces and hyphens."
    return name.lower(), None


def _cache_key(name: str) -> str:
    """Normalised, stable cache key for a drug name."""
    normalised = unicodedata.normalize("NFC", name.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


# ── Per-session rate limiting ─────────────────────────────────────────────────

def check_session_limit(session: dict) -> tuple[bool, str]:
    """
    Check whether the current session has exceeded SESSION_LIMIT requests
    in the last SESSION_WINDOW seconds.
    Returns (allowed: bool, message: str).
    """
    now = time.time()
    history: list = session.get("fda_lookups", [])
    # Prune entries older than SESSION_WINDOW
    history = [t for t in history if now - t < SESSION_WINDOW]
    if len(history) >= SESSION_LIMIT:
        remaining = int(SESSION_WINDOW - (now - history[0]))
        mins = remaining // 60
        return False, (
            f"You have reached the lookup limit ({SESSION_LIMIT} per hour). "
            f"Please try again in {mins} minute(s)."
        )
    history.append(now)
    session["fda_lookups"] = history
    return True, ""


# ── API call ──────────────────────────────────────────────────────────────────

def _build_url(search_expr: str) -> str:
    """Build a safe, encoded FDA label API URL."""
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    params = {"search": search_expr, "limit": "1"}
    if api_key:
        params["api_key"] = api_key
    # Use requests.PreparedRequest for safe URL encoding
    req = requests.Request("GET", FDA_LABEL_URL, params=params)
    prepared = req.prepare()
    return prepared.url


def _fetch_label(name: str) -> Optional[dict]:
    """
    Query openFDA for a drug label.
    Tries generic_name first, then brand_name.
    Returns the first result dict, or None.
    Raises requests.RequestException on network failure (caller handles it).
    """
    # Quote the term so user text cannot alter the query syntax
    safe_name = name.replace('"', '\\"')
    strategies = [
        f'openfda.generic_name:"{safe_name}"',
        f'openfda.brand_name:"{safe_name}"',
    ]
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()

    for expr in strategies:
        params: dict = {"search": expr, "limit": "1"}
        if api_key:
            params["api_key"] = api_key

        resp = requests.get(
            FDA_LABEL_URL,
            params=params,
            timeout=TIMEOUT_SEC,
            headers={"User-Agent": "DiaCare-Analytics/1.0 (university project)"},
        )

        if resp.status_code == 404:
            continue            # No results for this strategy; try next
        if resp.status_code == 429:
            raise requests.HTTPError("Rate limit reached (429)", response=resp)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])
        if results:
            return results[0]

    return None   # Exhausted all strategies


# ── Section extraction ────────────────────────────────────────────────────────

def _extract_sections(raw: dict) -> dict:
    """Pull and trim each section from a raw FDA label dict."""
    out = {}
    openfda = raw.get("openfda", {})
    out["brand_names"]   = openfda.get("brand_name", [])
    out["generic_names"] = openfda.get("generic_name", [])
    out["label_date"]    = raw.get("effective_time", "")

    for key in SECTIONS:
        val = raw.get(key)
        if isinstance(val, list) and val:
            text = " ".join(val)
        elif isinstance(val, str) and val:
            text = val
        else:
            text = None
        out[key] = text  # None means "Not provided in this label"

    return out


def _trim(text: Optional[str], limit: int = SECTION_TRIM) -> tuple[str, bool]:
    """Return (trimmed_text, was_truncated)."""
    if not text:
        return "Not provided in this label.", False
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def _load_snapshot(name: str) -> Optional[dict]:
    """Load cached snapshot data for a drug from the snapshot file."""
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        return snap.get("drugs", {}).get(name.lower())
    except Exception:
        return None


def _snapshot_date() -> str:
    """Return the snapshot creation date, or empty string."""
    if not SNAPSHOT_PATH.exists():
        return ""
    try:
        snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        return snap.get("generated_at", "")[:10]
    except Exception:
        return ""


# ── Public entry point ────────────────────────────────────────────────────────

def lookup(name: str) -> dict:
    """
    Main entry point.  Returns a dict with keys:
      ok          : bool — True if data found
      source      : "live" | "cache" | "snapshot" | "error"
      snapshot_date: str (set when source == "snapshot")
      brand_names, generic_names, label_date
      boxed_warning, indications_and_usage, drug_interactions, warnings
        each as {"text": str, "truncated": bool}
      error       : str (set when ok is False)
    """
    key = _cache_key(name)
    now = time.time()

    # ── 1. Memory cache hit ───────────────────────────────────────────────────
    entry = _cache.get(key)
    if entry and (now - entry["ts"] < CACHE_TTL_SEC):
        payload = dict(entry["payload"])
        payload["source"] = "cache"
        return payload

    # ── 2. Live API call ──────────────────────────────────────────────────────
    try:
        raw = _fetch_label(name)
        if raw is None:
            # No match from API
            result = _no_match_payload(name)
            _cache[key] = {"ts": now, "payload": result}
            return result

        sections = _extract_sections(raw)
        payload = _build_payload(sections, source="live")
        _cache[key] = {"ts": now, "payload": payload}
        return payload

    except requests.Timeout:
        logger.warning("openFDA timeout for %s", name)
        return _fallback(name, "The openFDA server did not respond within 5 seconds.")

    except requests.HTTPError as exc:
        resp = exc.response
        if resp is not None and resp.status_code == 429:
            return _error_payload("openFDA rate limit reached. Please try again later.")
        logger.warning("openFDA HTTP error for %s: %s", name, exc)
        return _fallback(name, f"openFDA returned an error ({exc}).")

    except requests.ConnectionError:
        logger.warning("openFDA connection error for %s", name)
        return _fallback(name, "Could not connect to openFDA. Check your network.")

    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("openFDA malformed response for %s: %s", name, exc)
        return _fallback(name, "openFDA returned an unexpected response.")

    except Exception as exc:
        logger.exception("Unexpected error in fda_service.lookup(%s)", name)
        return _error_payload(f"An unexpected error occurred: {exc}")


def _build_payload(sections: dict, source: str, snapshot_date: str = "") -> dict:
    payload: dict = {
        "ok": True,
        "source": source,
        "snapshot_date": snapshot_date,
        "brand_names":   sections.get("brand_names", []),
        "generic_names": sections.get("generic_names", []),
        "label_date":    sections.get("label_date", ""),
    }
    for key in SECTIONS:
        text, trunc = _trim(sections.get(key))
        payload[key] = {"text": text, "truncated": trunc, "full": sections.get(key) or ""}
    return payload


def _no_match_payload(name: str) -> dict:
    return {
        "ok": False,
        "source": "live",
        "snapshot_date": "",
        "error": f'No FDA drug label found for "{name}". Try an alternative name or check the spelling.',
    }


def _error_payload(message: str) -> dict:
    return {
        "ok": False,
        "source": "error",
        "snapshot_date": "",
        "error": message,
    }


def _fallback(name: str, live_error: str) -> dict:
    """Try the snapshot; if missing return an error payload."""
    snap = _load_snapshot(name)
    if snap:
        date = _snapshot_date()
        return _build_payload(snap, source="snapshot", snapshot_date=date)
    return _error_payload(
        f"{live_error} No saved snapshot is available for this drug."
    )
