#!/usr/bin/env python3
"""
scripts/fetch_fda_snapshot.py
==============================
Fetches real FDA drug label data for the 20 chip drugs from the openFDA API
and saves a dated snapshot to app/static/fda_snapshot.json.

The snapshot is used as a fallback when the live API is unavailable.
ALL label text comes from real API responses; nothing is written by hand.

Usage:
    python scripts/fetch_fda_snapshot.py

Optional env var:
    OPENFDA_API_KEY  — free key from https://open.fda.gov/apis/authentication/
    If not set, the script uses the public (unauthenticated) tier.

Regenerate whenever you want a fresher copy. The snapshot file records
the date it was generated so the UI can show "Saved copy from <date>".
"""

import json
import os
import sys
import time
import pathlib
from datetime import datetime, timezone

# Allow running from project root or from scripts/ directory
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
TIMEOUT_SEC   = 10
DELAY_SEC     = 0.5   # between requests — stay well under rate limits

SNAPSHOT_OUT  = ROOT / "app" / "static" / "fda_snapshot.json"

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

SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "drug_interactions",
    "warnings",
]


def fetch_one(name: str, api_key: str) -> dict | None:
    """Try generic_name then brand_name. Returns the first result dict or None."""
    safe = name.replace('"', '\\"')
    strategies = [
        f'openfda.generic_name:"{safe}"',
        f'openfda.brand_name:"{safe}"',
    ]
    for expr in strategies:
        params: dict = {"search": expr, "limit": "1"}
        if api_key:
            params["api_key"] = api_key
        try:
            resp = requests.get(
                FDA_LABEL_URL,
                params=params,
                timeout=TIMEOUT_SEC,
                headers={"User-Agent": "DiaCare-Analytics/1.0 (university project)"},
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0]
        except Exception as exc:
            print(f"  ✗ Error fetching '{name}' ({expr}): {exc}")
            return None
    return None


def extract(raw: dict) -> dict:
    """Pull relevant sections from a raw FDA result."""
    openfda = raw.get("openfda", {})
    out = {
        "brand_names":   openfda.get("brand_name", []),
        "generic_names": openfda.get("generic_name", []),
        "label_date":    raw.get("effective_time", ""),
    }
    for sec in SECTIONS:
        val = raw.get(sec)
        if isinstance(val, list) and val:
            out[sec] = " ".join(val)
        elif isinstance(val, str) and val:
            out[sec] = val
        else:
            out[sec] = None
    return out


def main():
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    if api_key:
        print(f"Using API key (***{api_key[-4:]})")
    else:
        print("No OPENFDA_API_KEY set — using public (unauthenticated) tier (1 000 req/day limit).")

    drugs: dict = {}
    for i, name in enumerate(CHIP_DRUGS, 1):
        print(f"[{i:02d}/{len(CHIP_DRUGS)}] Fetching: {name} …", end=" ", flush=True)
        raw = fetch_one(name, api_key)
        if raw:
            drugs[name] = extract(raw)
            print("✓")
        else:
            print("✗ not found")
            drugs[name] = None
        if i < len(CHIP_DRUGS):
            time.sleep(DELAY_SEC)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source":       "openFDA drug/label.json",
        "note":         "Real data fetched from openFDA API. Do not edit by hand.",
        "drugs":        drugs,
    }

    SNAPSHOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_OUT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Snapshot saved to: {SNAPSHOT_OUT}")
    print(f"  Generated at: {snapshot['generated_at']}")
    found  = sum(1 for v in drugs.values() if v is not None)
    missed = len(drugs) - found
    print(f"  {found} found, {missed} not found")


if __name__ == "__main__":
    main()
