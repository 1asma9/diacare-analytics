"""
api.py — Phase 6: JSON API routes for DiaCare Analytics
========================================================
Currently contains:
  GET /api/drug-label?name=<drug>  — openFDA drug label reference (doctors only)
"""

import re
from flask import Blueprint, jsonify, request, session, abort

from app.auth import get_current_user
from app.services import fda_service

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/drug-label", methods=["GET"])
def drug_label():
    """
    Look up a drug label from openFDA.
    Enforced: doctors only (server-side; returns 403 JSON for others).
    Query param: name (string, 1–60 chars, letters/digits/spaces/hyphens only)
    Returns JSON payload (see fda_service.lookup docstring).
    """
    # ── Role enforcement ──────────────────────────────────────────────────────
    user = get_current_user()
    if not user or user.get("role") != "doctor":
        return jsonify({"ok": False, "error": "Access restricted to doctors."}), 403

    # ── Input validation ──────────────────────────────────────────────────────
    raw = request.args.get("name", "")
    name, err = fda_service.validate_name(raw)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    # ── Per-session rate limit ────────────────────────────────────────────────
    allowed, msg = fda_service.check_session_limit(session)
    if not allowed:
        return jsonify({"ok": False, "error": msg}), 429

    # ── Lookup (live → cache → snapshot) ─────────────────────────────────────
    result = fda_service.lookup(name)
    status = 200 if result.get("ok") else 502
    return jsonify(result), status
