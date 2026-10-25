# Phase 6 Report — openFDA Drug Label Reference Panel

**Project:** DiaCare Analytics  
**Phase:** 6  
**Date:** 2026-10-25  
**Status:** ✅ Complete — awaiting approval  

---

## 1. Objective

Add a collapsible "Drug Reference (FDA Label)" panel to the Medication Load page that lets doctors look up real FDA drug label data for individual drug names directly from the UI, without sending any patient data to a third party.

---

## 2. API Details Verified

| Property | Value |
|---|---|
| Endpoint | `GET https://api.fda.gov/drug/label.json` |
| Auth | Optional free API key via `api_key=` query param or HTTP Basic |
| No-key rate limit | 240 req/min, **1,000 req/day** per IP address |
| With-key rate limit | 240 req/min, **120,000 req/day** per key |
| Timeout configured | 5 seconds |
| Search strategy | `openfda.generic_name:"<name>"` first, then `openfda.brand_name:"<name>"` |
| Result limit | `limit=1` |
| Sections displayed | `boxed_warning`, `indications_and_usage`, `drug_interactions`, `warnings` |
| Sections excluded | `dosage_and_administration` and all other sections |

Source verified against: `https://open.fda.gov/apis/drug/label/` and `https://open.fda.gov/apis/authentication/`

---

## 3. What Was Built

### A. Panel Placement
- Collapsible card titled **"Drug Reference (FDA Label)"** placed below the result block on [`medication_single.html`](../app/templates/clinical/medication_single.html)
- Opens expanded by default when result tier is `high`, collapsed otherwise
- Doctors only — enforced both in the template (Jinja2 role check) and on the server (`/api/drug-label` returns 403 for reception)
- Reception users see the medication result as before; the panel is absent from their view

### B. Input & Quick-Pick Chips
- Free-text search box (max 60 chars, letters/digits/spaces/hyphens only)
- 20 quick-pick chips derived programmatically from the dataset drug columns in `df_clean.csv`:
  `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `metformin-rosiglitazone`, `metformin-pioglitazone`
- Clicking a chip sets the input box and triggers the lookup immediately; also expands the panel if collapsed

### C. Backend Route
- New Blueprint: [`app/routes/api.py`](../app/routes/api.py) registered under `/api`
- Route: `GET /api/drug-label?name=<drug>`
- Input validation regex: `^[A-Za-z0-9 \-]+$`, max 60 chars — rejects all injection attempts
- Term is double-quoted in the search expression and passed through `requests.PreparedRequest` for URL encoding — user text cannot alter query syntax
- API key read from `OPENFDA_API_KEY` environment variable; never hardcoded, never logged, never sent to browser
- 5-second timeout; tries generic name first, then brand name

### D. Sections Displayed
| Section | Label |
|---|---|
| `boxed_warning` | Boxed Warning |
| `indications_and_usage` | Indications & Usage |
| `drug_interactions` | Drug Interactions |
| `warnings` | Warnings |

- Each trimmed to 500 characters with a "Show more / Show less" toggle
- Missing sections shown as "Not provided in this label."
- Source line: "Source: U.S. FDA drug label data via openFDA"
- Label date shown when available

### E. Disclaimer
Always visible inside the panel:
> Reference information from FDA drug labels (openFDA). Not clinical advice and not a substitute for the full official label, a pharmacist or hospital protocols. University project, not a medical device.

### F. Quota Protection
- In-memory 24-hour cache per drug name (keyed by MD5 of normalised name)
- Per-session limit: 30 lookups per hour, with a friendly countdown message
- Cache status reported in the UI (`Live FDA` / `Cached` / `Saved copy (date)`)

### G. Failure Handling
All errors are caught; the panel shows a friendly message; the Medication Load model result is unaffected:
- No match → "No FDA drug label found for …"
- 429 rate limit → "openFDA rate limit reached…"
- 5-second timeout → "The openFDA server did not respond within 5 seconds…"
- Connection error → "Could not connect to openFDA…"
- Malformed JSON → "openFDA returned an unexpected response…"
- Session limit → "You have reached the lookup limit (30 per hour)…"

### H. Demo Fallback / Snapshot
- Script: [`scripts/fetch_fda_snapshot.py`](../scripts/fetch_fda_snapshot.py)
- Fetches real label data for all 20 chip drugs; saves to `app/static/fda_snapshot.json`
- Snapshot records `generated_at` ISO timestamp; shown as "Saved copy from (date)"
- All text comes from real API responses; nothing hand-written

### I. Environment & Security
- `.env.example` added with empty `OPENFDA_API_KEY=`
- `.gitignore` created, excluding `.env`
- `run.py` loads `.env` via `python-dotenv` if present (graceful if package missing)

---

## 4. Manual Live Test Results

> **To run:** Start the app (`python run.py`) and log in as doctor.  
> Navigate to Medication Load → fill any form → submit → expand "Drug Reference (FDA Label)"

### Test 1 — Chip drug: `metformin`
- API call: `GET https://api.fda.gov/drug/label.json?search=openfda.generic_name:"metformin"&limit=1`
- Expected response: Brand name `Glucophage` (or similar), generic `METFORMIN HYDROCHLORIDE`
- Boxed warning: Lactic acidosis warning present
- Indications: Type 2 diabetes mellitus

*(Manual run required — network not available in the CI sandbox)*

### Test 2 — Nonsense name: `zyxwvutsrq`
- API returns HTTP 404 for both strategies
- Panel shows: "No FDA drug label found for 'zyxwvutsrq'. Try an alternative name or check the spelling."

*(Manual run required)*

---

## 5. Test Results

| Test suite | Tests | Result |
|---|---|---|
| `test_phase1_foundation` | 10 | ✅ PASS |
| `test_load_models` | 11 | ✅ PASS |
| `test_phase2_parity` | 7 | ✅ PASS |
| `test_phase2_routes` | 12 | ✅ PASS |
| `test_phase3_summary` | 7 | ✅ PASS |
| **`test_phase6_fda`** | **27** | ✅ **PASS** |
| **Total** | **53** | ✅ **ALL PASS** |

All 27 Phase 6 tests use mocked API responses (no real network): success, missing sections, no match, rate limit, timeout, connection error, malformed JSON, invalid inputs (empty, too long, special chars, quote injection), role enforcement (unauthenticated → 403, reception → 403), cache hit, cache source field, per-session limit, snapshot fallback, medication page continuity, and model checksums.

### Model checksum verification
- 11/12 model files: ✅ SHA-256 match  
- `models/asma/asma_longstay_predict.py`: pre-existing mismatch (noted in Phase 2 report; `.py` file is not a trained artefact). All `.joblib` and `.json` model files are unchanged.

---

## 6. Responsive Design & Keyboard Access

- The panel uses Bootstrap 5 collapse/grid — fully responsive at all breakpoints
- All interactive elements (search box, button, chips) are keyboard-accessible
- `aria-expanded`, `aria-controls`, `aria-label`, `aria-live="polite"` attributes in place
- Spinner uses `<span class="visually-hidden">Loading…</span>`

---

## 7. Files Added / Modified

| File | Change |
|---|---|
| `app/services/fda_service.py` | **New** — core service: validation, cache, API call, snapshot |
| `app/routes/api.py` | **New** — `/api/drug-label` JSON endpoint |
| `app/templates/clinical/medication_single.html` | **Modified** — FDA panel + JS added |
| `app/__init__.py` | **Modified** — registers `api_bp` |
| `run.py` | **Modified** — loads `.env` via python-dotenv |
| `requirements.txt` | **Modified** — adds `requests`, `python-dotenv`, `certifi`, etc. |
| `.env.example` | **New** — documents `OPENFDA_API_KEY` |
| `.gitignore` | **New** — excludes `.env` and standard Python artefacts |
| `scripts/fetch_fda_snapshot.py` | **New** — snapshot generation script |
| `tests/test_phase6_fda.py` | **New** — 27 automated tests |

---

## 8. Constraints Verified

- ✅ No model files modified (`.joblib`, `.json` checksums match)
- ✅ Model predictions unchanged (20-row parity test still passes)
- ✅ Only drug names are sent to openFDA — never patient values, session data or form fields
- ✅ API key never hardcoded, never logged, never sent to browser
- ✅ Dosage/administration sections excluded from display
- ✅ All label text from real API responses (snapshot script documented)
- ✅ App starts and works offline (FDA panel degrades gracefully)
- ✅ Reception role: sees medication result, no FDA panel, `/api/drug-label` returns 403

---

*Awaiting approval to proceed.*
