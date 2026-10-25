"""Clinical routes for DiaCare Analytics.
Tools:
- Seif: Readmission Risk Screening (Doctors only)
- Ons: HbA1c & Treatment Decision (Doctors only)
- Roua: Medication Load Assessment (Doctors and Reception)
- Moez: Unsupervised Patient Profiles (Doctors only)
"""
import io
from flask import Blueprint, current_app, render_template, request, send_file, flash, redirect, url_for
from app.auth import role_required
from app.services import seif_service, ons_service, roua_service, moez_service

clinical_bp = Blueprint("clinical", __name__)


# ── 1. Seif: Readmission Screening ───────────────────────────────────────────

@clinical_bp.route("/readmission", methods=["GET", "POST"])
@role_required(["doctor"])
def readmission_single():
    """Readmission screening - Single patient (Seif)."""
    ctx = seif_service.get_form_context()
    result = None

    if request.method == "POST":
        row, errors = seif_service.clean(request.form)
        if errors:
            ctx = seif_service.get_form_context(form_data=request.form.to_dict(), errors=errors)
            return render_template("clinical/readmission_single.html", **ctx, result=None), 400

        p, X = seif_service.score([row])
        p_val = float(p[0])
        b_val = seif_service.band(p_val)
        meta = seif_service.get_metadata()
        band_info = meta["bands"][b_val]

        result = {
            "p": p_val,
            "pct": round(p_val * 100, 1),
            "band": b_val,
            "thr": meta["thr"],
            "high": meta["high"],
            "base_rate": meta["base_rate"],
            "rate": band_info["rate"],
            "share": band_info["share"],
            "factors": seif_service.explain(X),
            "row": row,
            "shown": {k: seif_service.display(X.iloc[0][k], k) for k in (seif_service.LIMITS.keys() | {"age", "discharge", "admission", "diag_1", "insulin", "A1Cresult", "change", "diabetesMed"})}
        }
        ctx = seif_service.get_form_context(form_data=row)

    return render_template("clinical/readmission_single.html", **ctx, result=result)


@clinical_bp.route("/readmission/batch", methods=["GET", "POST"])
@role_required(["doctor"])
def readmission_batch():
    """Readmission screening - Several patients (Seif)."""
    columns = seif_service.CSV_COLUMNS
    if request.method == "GET":
        return render_template("clinical/readmission_batch.html", columns=columns, result=None, error=None)

    file_obj = request.files.get("file")
    if not file_obj or not file_obj.filename.lower().endswith(".csv"):
        return render_template("clinical/readmission_batch.html", columns=columns, result=None,
                               error="Please select a valid CSV file (.csv)."), 400

    res, err = seif_service.process_batch_file(file_obj)
    if err:
        return render_template("clinical/readmission_batch.html", columns=columns, result=None, error=err), 400

    current_app.config["LAST_BATCH_SEIF"] = res["dataframe"]
    return render_template("clinical/readmission_batch.html", columns=columns, result=res, error=None)


@clinical_bp.route("/readmission/batch/template")
@role_required(["doctor"])
def readmission_batch_template():
    """Download sample CSV template for Seif's model."""
    buf = seif_service.generate_template_csv()
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="patients_readmission_template.csv")


@clinical_bp.route("/readmission/batch/download")
@role_required(["doctor"])
def readmission_batch_download():
    """Download scored CSV for Seif's model."""
    df = current_app.config.get("LAST_BATCH_SEIF")
    if df is None:
        flash("No batch has been processed yet to download.", "warning")
        return redirect(url_for("clinical.readmission_batch"))
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="readmission_risk_scores.csv")


# ── 2. Ons: HbA1c & Treatment Decision ───────────────────────────────────────

@clinical_bp.route("/hba1c", methods=["GET", "POST"])
@role_required(["doctor"])
def hba1c_single():
    """HbA1c & treatment decision - Single patient (Ons)."""
    ctx = ons_service.get_form_context()
    result = None

    if request.method == "POST":
        row, errors = ons_service.clean(request.form)
        if errors:
            ctx = ons_service.get_form_context(form_data=request.form.to_dict(), errors=errors)
            return render_template("clinical/hba1c_single.html", **ctx, result=None), 400

        p, X = ons_service.score([row])
        p_val = float(p[0])
        b_val = ons_service.band(p_val)
        meta = ons_service.get_metadata()
        band_info = meta["bands"][b_val]

        result = {
            "p": p_val,
            "pct": round(p_val * 100, 1),
            "band": b_val,
            "thr": meta["thr"],
            "high": meta["high"],
            "base_rate": meta["base_rate"],
            "rate": band_info["rate"],
            "share": band_info["share"],
            "factors": ons_service.explain(X[0], row),
            "row": row,
            "shown": {k: ons_service.display(row[k], k) for k in ons_service.LABELS}
        }
        ctx = ons_service.get_form_context(form_data=row)

    return render_template("clinical/hba1c_single.html", **ctx, result=result)


@clinical_bp.route("/hba1c/batch", methods=["GET", "POST"])
@role_required(["doctor"])
def hba1c_batch():
    """HbA1c & treatment decision - Several patients (Ons)."""
    columns = ons_service.CSV_COLUMNS
    if request.method == "GET":
        return render_template("clinical/hba1c_batch.html", columns=columns, result=None, error=None)

    file_obj = request.files.get("file")
    if not file_obj or not file_obj.filename.lower().endswith(".csv"):
        return render_template("clinical/hba1c_batch.html", columns=columns, result=None,
                               error="Please select a valid CSV file (.csv)."), 400

    res, err = ons_service.process_batch_file(file_obj)
    if err:
        return render_template("clinical/hba1c_batch.html", columns=columns, result=None, error=err), 400

    current_app.config["LAST_BATCH_ONS"] = res["dataframe"]
    return render_template("clinical/hba1c_batch.html", columns=columns, result=res, error=None)


@clinical_bp.route("/hba1c/batch/template")
@role_required(["doctor"])
def hba1c_batch_template():
    """Download sample CSV template for Ons's model."""
    buf = ons_service.generate_template_csv()
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="patients_hba1c_template.csv")


@clinical_bp.route("/hba1c/batch/download")
@role_required(["doctor"])
def hba1c_batch_download():
    """Download scored CSV for Ons's model."""
    df = current_app.config.get("LAST_BATCH_ONS")
    if df is None:
        flash("No batch has been processed yet to download.", "warning")
        return redirect(url_for("clinical.hba1c_batch"))
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="hba1c_risk_scores.csv")


# ── 3. Roua: Medication Load Assessment ──────────────────────────────────────

@clinical_bp.route("/medication", methods=["GET", "POST"])
@role_required(["doctor", "reception"])
def medication_single():
    """Medication load assessment - Single patient (Roua)."""
    ctx = roua_service.get_form_context()
    result = None

    if request.method == "POST":
        row, errors = roua_service.clean(request.form)
        if errors:
            ctx = roua_service.get_form_context(form_data=request.form.to_dict(), errors=errors)
            return render_template("clinical/medication_single.html", **ctx, result=None), 400

        result = roua_service.predict_single(row)
        ctx = roua_service.get_form_context(form_data=row)

    return render_template("clinical/medication_single.html", **ctx, result=result)


@clinical_bp.route("/medication/batch", methods=["GET", "POST"])
@role_required(["doctor", "reception"])
def medication_batch():
    """Medication load assessment - Several patients (Roua)."""
    columns = roua_service.CSV_COLUMNS
    if request.method == "GET":
        return render_template("clinical/medication_batch.html", columns=columns, result=None, error=None)

    file_obj = request.files.get("file")
    if not file_obj or not file_obj.filename.lower().endswith(".csv"):
        return render_template("clinical/medication_batch.html", columns=columns, result=None,
                               error="Please select a valid CSV file (.csv)."), 400

    res, err = roua_service.process_batch_file(file_obj)
    if err:
        return render_template("clinical/medication_batch.html", columns=columns, result=None, error=err), 400

    current_app.config["LAST_BATCH_ROUA"] = res["dataframe"]
    return render_template("clinical/medication_batch.html", columns=columns, result=res, error=None)


@clinical_bp.route("/medication/batch/template")
@role_required(["doctor", "reception"])
def medication_batch_template():
    """Download sample CSV template for Roua's model."""
    buf = roua_service.generate_template_csv()
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="patients_medication_template.csv")


@clinical_bp.route("/medication/batch/download")
@role_required(["doctor", "reception"])
def medication_batch_download():
    """Download scored CSV for Roua's model."""
    df = current_app.config.get("LAST_BATCH_ROUA")
    if df is None:
        flash("No batch has been processed yet to download.", "warning")
        return redirect(url_for("clinical.medication_batch"))
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="medication_load_scores.csv")


# ── 4. Moez: Patient Profiles (Clustering) ───────────────────────────────────

@clinical_bp.route("/profiles")
@role_required(["doctor"])
def profiles_info():
    """Patient profiles - The two profiles (Moez)."""
    info = moez_service.get_all_profiles_info()
    return render_template("clinical/profiles_info.html", info=info)


@clinical_bp.route("/profiles/patient", methods=["GET", "POST"])
@role_required(["doctor"])
def profiles_patient():
    """Patient profiles - Where does this patient fit (Moez)."""
    ctx = moez_service.get_form_context()
    result = None

    if request.method == "POST":
        row, errors = moez_service.clean(request.form)
        if errors:
            ctx = moez_service.get_form_context(form_data=request.form.to_dict(), errors=errors)
            return render_template("clinical/profiles_patient.html", **ctx, result=None), 400

        result = moez_service.assign_patient(row)
        ctx = moez_service.get_form_context(form_data=row)

    return render_template("clinical/profiles_patient.html", **ctx, result=result)
