"""Admission and planning routes for DiaCare Analytics.
Tools:
- Asma: Length of Stay (>3 Days) (Doctors and Reception)
- Dhie: Discharge Destination Prediction (Doctors and Reception)
"""
import io
from flask import Blueprint, current_app, render_template, request, send_file, flash, redirect, url_for
from app.auth import role_required
from app.services import asma_service, dhie_service

admission_bp = Blueprint("admission", __name__)


# ── 1. Asma: Length of Stay Prediction ────────────────────────────────────────

@admission_bp.route("/los", methods=["GET", "POST"])
@role_required(["doctor", "reception"])
def los_single():
    """Length of stay prediction - Single patient (Asma)."""
    version = request.args.get("version", "admission")
    if request.method == "POST":
        version = request.form.get("version", "admission")
    
    ctx = asma_service.get_form_context(version=version)
    result = None

    if request.method == "POST":
        row, errors = asma_service.clean(request.form, version=version)
        if errors:
            ctx = asma_service.get_form_context(form_data=request.form.to_dict(), errors=errors, version=version)
            return render_template("admission/los_single.html", **ctx, result=None), 400

        result = asma_service.predict_single(row, version=version)
        ctx = asma_service.get_form_context(form_data=row, version=version)

    return render_template("admission/los_single.html", **ctx, result=result)


@admission_bp.route("/los/batch", methods=["GET", "POST"])
@role_required(["doctor", "reception"])
def los_batch():
    """Length of stay prediction - Several patients (Asma)."""
    version = request.args.get("version", "admission")
    cols = asma_service.CSV_COLUMNS_AFTER_STAY if version == "after_stay" else asma_service.CSV_COLUMNS_ADMISSION
    if request.method == "GET":
        return render_template("admission/los_batch.html", columns=cols, result=None, error=None, version=version)

    version = request.form.get("version", "admission")
    cols = asma_service.CSV_COLUMNS_AFTER_STAY if version == "after_stay" else asma_service.CSV_COLUMNS_ADMISSION
    file_obj = request.files.get("file")
    if not file_obj or not file_obj.filename.lower().endswith(".csv"):
        return render_template("admission/los_batch.html", columns=cols, result=None,
                               error="Please select a valid CSV file (.csv).", version=version), 400

    res, err = asma_service.process_batch_file(file_obj, version=version)
    if err:
        return render_template("admission/los_batch.html", columns=cols, result=None, error=err, version=version), 400

    current_app.config["LAST_BATCH_ASMA"] = res["dataframe"]
    return render_template("admission/los_batch.html", columns=cols, result=res, error=None, version=version)


@admission_bp.route("/los/batch/template")
@role_required(["doctor", "reception"])
def los_batch_template():
    """Download sample CSV template for Asma's model."""
    version = request.args.get("version", "admission")
    buf = asma_service.generate_template_csv(version=version)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"patients_los_{version}_template.csv")


@admission_bp.route("/los/batch/download")
@role_required(["doctor", "reception"])
def los_batch_download():
    """Download scored CSV for Asma's model."""
    df = current_app.config.get("LAST_BATCH_ASMA")
    if df is None:
        flash("No batch has been processed yet to download.", "warning")
        return redirect(url_for("admission.los_batch"))
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="los_predictions.csv")


# ── 2. Dhie: Discharge Destination Prediction ─────────────────────────────────

@admission_bp.route("/discharge", methods=["GET", "POST"])
@role_required(["doctor", "reception"])
def discharge_single():
    """Discharge destination prediction - Single patient (Dhie)."""
    ctx = dhie_service.get_form_context()
    result = None

    if request.method == "POST":
        row, errors = dhie_service.clean(request.form)
        if errors:
            ctx = dhie_service.get_form_context(form_data=request.form.to_dict(), errors=errors)
            return render_template("admission/discharge_single.html", **ctx, result=None), 400

        result = dhie_service.predict_single(row)
        ctx = dhie_service.get_form_context(form_data=row)

    return render_template("admission/discharge_single.html", **ctx, result=result)


@admission_bp.route("/discharge/batch", methods=["GET", "POST"])
@role_required(["doctor", "reception"])
def discharge_batch():
    """Discharge destination prediction - Several patients (Dhie)."""
    columns = dhie_service.CSV_COLUMNS
    if request.method == "GET":
        return render_template("admission/discharge_batch.html", columns=columns, result=None, error=None)

    file_obj = request.files.get("file")
    if not file_obj or not file_obj.filename.lower().endswith(".csv"):
        return render_template("admission/discharge_batch.html", columns=columns, result=None,
                               error="Please select a valid CSV file (.csv)."), 400

    res, err = dhie_service.process_batch_file(file_obj)
    if err:
        return render_template("admission/discharge_batch.html", columns=columns, result=None, error=err), 400

    current_app.config["LAST_BATCH_DHIE"] = res["dataframe"]
    return render_template("admission/discharge_batch.html", columns=columns, result=res, error=None)


@admission_bp.route("/discharge/batch/template")
@role_required(["doctor", "reception"])
def discharge_batch_template():
    """Download sample CSV template for Dhie's model."""
    buf = dhie_service.generate_template_csv()
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="patients_discharge_template.csv")


@admission_bp.route("/discharge/batch/download")
@role_required(["doctor", "reception"])
def discharge_batch_download():
    """Download scored CSV for Dhie's model."""
    df = current_app.config.get("LAST_BATCH_DHIE")
    if df is None:
        flash("No batch has been processed yet to download.", "warning")
        return redirect(url_for("admission.discharge_batch"))
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="discharge_predictions.csv")
