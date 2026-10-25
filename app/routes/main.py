from flask import Blueprint, render_template, redirect, url_for, session, request
from app.auth import login_required
import app.services.summary_service as summary_svc

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    """Root route: redirect to dashboard if logged in, else to login."""
    if "user_email" in session:
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("auth.login"))

@main_bp.route("/dashboard")
@login_required
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard/index.html")

@main_bp.route("/patient-summary", methods=["GET", "POST"])
@main_bp.route("/patient-report", methods=["GET", "POST"])
@login_required
def patient_summary():
    """Unified patient report — runs all applicable models simultaneously."""
    role = session.get("user_role", "reception")

    ctx = summary_svc.get_form_context()

    if request.method == "POST":
        form = request.form.to_dict()
        results = summary_svc.run_all(form, role=role)
        ctx["results"] = results
        ctx["d"] = form          # repopulate form fields

    return render_template("patient_summary/index.html", **ctx)

@main_bp.route("/terms")
def terms():
    """Terms of service."""
    return render_template("legal/terms.html")
