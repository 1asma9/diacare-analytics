from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, abort, current_app
from werkzeug.security import check_password_hash

auth_bp = Blueprint("auth", __name__)

def get_current_user():
    """Retrieve currently logged in user info from session, or None."""
    email = session.get("user_email")
    if not email:
        return None
    users = current_app.config.get("DEMO_USERS", {})
    user_info = users.get(email)
    if not user_info:
        # Fallback to session details if email not in config dict
        return {
            "email": email,
            "name": session.get("user_name", "User"),
            "role": session.get("user_role", "guest"),
            "role_title": session.get("user_role_title", "Staff Member"),
            "department": session.get("user_department", "Hospital Staff")
        }
    return user_info

def login_required(f):
    """Ensure user is logged in before accessing the view."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_email" not in session:
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for("auth.login", next=request.full_path if request.query_string else request.path))
        return f(*args, **kwargs)
    return decorated_function

def role_required(allowed_roles):
    """Ensure current user has one of the required roles, returning 403 Forbidden if not."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_email" not in session:
                flash("Please sign in to access this page.", "warning")
                return redirect(url_for("auth.login", next=request.full_path if request.query_string else request.path))
            user_role = session.get("user_role")
            if user_role not in allowed_roles:
                # Return 403 Forbidden with access denied page
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Login view supporting both demo accounts with clear instructions."""
    if "user_email" in session:
        return redirect(url_for("main.dashboard"))
    
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        
        users = current_app.config.get("DEMO_USERS", {})
        user = users.get(email)
        
        if user and check_password_hash(user["password_hash"], password):
            # Session setup
            session.clear()
            session["user_email"] = user["email"]
            session["user_name"] = user["name"]
            session["user_role"] = user["role"]
            session["user_role_title"] = user["role_title"]
            session["user_department"] = user["department"]
            session.permanent = True
            
            flash(f"Welcome back, {user['name']}! Signed in as {user['role_title']}.", "success")
            
            next_page = request.args.get("next")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("main.dashboard"))
        else:
            error = "Invalid credentials. Please select one of the demo accounts below."
    
    demo_accounts = current_app.config.get("DEMO_USERS", {})
    return render_template("auth/login.html", error=error, demo_accounts=demo_accounts)

@auth_bp.route("/logout")
def logout():
    """Sign out current user and clear session."""
    user_name = session.get("user_name", "User")
    session.clear()
    flash(f"Goodbye, {user_name}. You have been securely signed out.", "info")
    return redirect(url_for("auth.login"))
