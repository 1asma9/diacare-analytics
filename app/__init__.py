import os
from flask import Flask, render_template
from app.config import Config
from app.auth import auth_bp, get_current_user
from app.routes.main import main_bp
from app.routes.admission import admission_bp
from app.routes.clinical import clinical_bp
from app.routes.api import api_bp

def create_app(config_class=Config):
    """Application factory for DiaCare Analytics."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="../assets",
        static_url_path="/assets"
    )
    app.config.from_object(config_class)

    # Inject current_user into every template context
    @app.context_processor
    def inject_context():
        return {
            "current_user": get_current_user()
        }

    # Error handlers
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template("errors/500.html"), 500

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admission_bp)
    app.register_blueprint(clinical_bp)
    app.register_blueprint(api_bp)          # Phase 6: JSON API (drug-label, etc.)

    return app
