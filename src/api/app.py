import os
from flask import Flask
from flask_cors import CORS

from src.api.routes import api_bp, init_pipeline
from src.api.middleware import (
    log_request_middleware,
    log_response_middleware,
    handle_error,
    handle_404,
    handle_405,
    handle_500
)
from src.database.connection import db_connection
from src.utils.logger import setup_logger
from src.utils.config import config

logger = setup_logger('flask_app')

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)

    # Load configuration
    app.config['MAX_CONTENT_LENGTH'] = config.get_max_file_size()
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

    # Setup CORS
    cors_origins = config.get('api.cors_origins', ['http://localhost:8501'])
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})

    # Initialize database
    try:
        db_connection.initialize()
        db_connection.create_tables()  # Create tables if they don't exist
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Initialize ML pipeline
    try:
        init_pipeline()
        logger.info("ML pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML pipeline: {e}")
        raise

    # Register blueprints
    app.register_blueprint(api_bp)

    # Register middleware
    @app.before_request
    def before_request():
        log_request_middleware()

    @app.after_request
    def after_request(response):
        return log_response_middleware(response)

    # Register error handlers
    app.register_error_handler(404, handle_404)
    app.register_error_handler(405, handle_405)
    app.register_error_handler(500, handle_500)
    app.register_error_handler(Exception, handle_error)

    logger.info("Flask application created successfully")

    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 5000)
    debug = config.get('api.debug', True)

    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
