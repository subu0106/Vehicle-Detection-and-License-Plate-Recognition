import time
from flask import request, jsonify
from functools import wraps

from src.utils.logger import get_logger

logger = get_logger(__name__)

def format_response(status='success', data=None, message=None, status_code=200):
    """Standardize API response format."""
    response = {
        'status': status
    }

    if message:
        response['message'] = message

    if data is not None:
        response['data'] = data

    return jsonify(response), status_code

def format_error_response(message, status_code=400):
    """Format error response."""
    return format_response(
        status='error',
        message=message,
        status_code=status_code
    )

def log_request_middleware():
    """Log incoming requests."""
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")
    request.start_time = time.time()

def log_response_middleware(response):
    """Log outgoing responses."""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(
            f"{request.method} {request.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.3f}s"
        )
    return response

def handle_error(error):
    """Handle application errors."""
    logger.error(f"Application error: {str(error)}")

    if hasattr(error, 'code'):
        status_code = error.code
    else:
        status_code = 500

    return format_error_response(
        message=str(error),
        status_code=status_code
    )

def handle_404(error):
    """Handle 404 errors."""
    return format_error_response(
        message="Resource not found",
        status_code=404
    )

def handle_405(error):
    """Handle 405 errors."""
    return format_error_response(
        message="Method not allowed",
        status_code=405
    )

def handle_500(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return format_error_response(
        message="Internal server error",
        status_code=500
    )

def require_file(f):
    """Decorator to require file in request."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'file' not in request.files:
            return format_error_response("No file part in request", 400)
        return f(*args, **kwargs)
    return decorated_function
