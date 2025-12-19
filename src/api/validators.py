from werkzeug.datastructures import FileStorage

from src.utils.config import config
from src.utils.file_handler import validate_file_extension, validate_file_size

def validate_image_file(file):
    """
    Validate uploaded image file.

    Args:
        file: FileStorage object

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"

    if not isinstance(file, FileStorage):
        return False, "Invalid file object"

    if file.filename == '':
        return False, "No file selected"

    # Validate extension
    allowed_extensions = config.get_allowed_extensions()
    if not validate_file_extension(file.filename, allowed_extensions):
        return False, f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"

    # Validate file size
    max_size = config.get_max_file_size()
    if not validate_file_size(file, max_size):
        max_mb = max_size / (1024 * 1024)
        return False, f"File size exceeds maximum allowed size of {max_mb:.1f}MB"

    return True, None

def validate_detection_id(detection_id):
    """
    Validate detection ID parameter.

    Args:
        detection_id: Detection ID (string or int)

    Returns:
        Tuple of (is_valid, error_message, parsed_id)
    """
    try:
        parsed_id = int(detection_id)
        if parsed_id <= 0:
            return False, "Detection ID must be a positive integer", None
        return True, None, parsed_id
    except (ValueError, TypeError):
        return False, "Invalid detection ID format", None

def validate_pagination_params(page, per_page):
    """
    Validate pagination parameters.

    Args:
        page: Page number (string or int)
        per_page: Items per page (string or int)

    Returns:
        Tuple of (is_valid, error_message, parsed_page, parsed_per_page)
    """
    try:
        parsed_page = int(page) if page else 1
        parsed_per_page = int(per_page) if per_page else 20

        if parsed_page <= 0:
            return False, "Page number must be a positive integer", None, None

        if parsed_per_page <= 0 or parsed_per_page > 100:
            return False, "Per page value must be between 1 and 100", None, None

        return True, None, parsed_page, parsed_per_page

    except (ValueError, TypeError):
        return False, "Invalid pagination parameters", None, None

def validate_sort_params(sort_by, order):
    """
    Validate sort parameters.

    Args:
        sort_by: Field to sort by
        order: Sort order

    Returns:
        Tuple of (is_valid, error_message, parsed_sort_by, parsed_order)
    """
    allowed_sort_fields = ['created_at', 'updated_at', 'status', 'vehicles_detected', 'plates_detected']
    allowed_orders = ['asc', 'desc']

    parsed_sort_by = sort_by if sort_by else 'created_at'
    parsed_order = order if order else 'desc'

    if parsed_sort_by not in allowed_sort_fields:
        return False, f"Invalid sort field. Allowed: {', '.join(allowed_sort_fields)}", None, None

    if parsed_order not in allowed_orders:
        return False, f"Invalid sort order. Allowed: {', '.join(allowed_orders)}", None, None

    return True, None, parsed_sort_by, parsed_order
