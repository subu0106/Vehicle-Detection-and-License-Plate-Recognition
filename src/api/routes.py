from flask import Blueprint, request

from src.api.middleware import format_response, format_error_response, require_file
from src.api.validators import (
    validate_image_file,
    validate_detection_id,
    validate_pagination_params,
    validate_sort_params
)
from src.database.repository import detection_repository
from src.inference.pipeline import DetectionPipeline
from src.utils.file_handler import save_uploaded_file, delete_detection_files, get_file_size
from src.utils.image_processor import get_image_dimensions
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize ML pipeline (will be done in app.py)
pipeline = None

def init_pipeline():
    """Initialize detection pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = DetectionPipeline()

@api_bp.route('/upload', methods=['POST'])
@require_file
def upload_image():
    """
    Upload an image for vehicle and license plate detection.

    Returns:
        JSON response with detection ID and status
    """
    try:
        file = request.files['file']

        # Validate file
        is_valid, error_message = validate_image_file(file)
        if not is_valid:
            return format_error_response(error_message, 400)

        # Save uploaded file
        upload_folder = config.get_upload_folder()
        uploaded_path = save_uploaded_file(file, upload_folder)

        # Get file metadata
        file_size = get_file_size(uploaded_path)
        image_width, image_height = get_image_dimensions(uploaded_path)

        # Create detection record
        detection_id = detection_repository.create_detection(
            filename=file.filename,
            image_path=uploaded_path,
            file_size=file_size,
            image_width=image_width,
            image_height=image_height
        )

        # Update status to processing
        detection_repository.update_detection_status(detection_id, 'processing')

        # Process image through ML pipeline
        result = pipeline.process(uploaded_path, save_visualization=True)

        # Update detection with results
        detection_repository.update_detection_results(detection_id, result)

        # Save vehicle and plate detections
        for vehicle in result.get('vehicles', []):
            vehicle_id = detection_repository.save_vehicle_detection(detection_id, vehicle)

            for plate in vehicle.get('plates', []):
                detection_repository.save_plate_detection(detection_id, vehicle_id, plate)

        logger.info(f"Successfully processed detection {detection_id}")

        return format_response(
            status='success',
            message='Image uploaded and processed successfully',
            data={
                'detection_id': detection_id,
                'filename': file.filename,
                'processing_status': result['status'],
                'vehicles_detected': result['vehicles_detected'],
                'plates_detected': result['plates_detected']
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return format_error_response(f"Error processing image: {str(e)}", 500)

@api_bp.route('/detection/<detection_id>', methods=['GET'])
def get_detection(detection_id):
    """
    Get detection results by ID.

    Args:
        detection_id: Detection ID

    Returns:
        JSON response with complete detection data
    """
    try:
        # Validate detection ID
        is_valid, error_message, parsed_id = validate_detection_id(detection_id)
        if not is_valid:
            return format_error_response(error_message, 400)

        # Get detection from database
        detection = detection_repository.get_detection_by_id(parsed_id)

        if not detection:
            return format_error_response(f"Detection {parsed_id} not found", 404)

        return format_response(
            status='success',
            data=detection,
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error getting detection: {e}")
        return format_error_response(f"Error retrieving detection: {str(e)}", 500)

@api_bp.route('/detections', methods=['GET'])
def list_detections():
    """
    Get paginated list of all detections.

    Query Parameters:
        page: Page number (default: 1)
        per_page: Items per page (default: 20)
        status: Filter by status (optional)
        sort_by: Sort field (default: created_at)
        order: Sort order (default: desc)

    Returns:
        JSON response with list of detections and pagination info
    """
    try:
        # Get query parameters
        page = request.args.get('page', 1)
        per_page = request.args.get('per_page', 20)
        status = request.args.get('status')
        sort_by = request.args.get('sort_by', 'created_at')
        order = request.args.get('order', 'desc')

        # Validate pagination
        is_valid, error_message, parsed_page, parsed_per_page = validate_pagination_params(page, per_page)
        if not is_valid:
            return format_error_response(error_message, 400)

        # Validate sort parameters
        is_valid, error_message, parsed_sort_by, parsed_order = validate_sort_params(sort_by, order)
        if not is_valid:
            return format_error_response(error_message, 400)

        # Get detections from database
        result = detection_repository.list_detections(
            page=parsed_page,
            per_page=parsed_per_page,
            status=status,
            sort_by=parsed_sort_by,
            order=parsed_order
        )

        return format_response(
            status='success',
            data=result,
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error listing detections: {e}")
        return format_error_response(f"Error retrieving detections: {str(e)}", 500)

@api_bp.route('/detection/<detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """
    Delete a detection record and associated files.

    Args:
        detection_id: Detection ID

    Returns:
        JSON response with success status
    """
    try:
        # Validate detection ID
        is_valid, error_message, parsed_id = validate_detection_id(detection_id)
        if not is_valid:
            return format_error_response(error_message, 400)

        # Get detection to retrieve file paths
        detection = detection_repository.get_detection_by_id(parsed_id)

        if not detection:
            return format_error_response(f"Detection {parsed_id} not found", 404)

        # Delete files
        delete_detection_files(
            detection.get('uploaded_image_path'),
            detection.get('processed_image_path')
        )

        # Delete from database
        success = detection_repository.delete_detection(parsed_id)

        if success:
            return format_response(
                status='success',
                message=f"Detection {parsed_id} deleted successfully",
                status_code=200
            )
        else:
            return format_error_response(f"Failed to delete detection {parsed_id}", 500)

    except Exception as e:
        logger.error(f"Error deleting detection: {e}")
        return format_error_response(f"Error deleting detection: {str(e)}", 500)

@api_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get overall detection statistics.

    Returns:
        JSON response with statistics
    """
    try:
        stats = detection_repository.get_statistics()

        return format_response(
            status='success',
            data=stats,
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return format_error_response(f"Error retrieving statistics: {str(e)}", 500)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON response with health status
    """
    try:
        # Check if ML models are loaded
        models_loaded = pipeline is not None

        # Check database connection
        database_connected = True
        try:
            detection_repository.get_statistics()
        except:
            database_connected = False

        status = 'healthy' if (models_loaded and database_connected) else 'unhealthy'

        return format_response(
            status=status,
            data={
                'timestamp': None,  # Will be added by format_response
                'models_loaded': models_loaded,
                'database_connected': database_connected
            },
            status_code=200 if status == 'healthy' else 503
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return format_error_response(f"Health check failed: {str(e)}", 500)
