from sqlalchemy import desc, func
from datetime import datetime

from src.database.models import Detection, VehicleDetection, PlateDetection
from src.database.connection import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DetectionRepository:
    """Data access layer for detection operations."""

    def create_detection(self, filename, image_path, file_size=None, image_width=None, image_height=None):
        """
        Create new detection record.

        Args:
            filename: Original filename
            image_path: Path to uploaded image
            file_size: File size in bytes
            image_width: Image width
            image_height: Image height

        Returns:
            Detection model instance
        """
        try:
            with db_connection.session_scope() as session:
                detection = Detection(
                    original_filename=filename,
                    uploaded_image_path=image_path,
                    status='pending',
                    file_size_bytes=file_size,
                    image_width=image_width,
                    image_height=image_height
                )
                session.add(detection)
                session.flush()
                detection_id = detection.id

            logger.info(f"Created detection record with ID: {detection_id}")
            return detection_id

        except Exception as e:
            logger.error(f"Error creating detection: {e}")
            raise

    def update_detection_status(self, detection_id, status, error_message=None):
        """Update detection processing status."""
        try:
            with db_connection.session_scope() as session:
                detection = session.query(Detection).filter_by(id=detection_id).first()
                if detection:
                    detection.status = status
                    detection.error_message = error_message
                    detection.updated_at = datetime.utcnow()
                    logger.info(f"Updated detection {detection_id} status to: {status}")
                    return True
                else:
                    logger.warning(f"Detection {detection_id} not found")
                    return False

        except Exception as e:
            logger.error(f"Error updating detection status: {e}")
            raise

    def update_detection_results(self, detection_id, result_data):
        """
        Update detection with processing results.

        Args:
            detection_id: Detection ID
            result_data: Dictionary containing processing results
        """
        try:
            with db_connection.session_scope() as session:
                detection = session.query(Detection).filter_by(id=detection_id).first()
                if detection:
                    detection.status = result_data.get('status', 'completed')
                    detection.processed_image_path = result_data.get('processed_image_path')
                    detection.vehicles_detected = result_data.get('vehicles_detected', 0)
                    detection.plates_detected = result_data.get('plates_detected', 0)
                    detection.processing_time_seconds = result_data.get('processing_time')
                    detection.error_message = result_data.get('error_message')
                    detection.image_width = result_data.get('image_width')
                    detection.image_height = result_data.get('image_height')
                    detection.updated_at = datetime.utcnow()

                    logger.info(f"Updated detection {detection_id} with results")
                    return True
                else:
                    logger.warning(f"Detection {detection_id} not found")
                    return False

        except Exception as e:
            logger.error(f"Error updating detection results: {e}")
            raise

    def save_vehicle_detection(self, detection_id, vehicle_data):
        """
        Save vehicle detection.

        Args:
            detection_id: Parent detection ID
            vehicle_data: Dictionary containing vehicle detection data

        Returns:
            Vehicle detection ID
        """
        try:
            with db_connection.session_scope() as session:
                bbox = vehicle_data['bbox']
                vehicle = VehicleDetection(
                    detection_id=detection_id,
                    bbox_x1=int(bbox[0]),
                    bbox_y1=int(bbox[1]),
                    bbox_x2=int(bbox[2]),
                    bbox_y2=int(bbox[3]),
                    confidence=vehicle_data['confidence'],
                    vehicle_class=vehicle_data.get('class_name')
                )
                session.add(vehicle)
                session.flush()
                vehicle_id = vehicle.id

            logger.info(f"Saved vehicle detection with ID: {vehicle_id}")
            return vehicle_id

        except Exception as e:
            logger.error(f"Error saving vehicle detection: {e}")
            raise

    def save_plate_detection(self, detection_id, vehicle_id, plate_data):
        """
        Save plate detection.

        Args:
            detection_id: Parent detection ID
            vehicle_id: Parent vehicle detection ID
            plate_data: Dictionary containing plate detection data

        Returns:
            Plate detection ID
        """
        try:
            with db_connection.session_scope() as session:
                bbox = plate_data['bbox']
                plate = PlateDetection(
                    detection_id=detection_id,
                    vehicle_detection_id=vehicle_id,
                    bbox_x1=int(bbox[0]),
                    bbox_y1=int(bbox[1]),
                    bbox_x2=int(bbox[2]),
                    bbox_y2=int(bbox[3]),
                    detection_confidence=plate_data['confidence'],
                    license_plate_text=plate_data.get('text'),
                    ocr_confidence=plate_data.get('ocr_confidence')
                )
                session.add(plate)
                session.flush()
                plate_id = plate.id

            logger.info(f"Saved plate detection with ID: {plate_id}")
            return plate_id

        except Exception as e:
            logger.error(f"Error saving plate detection: {e}")
            raise

    def get_detection_by_id(self, detection_id):
        """
        Get complete detection by ID with all relationships.

        Returns:
            Dictionary with complete detection data
        """
        try:
            with db_connection.session_scope() as session:
                detection = session.query(Detection).filter_by(id=detection_id).first()

                if not detection:
                    logger.warning(f"Detection {detection_id} not found")
                    return None

                # Build result dictionary
                result = detection.to_dict()

                # Add vehicles with plates
                vehicles = []
                for vehicle in detection.vehicles:
                    vehicle_dict = vehicle.to_dict()

                    # Add plates for this vehicle
                    plates = []
                    for plate in vehicle.plates:
                        plates.append(plate.to_dict())

                    vehicle_dict['plates'] = plates
                    vehicles.append(vehicle_dict)

                result['vehicles'] = vehicles

                return result

        except Exception as e:
            logger.error(f"Error getting detection by ID: {e}")
            raise

    def list_detections(self, page=1, per_page=20, status=None, sort_by='created_at', order='desc'):
        """
        Get paginated list of detections.

        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            status: Filter by status (optional)
            sort_by: Sort field
            order: Sort order ('asc' or 'desc')

        Returns:
            Dictionary with detections and pagination info
        """
        try:
            with db_connection.session_scope() as session:
                query = session.query(Detection)

                # Apply status filter
                if status:
                    query = query.filter(Detection.status == status)

                # Apply sorting
                sort_field = getattr(Detection, sort_by, Detection.created_at)
                if order == 'desc':
                    query = query.order_by(desc(sort_field))
                else:
                    query = query.order_by(sort_field)

                # Get total count
                total_items = query.count()

                # Apply pagination
                offset = (page - 1) * per_page
                detections = query.limit(per_page).offset(offset).all()

                # Convert to dictionaries
                detection_list = [det.to_dict() for det in detections]

                # Calculate pagination info
                total_pages = (total_items + per_page - 1) // per_page

                return {
                    'detections': detection_list,
                    'pagination': {
                        'page': page,
                        'per_page': per_page,
                        'total_pages': total_pages,
                        'total_items': total_items
                    }
                }

        except Exception as e:
            logger.error(f"Error listing detections: {e}")
            raise

    def delete_detection(self, detection_id):
        """Delete detection and all related records (cascade)."""
        try:
            with db_connection.session_scope() as session:
                detection = session.query(Detection).filter_by(id=detection_id).first()

                if not detection:
                    logger.warning(f"Detection {detection_id} not found")
                    return False

                session.delete(detection)
                logger.info(f"Deleted detection {detection_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting detection: {e}")
            raise

    def get_statistics(self):
        """Get overall detection statistics."""
        try:
            with db_connection.session_scope() as session:
                total_detections = session.query(func.count(Detection.id)).scalar() or 0
                total_vehicles = session.query(func.count(VehicleDetection.id)).scalar() or 0
                total_plates = session.query(func.count(PlateDetection.id)).scalar() or 0

                # Average processing time
                avg_time = session.query(func.avg(Detection.processing_time_seconds))\
                    .filter(Detection.processing_time_seconds.isnot(None)).scalar() or 0.0

                # Success rate
                completed = session.query(func.count(Detection.id))\
                    .filter(Detection.status == 'completed').scalar() or 0
                success_rate = completed / total_detections if total_detections > 0 else 0.0

                return {
                    'total_detections': total_detections,
                    'total_vehicles_detected': total_vehicles,
                    'total_plates_detected': total_plates,
                    'average_processing_time': round(avg_time, 2),
                    'success_rate': round(success_rate, 2)
                }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise

# Global repository instance
detection_repository = DetectionRepository()
