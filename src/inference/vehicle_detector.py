import cv2
from ultralytics import YOLO

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

class VehicleDetector:
    """YOLOv8 vehicle detection."""

    # COCO dataset class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, model_path=None, confidence_threshold=0.5, device='cpu'):
        """Initialize vehicle detector with YOLOv8 model."""
        if model_path is None:
            model_path = config.get_model_path('vehicle_detector')

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None

        self.load_model()

    def load_model(self):
        """Load YOLOv8 model."""
        try:
            logger.info(f"Loading vehicle detection model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Vehicle detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vehicle detection model: {e}")
            raise

    def detect(self, image_path):
        """
        Detect vehicles in image.

        Args:
            image_path: Path to input image

        Returns:
            List of dictionaries containing vehicle detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                },
                ...
            ]
        """
        try:
            logger.info(f"Running vehicle detection on {image_path}")

            # Run inference
            results = self.model(image_path, device=self.device)

            # Extract detections
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls.cpu().numpy()[0])

                    # Filter for vehicle classes only
                    if class_id in self.VEHICLE_CLASSES:
                        confidence = float(box.conf.cpu().numpy()[0])

                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            bbox = box.xyxy.cpu().numpy()[0].tolist()

                            detection = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.VEHICLE_CLASSES[class_id]
                            }
                            detections.append(detection)

            logger.info(f"Detected {len(detections)} vehicles")
            return detections

        except Exception as e:
            logger.error(f"Error during vehicle detection: {e}")
            raise

    def extract_vehicle_regions(self, image, detections):
        """
        Extract vehicle regions from image based on detections.

        Args:
            image: Input image (numpy array)
            detections: List of vehicle detections

        Returns:
            List of tuples (vehicle_image, detection_dict)
        """
        vehicle_regions = []

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            vehicle_img = image[y1:y2, x1:x2]
            vehicle_regions.append((vehicle_img, detection))

        return vehicle_regions
