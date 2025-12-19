import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

class PlateDetector:
    """Custom trained YOLO license plate detection."""

    def __init__(self, model_path=None, confidence_threshold=0.6, device='cpu'):
        """Initialize plate detector with custom trained YOLO model."""
        if model_path is None:
            model_path = config.get_model_path('plate_detector')

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None

        self.load_model()

    def load_model(self):
        """Load custom trained YOLO model for license plate detection."""
        try:
            logger.info(f"Loading license plate detection model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("License plate detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading license plate detection model: {e}")
            raise

    def detect(self, image):
        """
        Detect license plates in image or vehicle region.

        Args:
            image: Input image (can be numpy array or path string)

        Returns:
            List of dictionaries containing plate detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'plate_image': numpy array
                },
                ...
            ]
        """
        try:
            logger.info("Running license plate detection")

            # Run inference
            results = self.model(image, device=self.device)

            # Extract detections
            detections = []
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf.cpu().numpy()[0])

                    # Filter by confidence threshold
                    if confidence >= self.confidence_threshold:
                        bbox = box.xyxy.cpu().numpy()[0].tolist()
                        x1, y1, x2, y2 = map(int, bbox)

                        # Extract plate image region
                        if isinstance(image, str):
                            full_image = cv2.imread(image)
                        else:
                            full_image = image

                        plate_img = full_image[y1:y2, x1:x2]

                        detection = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'plate_image': plate_img
                        }
                        detections.append(detection)

            logger.info(f"Detected {len(detections)} license plates")
            return detections

        except Exception as e:
            logger.error(f"Error during license plate detection: {e}")
            raise

    def detect_in_roi(self, full_image, vehicle_bbox):
        """
        Detect license plates within vehicle bounding box region.

        Args:
            full_image: Full input image (numpy array or path)
            vehicle_bbox: Vehicle bounding box [x1, y1, x2, y2]

        Returns:
            List of plate detections with bbox coordinates adjusted to full image
        """
        try:
            # Load image if path is provided
            if isinstance(full_image, str):
                full_image = cv2.imread(full_image)

            # Extract vehicle region
            x1, y1, x2, y2 = map(int, vehicle_bbox)
            vehicle_region = full_image[y1:y2, x1:x2]

            # Detect plates in vehicle region
            plate_detections = self.detect(vehicle_region)

            # Adjust bounding boxes to full image coordinates
            for detection in plate_detections:
                px1, py1, px2, py2 = detection['bbox']
                detection['bbox'] = [
                    px1 + x1,  # Adjust x1
                    py1 + y1,  # Adjust y1
                    px2 + x1,  # Adjust x2
                    py2 + y1   # Adjust y2
                ]

                # Extract plate image from full image with adjusted coordinates
                adj_x1, adj_y1, adj_x2, adj_y2 = map(int, detection['bbox'])
                detection['plate_image'] = full_image[adj_y1:adj_y2, adj_x1:adj_x2]

            return plate_detections

        except Exception as e:
            logger.error(f"Error detecting plates in ROI: {e}")
            raise
