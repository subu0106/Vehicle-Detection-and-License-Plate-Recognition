import cv2
import os
import time
from datetime import datetime

from src.inference.vehicle_detector import VehicleDetector
from src.inference.plate_detector import PlateDetector
from src.inference.ocr_reader import OCRReader
from src.utils.logger import get_logger
from src.utils.image_processor import load_image, save_image, draw_bounding_box, get_image_dimensions
from src.utils.config import config

logger = get_logger(__name__)

class DetectionPipeline:
    """Complete inference pipeline orchestrator."""

    def __init__(self):
        """Initialize detection pipeline with all components."""
        logger.info("Initializing detection pipeline")

        try:
            self.vehicle_detector = VehicleDetector()
            self.plate_detector = PlateDetector()
            self.ocr_reader = OCRReader()
            logger.info("Detection pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detection pipeline: {e}")
            raise

    def process(self, image_path, save_visualization=True):
        """
        Execute complete detection pipeline.

        Args:
            image_path: Path to input image
            save_visualization: Whether to save annotated image

        Returns:
            Dictionary containing complete detection results:
            {
                'status': str,
                'processing_time': float,
                'image_path': str,
                'processed_image_path': str,
                'image_width': int,
                'image_height': int,
                'vehicles_detected': int,
                'plates_detected': int,
                'vehicles': [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float,
                        'class_name': str,
                        'plates': [
                            {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float,
                                'text': str,
                                'ocr_confidence': float
                            },
                            ...
                        ]
                    },
                    ...
                ],
                'error_message': str (if any)
            }
        """
        start_time = time.time()

        try:
            logger.info(f"Processing image: {image_path}")

            # Get image dimensions
            width, height = get_image_dimensions(image_path)

            # Step 1: Detect vehicles
            vehicle_detections = self.vehicle_detector.detect(image_path)

            if not vehicle_detections:
                logger.warning("No vehicles detected")
                return {
                    'status': 'completed',
                    'processing_time': time.time() - start_time,
                    'image_path': image_path,
                    'processed_image_path': None,
                    'image_width': width,
                    'image_height': height,
                    'vehicles_detected': 0,
                    'plates_detected': 0,
                    'vehicles': [],
                    'error_message': 'No vehicles detected in the image'
                }

            # Load full image for plate detection
            full_image = load_image(image_path)

            # Step 2 & 3: For each vehicle, detect plates and perform OCR
            total_plates = 0
            for vehicle in vehicle_detections:
                # Detect plates in vehicle region
                plate_detections = self.plate_detector.detect_in_roi(
                    full_image,
                    vehicle['bbox']
                )

                # Perform OCR on each detected plate
                plates_with_text = []
                for plate in plate_detections:
                    ocr_result = self.ocr_reader.read_plate(plate['plate_image'])

                    plate_info = {
                        'bbox': plate['bbox'],
                        'confidence': plate['confidence'],
                        'text': ocr_result['text'],
                        'ocr_confidence': ocr_result['confidence'],
                        'raw_text': ocr_result['raw_text']
                    }
                    plates_with_text.append(plate_info)
                    total_plates += 1

                vehicle['plates'] = plates_with_text

            # Step 4: Visualize results
            processed_image_path = None
            if save_visualization:
                processed_image_path = self.visualize_results(
                    image_path,
                    vehicle_detections
                )

            processing_time = time.time() - start_time

            logger.info(
                f"Processing completed: {len(vehicle_detections)} vehicles, "
                f"{total_plates} plates in {processing_time:.2f}s"
            )

            return {
                'status': 'completed',
                'processing_time': processing_time,
                'image_path': image_path,
                'processed_image_path': processed_image_path,
                'image_width': width,
                'image_height': height,
                'vehicles_detected': len(vehicle_detections),
                'plates_detected': total_plates,
                'vehicles': vehicle_detections,
                'error_message': None
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'status': 'failed',
                'processing_time': time.time() - start_time,
                'image_path': image_path,
                'processed_image_path': None,
                'image_width': None,
                'image_height': None,
                'vehicles_detected': 0,
                'plates_detected': 0,
                'vehicles': [],
                'error_message': str(e)
            }

    def visualize_results(self, image_path, vehicle_detections):
        """
        Draw bounding boxes on image and save.

        Args:
            image_path: Path to original image
            vehicle_detections: List of vehicle detections with plates

        Returns:
            Path to saved processed image
        """
        try:
            # Load image
            image = load_image(image_path)

            # Draw vehicle bounding boxes and plates
            for vehicle in vehicle_detections:
                # Draw vehicle box (green)
                vehicle_label = f"{vehicle['class_name']}: {vehicle['confidence']:.2f}"
                draw_bounding_box(
                    image,
                    vehicle['bbox'],
                    vehicle_label,
                    color=(0, 255, 0),
                    thickness=2
                )

                # Draw plate boxes (red)
                for plate in vehicle.get('plates', []):
                    plate_text = plate.get('text', 'Unknown')
                    plate_label = f"Plate: {plate_text}" if plate_text else "Plate"
                    draw_bounding_box(
                        image,
                        plate['bbox'],
                        plate_label,
                        color=(0, 0, 255),
                        thickness=2
                    )

            # Generate output path
            output_dir = config.get_output_folder()
            os.makedirs(output_dir, exist_ok=True)

            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_processed{ext}"
            output_path = os.path.join(output_dir, output_filename)

            # Save annotated image
            save_image(image, output_path)

            logger.info(f"Visualization saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
