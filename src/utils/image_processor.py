import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_image(image_path):
    """Load image from path using OpenCV."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def convert_to_rgb(image):
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_bgr(image):
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def resize_image(image, target_size):
    """Resize image to target size (width, height)."""
    return cv2.resize(image, target_size)

def save_image(image, output_path):
    """Save image to disk."""
    try:
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False

def get_image_dimensions(image_path):
    """Get image width and height."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        height, width = image.shape[:2]
        return width, height
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        return None, None

def preprocess_for_ocr(plate_image):
    """Preprocess license plate image for better OCR results."""
    try:
        # Resize if too small (OCR works better with larger images)
        height, width = plate_image.shape[:2]
        if height < 50 or width < 150:
            scale = max(50 / height, 150 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            plate_image = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Dilate slightly to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        return dilated
    except Exception as e:
        logger.error(f"Error preprocessing image for OCR: {e}")
        return plate_image

def draw_bounding_box(image, bbox, label, color=(0, 255, 0), thickness=2):
    """Draw bounding box on image with label."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
    label_y = max(y1, label_size[1] + 10)
    cv2.rectangle(image, (x1, label_y - label_size[1] - 10),
                  (x1 + label_size[0], label_y), color, -1)

    # Draw label text
    cv2.putText(image, label, (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness)

    return image
