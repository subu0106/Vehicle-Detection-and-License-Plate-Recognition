import cv2
import pytesseract
import re

from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.image_processor import preprocess_for_ocr

logger = get_logger(__name__)

class OCRReader:
    """Pytesseract OCR for license plate text extraction."""

    def __init__(self, tesseract_path=None, psm_config='--psm 7 --oem 3'):
        """
        Initialize OCR reader.

        Args:
            tesseract_path: Path to tesseract executable
            psm_config: Tesseract page segmentation mode
                       PSM 7: Treat the image as a single text line
                       OEM 3: Default OCR Engine Mode
        """
        if tesseract_path is None:
            tesseract_path = config.get('models.ocr.tesseract_path', '/usr/bin/tesseract')

        self.tesseract_path = tesseract_path
        self.psm_config = psm_config

        self.configure_tesseract()

    def configure_tesseract(self):
        """Configure Tesseract path."""
        try:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            logger.info(f"Tesseract configured at {self.tesseract_path}")
        except Exception as e:
            logger.error(f"Error configuring Tesseract: {e}")
            raise

    def read_plate(self, plate_image):
        """
        Extract text from license plate image.

        Args:
            plate_image: License plate image (numpy array)

        Returns:
            Dictionary containing:
            {
                'text': str,
                'confidence': float,
                'raw_text': str
            }
        """
        try:
            if plate_image is None or plate_image.size == 0:
                logger.warning("Empty or invalid plate image")
                return {
                    'text': None,
                    'confidence': 0.0,
                    'raw_text': ''
                }

            # Try multiple preprocessing approaches
            results = []

            # Approach 1: Standard preprocessing
            processed_plate = preprocess_for_ocr(plate_image)
            text1 = pytesseract.image_to_string(processed_plate, config=self.psm_config)
            results.append(text1)

            # Approach 2: Try with original image
            text2 = pytesseract.image_to_string(plate_image, config=self.psm_config)
            results.append(text2)

            # Approach 3: Try with PSM 8 (single word)
            text3 = pytesseract.image_to_string(processed_plate, config='--psm 8 --oem 3')
            results.append(text3)

            # Approach 4: Try with PSM 13 (raw line)
            text4 = pytesseract.image_to_string(processed_plate, config='--psm 13 --oem 3')
            results.append(text4)

            # Choose the best result (longest valid text)
            best_text = ''
            for text in results:
                cleaned = self.post_process_text(text)
                if len(cleaned) > len(best_text) and self.validate_plate_format(cleaned):
                    best_text = cleaned

            # If no valid text found, take the longest
            if not best_text:
                best_text = max([self.post_process_text(t) for t in results], key=len)

            # Get confidence for the best result
            try:
                ocr_data = pytesseract.image_to_data(
                    processed_plate,
                    config=self.psm_config,
                    output_type=pytesseract.Output.DICT
                )
                confidences = [
                    int(conf) for conf in ocr_data['conf']
                    if conf != '-1' and int(conf) > 0
                ]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except:
                avg_confidence = 50.0 if best_text else 0.0

            logger.info(f"OCR extracted text: '{best_text}' (confidence: {avg_confidence:.2f}%)")
            logger.debug(f"All OCR attempts: {results}")

            return {
                'text': best_text if best_text else None,
                'confidence': avg_confidence / 100.0,  # Normalize to 0-1
                'raw_text': results[0].strip()
            }

        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return {
                'text': None,
                'confidence': 0.0,
                'raw_text': ''
            }

    def post_process_text(self, text):
        """
        Clean and validate OCR output.

        Args:
            text: Raw OCR text output

        Returns:
            Cleaned text string
        """
        if not text:
            return ''

        # Remove whitespace and newlines
        text = text.strip().replace('\n', ' ').replace('\r', '')

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (keep only alphanumeric and common separators)
        text = re.sub(r'[^A-Z0-9\-\s]', '', text.upper())

        # Remove spaces
        text = text.replace(' ', '')

        return text

    def validate_plate_format(self, text, pattern=None):
        """
        Validate if text matches expected license plate format.

        Args:
            text: License plate text
            pattern: Regex pattern for validation (optional)

        Returns:
            Boolean indicating if text is valid
        """
        if not text:
            return False

        # Default: at least 2 characters and max 10
        if len(text) < 2 or len(text) > 10:
            return False

        # If pattern provided, validate against it
        if pattern:
            return bool(re.match(pattern, text))

        # Must contain at least one letter or number
        return bool(re.search(r'[A-Z0-9]', text))
