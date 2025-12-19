import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_unique_filename(original_filename):
    """Generate a unique filename while preserving extension."""
    name, ext = os.path.splitext(secure_filename(original_filename))
    unique_id = str(uuid.uuid4())[:8]
    return f"{name}_{unique_id}{ext}"

def save_uploaded_file(file, upload_folder):
    """Save uploaded file to upload folder."""
    try:
        os.makedirs(upload_folder, exist_ok=True)
        filename = generate_unique_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        logger.info(f"File saved successfully: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

def delete_detection_files(uploaded_path, processed_path):
    """Delete both uploaded and processed images."""
    try:
        if uploaded_path and os.path.exists(uploaded_path):
            os.remove(uploaded_path)
            logger.info(f"Deleted uploaded file: {uploaded_path}")

        if processed_path and os.path.exists(processed_path):
            os.remove(processed_path)
            logger.info(f"Deleted processed file: {processed_path}")

        return True
    except Exception as e:
        logger.error(f"Error deleting files: {e}")
        return False

def validate_file_size(file, max_size):
    """Check if file size is within limit."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size <= max_size

def validate_file_extension(filename, allowed_extensions):
    """Check if file extension is allowed."""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in allowed_extensions

def get_file_size(filepath):
    """Get file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        return 0
