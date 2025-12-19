# Vehicle Detection & License Plate Recognition Web Application

A full-stack web application that detects vehicles in images and extracts license plate numbers using computer vision and OCR. Built with Flask (backend API), Streamlit (frontend UI), and custom-trained YOLO models.

## Features

- **Vehicle Detection**: Automatically detect cars, trucks, buses, and motorcycles using YOLOv8
- **License Plate Detection**: Custom-trained YOLO model for accurate plate localization
- **OCR Text Extraction**: Extract license plate numbers using Pytesseract
- **Web Interface**: User-friendly Streamlit UI for image upload and visualization
- **RESTful API**: Flask backend with comprehensive API endpoints
- **Detection History**: Store and browse past detections with SQLite database
- **Bounding Box Visualization**: See detected vehicles and plates with confidence scores

## Architecture

```
Frontend (Streamlit) ←→ Backend API (Flask) ←→ ML Pipeline (YOLO + OCR) ←→ Database (SQLite)
```

**ML Pipeline:**
1. YOLOv8 for vehicle detection
2. Custom-trained YOLO for license plate detection
3. Pytesseract OCR for text extraction

## Project Structure

```
Vehicle-Detection-and-License-Plate-Recognition/
├── data/                    # Dataset storage
├── models/                  # Trained ML models
├── src/
│   ├── inference/          # ML pipeline
│   ├── api/                # Flask backend
│   ├── database/           # Database layer
│   ├── ui/                 # Streamlit frontend
│   └── utils/              # Utilities
├── uploads/                # Uploaded images
├── outputs/                # Processed results
├── database/               # SQLite database
├── config/                 # Configuration files
└── scripts/                # Utility scripts
```

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/subu0106/Vehicle-Detection-and-License-Plate-Recognition.git
cd Vehicle-Detection-and-License-Plate-Recognition
```

### Step 2: Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y tesseract-ocr libtesseract-dev
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Configuration

The application uses the default configuration in `config/config.yaml`. No changes are needed for local development.

If your Tesseract installation is in a non-standard location, update the path in `.env`:
```env
TESSERACT_PATH=/your/custom/path/to/tesseract
```

### Step 5: Train the License Plate Detection Model

Train the custom license plate detection model using the training script:

```bash
python scripts/train_model.py
```

This will train the model and save it to `models/best_license_plate_model.pt`.

## Quick Start

### Option 1: Run with Startup Script (Recommended)

```bash
./run_app.sh
```

This will:
1. Initialize the database
2. Start Flask API on http://localhost:8000
3. Start Streamlit UI on http://localhost:8501

### Option 2: Manual Startup

**Terminal 1 - Initialize Database:**
```bash
python scripts/init_database.py
```

**Terminal 2 - Start Flask API:**
```bash
python -m src.api.app
```

**Terminal 3 - Start Streamlit UI:**
```bash
streamlit run src/ui/streamlit_app.py
```

## Usage

### Web Interface

1. Open http://localhost:8501 in your browser
2. Navigate to "Upload Image"
3. Upload a vehicle image (JPG, JPEG, or PNG)
4. Click "Process Image"
5. View results with detected vehicles and license plates
6. Browse detection history

### API Endpoints

Base URL: `http://localhost:8000/api/v1`

#### POST /upload
Upload an image for detection.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "detection_id": 1,
    "vehicles_detected": 2,
    "plates_detected": 2
  }
}
```

#### GET /detection/{id}
Get detection results by ID.

```bash
curl http://localhost:8000/api/v1/detection/1
```

#### GET /detections
List all detections with pagination.

```bash
curl "http://localhost:8000/api/v1/detections?page=1&per_page=20"
```

#### DELETE /detection/{id}
Delete a detection.

```bash
curl -X DELETE http://localhost:8000/api/v1/detection/1
```

#### GET /statistics
Get overall statistics.

```bash
curl http://localhost:8000/api/v1/statistics
```

#### GET /health
Health check.

```bash
curl http://localhost:8000/api/v1/health
```

## Configuration

### config/config.yaml

Key configuration options:

```yaml
models:
  vehicle_detector:
    path: "models/yolov8n.pt"
    confidence_threshold: 0.5

  plate_detector:
    path: "models/best_license_plate_model.pt"
    confidence_threshold: 0.6

storage:
  max_file_size: 10485760  # 10MB
  allowed_extensions: ["jpg", "jpeg", "png"]

api:
  host: "0.0.0.0"
  port: 8000
```

## Dataset

The dataset includes:
- **Images**: Pictures of vehicles with visible license plates
- **Annotations**: XML files with bounding box information

Dataset location: `data/`

### Data Structure

```
data/
├── raw/          # Original images
├── processed/    # Preprocessed images
└── annotations/  # XML annotation files
```

## Model Training

To train the license plate detection model, use the training script:

```bash
python scripts/train_model.py
```

Training process:
1. Load and preprocess dataset from `data/` folder
2. Convert XML annotations to YOLO format
3. Split data into train/val/test sets
4. Train YOLOv8 model
5. Save best model to `models/best_license_plate_model.pt`

Default training parameters:
- Epochs: 5 (configurable in script)
- Batch size: 16
- Image size: 320x320
- Device: Auto-detect (CUDA if available, else CPU)

## Database Schema

### Tables

**detections**: Main detection records
- Stores image metadata, processing status, and results

**vehicle_detections**: Individual vehicle detections
- Stores bounding boxes and confidence scores

**plate_detections**: License plate detections
- Stores plate bounding boxes, OCR text, and confidence scores

## Development

### Project Requirements

- Flask 3.0.0 - Web framework
- Streamlit 1.29.0 - UI framework
- Ultralytics 8.2.87 - YOLO models
- OpenCV 4.10.0 - Image processing
- Pytesseract 0.3.13 - OCR
- SQLAlchemy 2.0.23 - Database ORM

### Running Tests

```bash
pytest tests/
```

### Code Structure

- `src/inference/` - ML inference pipeline
  - `vehicle_detector.py` - YOLOv8 vehicle detection
  - `plate_detector.py` - Custom YOLO plate detection
  - `ocr_reader.py` - Pytesseract OCR
  - `pipeline.py` - Complete pipeline orchestration

- `src/api/` - Flask backend
  - `app.py` - Flask application
  - `routes.py` - API endpoints
  - `validators.py` - Input validation
  - `middleware.py` - Error handling

- `src/database/` - Database layer
  - `models.py` - SQLAlchemy ORM models
  - `connection.py` - Database connection
  - `repository.py` - CRUD operations

- `src/ui/` - Streamlit frontend
  - `streamlit_app.py` - Main UI application
  - `components/` - UI components

## Performance

- Average processing time: ~3-5 seconds per image
- Vehicle detection accuracy: ~95%
- Plate detection accuracy: ~85-90%
- OCR accuracy: ~70-80% (varies with image quality)

## Challenges & Solutions

### 1. GPU Limitations
**Challenge**: Limited GPU resources in development
**Solution**: Optimized for CPU inference, with GPU support available

### 2. OCR Accuracy
**Challenge**: Tesseract struggles with low-resolution or distorted plates
**Solution**: Implemented image preprocessing (grayscale, thresholding, denoising)

### 3. Real-time Processing
**Challenge**: Processing time for high-resolution images
**Solution**: Synchronous processing with planned async support

## Future Improvements

1. **Enhanced OCR**: Integrate EasyOCR or PaddleOCR for better accuracy
2. **Video Processing**: Support for video file uploads and real-time streams
3. **Batch Processing**: Process multiple images simultaneously
4. **GPU Acceleration**: Add CUDA support for faster inference
5. **Authentication**: User accounts and API key management
6. **Analytics Dashboard**: Visualization of detection trends
7. **Export Features**: CSV/PDF export of detection results
8. **Mobile App**: React Native or Flutter mobile interface
9. **Cloud Deployment**: Docker containers and cloud hosting
10. **Advanced Models**: Fine-tune models for specific regions/plate formats

## Troubleshooting

### Cannot connect to Flask API
- Ensure Flask is running: `python -m src.api.app`
- Check port 8000 is not in use (default port)
- On macOS, disable AirPlay Receiver if using port 5000

### Tesseract not found
- Verify installation: `tesseract --version`
- Update path in `config/config.yaml` or `.env` file

### Database errors
- Reinitialize database: `python scripts/init_database.py`

### Model not found
- Ensure `models/best_license_plate_model.pt` exists
- Train model using: `python scripts/train_model.py`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Contributors

- [Subavarshana A](https://github.com/subu0106)

## Acknowledgments

- YOLOv8 by Ultralytics
- Tesseract OCR by Google
- Flask and Streamlit communities

## Contact

For questions or issues, please open an issue on GitHub or contact [subu0106](https://github.com/subu0106).

---

**Project Status**: Active Development

**Last Updated**: December 2025
