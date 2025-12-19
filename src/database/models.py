from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Detection(Base):
    """Main detection record for each uploaded image."""

    __tablename__ = 'detections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # File information
    original_filename = Column(String(255), nullable=False)
    uploaded_image_path = Column(String(500), nullable=False)
    processed_image_path = Column(String(500), nullable=True)

    # Detection results
    vehicles_detected = Column(Integer, default=0)
    plates_detected = Column(Integer, default=0)

    # Processing status
    status = Column(String(50), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)

    # Metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    # Relationships
    vehicles = relationship("VehicleDetection", back_populates="detection", cascade="all, delete-orphan")
    plates = relationship("PlateDetection", back_populates="detection", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'original_filename': self.original_filename,
            'uploaded_image_path': self.uploaded_image_path,
            'processed_image_path': self.processed_image_path,
            'vehicles_detected': self.vehicles_detected,
            'plates_detected': self.plates_detected,
            'status': self.status,
            'error_message': self.error_message,
            'processing_time_seconds': self.processing_time_seconds,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'file_size_bytes': self.file_size_bytes
        }

class VehicleDetection(Base):
    """Individual vehicle detections within each image."""

    __tablename__ = 'vehicle_detections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(Integer, ForeignKey('detections.id', ondelete='CASCADE'), nullable=False)

    # Bounding box coordinates
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)

    # Detection metadata
    confidence = Column(Float, nullable=False)
    vehicle_class = Column(String(50), nullable=True)

    # Relationships
    detection = relationship("Detection", back_populates="vehicles")
    plates = relationship("PlateDetection", back_populates="vehicle", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'detection_id': self.detection_id,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'confidence': self.confidence,
            'vehicle_class': self.vehicle_class
        }

class PlateDetection(Base):
    """License plate detections within vehicles."""

    __tablename__ = 'plate_detections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_detection_id = Column(Integer, ForeignKey('vehicle_detections.id', ondelete='CASCADE'), nullable=False)
    detection_id = Column(Integer, ForeignKey('detections.id', ondelete='CASCADE'), nullable=False)

    # Bounding box coordinates (relative to original image)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)

    # Detection and OCR results
    detection_confidence = Column(Float, nullable=False)
    license_plate_text = Column(String(50), nullable=True)
    ocr_confidence = Column(Float, nullable=True)

    # Preprocessing info
    plate_image_path = Column(String(500), nullable=True)

    # Relationships
    vehicle = relationship("VehicleDetection", back_populates="plates")
    detection = relationship("Detection", back_populates="plates")

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'vehicle_detection_id': self.vehicle_detection_id,
            'detection_id': self.detection_id,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'detection_confidence': self.detection_confidence,
            'license_plate_text': self.license_plate_text,
            'ocr_confidence': self.ocr_confidence,
            'plate_image_path': self.plate_image_path
        }
