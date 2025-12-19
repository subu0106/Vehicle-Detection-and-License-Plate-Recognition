#!/usr/bin/env python3
"""Initialize database with tables."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import db_connection
from src.utils.logger import setup_logger

logger = setup_logger('init_database')

def init_database():
    """Initialize database and create all tables."""
    try:
        logger.info("Starting database initialization")

        # Initialize connection
        db_connection.initialize()

        # Create tables
        db_connection.create_tables()

        logger.info("Database initialized successfully")
        logger.info("Tables created: detections, vehicle_detections, plate_detections")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == '__main__':
    init_database()
