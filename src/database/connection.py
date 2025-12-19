from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

from src.database.models import Base
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseConnection:
    """Database connection management."""

    def __init__(self):
        self.engine = None
        self.Session = None

    def initialize(self, database_url=None):
        """Initialize database connection."""
        if database_url is None:
            database_url = config.get_db_uri()

        try:
            logger.info(f"Initializing database connection")

            # Create engine
            self.engine = create_engine(
                database_url,
                echo=False,
                pool_pre_ping=True
            )

            # Create session factory
            session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(session_factory)

            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database connection: {e}")
            raise

    def create_tables(self):
        """Create all tables in the database."""
        try:
            logger.info("Creating database tables")
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    def drop_tables(self):
        """Drop all tables in the database."""
        try:
            logger.info("Dropping database tables")
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise

    def get_session(self):
        """Get database session."""
        if self.Session is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.Session()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Close database connection."""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")

# Global database connection instance
db_connection = DatabaseConnection()
