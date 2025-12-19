import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for the application."""

    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self.base_dir = Path(__file__).parent.parent.parent

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def get(self, key, default=None):
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_model_path(self, model_type):
        """Get absolute path for model files."""
        relative_path = self.get(f'models.{model_type}.path')
        if relative_path:
            return str(self.base_dir / relative_path)
        return None

    def get_db_uri(self):
        """Get database connection URI."""
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url

        db_type = self.get('database.type', 'sqlite')
        if db_type == 'sqlite':
            db_path = self.get('database.sqlite.path', 'database/detections.db')
            return f"sqlite:///{self.base_dir / db_path}"
        elif db_type == 'postgresql':
            host = self.get('database.postgresql.host')
            port = self.get('database.postgresql.port')
            database = self.get('database.postgresql.database')
            username = self.get('database.postgresql.username')
            password = self.get('database.postgresql.password')
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        return None

    def get_upload_folder(self):
        """Get uploads directory path."""
        uploads_dir = self.get('storage.uploads_dir', 'uploads')
        return str(self.base_dir / uploads_dir)

    def get_output_folder(self):
        """Get outputs directory path."""
        outputs_dir = self.get('storage.outputs_dir', 'outputs/detections')
        return str(self.base_dir / outputs_dir)

    def get_allowed_extensions(self):
        """Get list of allowed file extensions."""
        return self.get('storage.allowed_extensions', ['jpg', 'jpeg', 'png'])

    def get_max_file_size(self):
        """Get maximum file size in bytes."""
        return self.get('storage.max_file_size', 10485760)

config = Config()
