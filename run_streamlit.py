#!/usr/bin/env python3
"""Wrapper to run Streamlit with correct Python path."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run Streamlit app
if __name__ == '__main__':
    import streamlit.web.cli as stcli
    import sys

    sys.argv = ["streamlit", "run", "src/ui/streamlit_app.py"]
    sys.exit(stcli.main())
