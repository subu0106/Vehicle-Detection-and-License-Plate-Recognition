import streamlit as st
import requests
import os

from src.ui.components.upload_widget import render_upload_page
from src.ui.components.results_display import render_results_page
from src.ui.components.history_viewer import render_history_page
from src.utils.config import config

# Page configuration
st.set_page_config(
    page_title=config.get('ui.page_title', 'Vehicle Detection & License Plate Recognition'),
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = config.get('ui.api_url', 'http://localhost:5000/api/v1')

# Initialize session state
if 'current_detection_id' not in st.session_state:
    st.session_state.current_detection_id = None

if 'page' not in st.session_state:
    st.session_state.page = 'upload'

def check_api_health():
    """Check if Flask API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main Streamlit application."""

    # Title
    st.title("🚗 Vehicle Detection & License Plate Recognition")

    # Check API health
    if not check_api_health():
        st.error("❌ Cannot connect to Flask API. Please ensure the backend server is running.")
        st.info("Start the Flask API with: `python -m src.api.app`")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📤 Upload Image", "📊 View Results", "📜 Detection History"],
        key="navigation"
    )

    # Map page selection to session state
    if page == "📤 Upload Image":
        st.session_state.page = 'upload'
    elif page == "📊 View Results":
        st.session_state.page = 'results'
    elif page == "📜 Detection History":
        st.session_state.page = 'history'

    # Render selected page
    if st.session_state.page == 'upload':
        render_upload_page(API_URL)
    elif st.session_state.page == 'results':
        render_results_page(API_URL)
    elif st.session_state.page == 'history':
        render_history_page(API_URL)

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application detects vehicles in images and extracts license plate numbers using:\n\n"
        "- **YOLOv8** for vehicle detection\n"
        "- **Custom YOLO** for license plate detection\n"
        "- **Pytesseract OCR** for text extraction"
    )

if __name__ == '__main__':
    main()
