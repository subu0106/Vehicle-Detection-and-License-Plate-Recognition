import streamlit as st
import requests
from PIL import Image
import io

def render_upload_page(api_url):
    """Render the image upload page."""

    st.header("📤 Upload Vehicle Image")

    st.markdown("""
    Upload an image containing vehicles to detect license plates and extract text.
    Supported formats: JPG, JPEG, PNG
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a vehicle image for detection"
    )

    if uploaded_file is not None:
        # Display image preview
        st.subheader("Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Upload button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Process Image", type="primary"):
                with st.spinner("Processing image... This may take a moment."):
                    # Reset file pointer
                    uploaded_file.seek(0)

                    # Prepare file for upload
                    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}

                    try:
                        # Send to Flask API
                        response = requests.post(
                            f"{api_url}/upload",
                            files=files,
                            timeout=60
                        )

                        if response.status_code == 200:
                            result = response.json()

                            if result['status'] == 'success':
                                data = result['data']

                                st.success("✅ Image processed successfully!")

                                # Display results summary
                                st.subheader("Detection Summary")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Vehicles Detected", data['vehicles_detected'])
                                with col2:
                                    st.metric("Plates Detected", data['plates_detected'])
                                with col3:
                                    st.metric("Status", data['processing_status'])

                                # Store detection ID in session state
                                st.session_state.current_detection_id = data['detection_id']

                                # Link to view results
                                st.info(f"Detection ID: {data['detection_id']}")
                                if st.button("📊 View Detailed Results"):
                                    st.session_state.page = 'results'
                                    st.rerun()

                            else:
                                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")

                        else:
                            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                            error_message = error_data.get('message', 'Unknown error occurred')
                            st.error(f"❌ Error processing image: {error_message}")

                    except requests.exceptions.Timeout:
                        st.error("❌ Request timed out. The image may be too large or the server is busy.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to the backend server. Please ensure it's running.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an image to begin")

        # Example instructions
        with st.expander("📖 How to use"):
            st.markdown("""
            1. Click on "Browse files" to select an image from your computer
            2. Preview the image to ensure it's correct
            3. Click "Process Image" to start detection
            4. View the results showing detected vehicles and license plates
            5. Check the "Detection History" page to see past detections
            """)
