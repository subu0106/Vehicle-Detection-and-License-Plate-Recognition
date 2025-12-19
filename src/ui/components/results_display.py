import streamlit as st
import requests
from PIL import Image
import os

def render_results_page(api_url):
    """Render the detection results page."""

    st.header("📊 Detection Results")

    # Get detection ID from session state or user input
    detection_id = st.session_state.get('current_detection_id')

    # Allow user to enter detection ID
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input_id = st.text_input(
            "Enter Detection ID",
            value=str(detection_id) if detection_id else "",
            placeholder="e.g., 123"
        )
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Load Results", type="primary")

    if search_button and user_input_id:
        try:
            detection_id = int(user_input_id)
            st.session_state.current_detection_id = detection_id
        except ValueError:
            st.error("❌ Invalid detection ID. Please enter a number.")
            return

    if detection_id:
        try:
            # Fetch detection results from API
            response = requests.get(f"{api_url}/detection/{detection_id}", timeout=10)

            if response.status_code == 200:
                result = response.json()

                if result['status'] == 'success':
                    data = result['data']

                    # Display overall information
                    st.subheader("Detection Information")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Detection ID", data['id'])
                    with col2:
                        st.metric("Vehicles", data['vehicles_detected'])
                    with col3:
                        st.metric("Plates", data['plates_detected'])
                    with col4:
                        status_color = "🟢" if data['status'] == 'completed' else "🔴"
                        st.metric("Status", f"{status_color} {data['status']}")

                    # Display processing time
                    if data.get('processing_time_seconds'):
                        st.info(f"⏱️ Processing Time: {data['processing_time_seconds']:.2f} seconds")

                    # Display processed image if available
                    if data.get('processed_image_path'):
                        st.subheader("Processed Image with Detections")

                        try:
                            image_path = data['processed_image_path']
                            if os.path.exists(image_path):
                                image = Image.open(image_path)
                                st.image(image, caption='Detected Vehicles and License Plates', use_column_width=True)
                            else:
                                st.warning("⚠️ Processed image file not found")
                        except Exception as e:
                            st.error(f"❌ Error loading image: {str(e)}")

                    # Display vehicle and plate details
                    if data.get('vehicles'):
                        st.subheader("Vehicle and Plate Details")

                        for idx, vehicle in enumerate(data['vehicles'], 1):
                            with st.expander(f"🚗 Vehicle {idx} - {vehicle['vehicle_class'].upper()} (Confidence: {vehicle['confidence']:.2%})"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**Vehicle Information:**")
                                    st.write(f"- Type: {vehicle['vehicle_class']}")
                                    st.write(f"- Confidence: {vehicle['confidence']:.2%}")
                                    st.write(f"- Bounding Box: {vehicle['bbox']}")

                                with col2:
                                    st.write("**License Plates:**")

                                    if vehicle.get('plates'):
                                        for pidx, plate in enumerate(vehicle['plates'], 1):
                                            plate_text = plate.get('license_plate_text') or 'Not detected'
                                            ocr_conf = plate.get('ocr_confidence', 0)

                                            st.write(f"**Plate {pidx}:**")
                                            st.write(f"- Text: **{plate_text}**")
                                            st.write(f"- Detection Confidence: {plate.get('detection_confidence', 0):.2%}")
                                            st.write(f"- OCR Confidence: {ocr_conf:.2%}")
                                            st.write(f"- Bounding Box: {plate['bbox']}")
                                            st.write("---")
                                    else:
                                        st.write("No license plates detected for this vehicle")

                    else:
                        st.info("No vehicles detected in this image")

                    # Display error message if any
                    if data.get('error_message'):
                        st.warning(f"⚠️ {data['error_message']}")

                    # Display metadata
                    with st.expander("📋 Metadata"):
                        st.write(f"**Original Filename:** {data['original_filename']}")
                        st.write(f"**Created At:** {data['created_at']}")
                        st.write(f"**Updated At:** {data['updated_at']}")
                        if data.get('image_width') and data.get('image_height'):
                            st.write(f"**Image Dimensions:** {data['image_width']} x {data['image_height']}")
                        if data.get('file_size_bytes'):
                            file_size_mb = data['file_size_bytes'] / (1024 * 1024)
                            st.write(f"**File Size:** {file_size_mb:.2f} MB")

                else:
                    st.error(f"❌ Error: {result.get('message', 'Unknown error')}")

            elif response.status_code == 404:
                st.error(f"❌ Detection ID {detection_id} not found")
            else:
                st.error("❌ Error fetching detection results")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the backend server")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    else:
        st.info("👆 Please enter a detection ID or upload a new image")

        # Show recent detections
        try:
            response = requests.get(f"{api_url}/detections?page=1&per_page=5", timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    detections = result['data']['detections']

                    if detections:
                        st.subheader("Recent Detections")
                        for det in detections:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col1:
                                st.write(f"**ID:** {det['id']}")
                            with col2:
                                st.write(f"{det['original_filename']}")
                            with col3:
                                if st.button("View", key=f"view_{det['id']}"):
                                    st.session_state.current_detection_id = det['id']
                                    st.rerun()
        except:
            pass
