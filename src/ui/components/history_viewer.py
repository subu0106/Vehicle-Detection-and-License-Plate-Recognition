import streamlit as st
import requests
import pandas as pd

def render_history_page(api_url):
    """Render the detection history page."""

    st.header("📜 Detection History")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "completed", "failed", "pending", "processing"]
        )

    with col2:
        sort_by = st.selectbox(
            "Sort By",
            ["created_at", "updated_at", "vehicles_detected", "plates_detected"]
        )

    with col3:
        order = st.selectbox(
            "Order",
            ["desc", "asc"]
        )

    with col4:
        per_page = st.selectbox(
            "Per Page",
            [10, 20, 50, 100],
            index=1
        )

    # Pagination
    if 'history_page' not in st.session_state:
        st.session_state.history_page = 1

    # Build query parameters
    params = {
        'page': st.session_state.history_page,
        'per_page': per_page,
        'sort_by': sort_by,
        'order': order
    }

    if status_filter != "All":
        params['status'] = status_filter

    try:
        # Fetch detections from API
        response = requests.get(f"{api_url}/detections", params=params, timeout=10)

        if response.status_code == 200:
            result = response.json()

            if result['status'] == 'success':
                data = result['data']
                detections = data['detections']
                pagination = data['pagination']

                # Display statistics
                st.subheader("Statistics")

                try:
                    stats_response = requests.get(f"{api_url}/statistics", timeout=10)
                    if stats_response.status_code == 200:
                        stats_result = stats_response.json()
                        if stats_result['status'] == 'success':
                            stats = stats_result['data']

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Detections", stats['total_detections'])
                            with col2:
                                st.metric("Total Vehicles", stats['total_vehicles_detected'])
                            with col3:
                                st.metric("Total Plates", stats['total_plates_detected'])
                            with col4:
                                st.metric("Success Rate", f"{stats['success_rate']:.0%}")
                except:
                    pass

                st.markdown("---")

                # Display detections
                if detections:
                    st.subheader(f"Detections (Page {pagination['page']} of {pagination['total_pages']})")

                    # Create DataFrame for better display
                    df_data = []
                    for det in detections:
                        df_data.append({
                            'ID': det['id'],
                            'Filename': det['original_filename'],
                            'Status': det['status'],
                            'Vehicles': det['vehicles_detected'],
                            'Plates': det['plates_detected'],
                            'Created': det['created_at'][:19] if det['created_at'] else 'N/A',
                            'Processing Time (s)': f"{det['processing_time_seconds']:.2f}" if det['processing_time_seconds'] else 'N/A'
                        })

                    df = pd.DataFrame(df_data)

                    # Display table
                    st.dataframe(df, hide_index=True)

                    # Actions for each detection
                    st.subheader("Actions")

                    for det in detections:
                        col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])

                        with col1:
                            st.write(f"**ID {det['id']}**")
                        with col2:
                            status_emoji = "🟢" if det['status'] == 'completed' else "🔴"
                            st.write(f"{status_emoji} {det['status']}")
                        with col3:
                            if st.button("👁️ View", key=f"view_{det['id']}"):
                                st.session_state.current_detection_id = det['id']
                                st.session_state.page = 'results'
                                st.rerun()
                        with col4:
                            if st.button("🗑️ Delete", key=f"delete_{det['id']}"):
                                if delete_detection(api_url, det['id']):
                                    st.success(f"✅ Detection {det['id']} deleted")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete detection {det['id']}")
                        with col5:
                            st.write("")  # Spacing

                    # Pagination controls
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        if st.button("⏮️ First", disabled=(pagination['page'] == 1)):
                            st.session_state.history_page = 1
                            st.rerun()

                    with col2:
                        if st.button("◀️ Previous", disabled=(pagination['page'] == 1)):
                            st.session_state.history_page -= 1
                            st.rerun()

                    with col3:
                        st.write(f"Page {pagination['page']} of {pagination['total_pages']}")

                    with col4:
                        if st.button("Next ▶️", disabled=(pagination['page'] >= pagination['total_pages'])):
                            st.session_state.history_page += 1
                            st.rerun()

                    with col5:
                        if st.button("Last ⏭️", disabled=(pagination['page'] >= pagination['total_pages'])):
                            st.session_state.history_page = pagination['total_pages']
                            st.rerun()

                else:
                    st.info("No detections found matching the filters")

            else:
                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")

        else:
            st.error("❌ Error fetching detection history")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend server")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def delete_detection(api_url, detection_id):
    """Delete a detection via API."""
    try:
        response = requests.delete(f"{api_url}/detection/{detection_id}", timeout=10)
        return response.status_code == 200
    except:
        return False
