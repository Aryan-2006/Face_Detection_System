import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import pandas as pd
# Import the backend logic from our merged file
import face_rec_merged as face_rec

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# --- Initialize Classes and Session State ---
if 'registration_form' not in st.session_state:
    st.session_state.registration_form = face_rec.RegistrationForm()

if 'realtime_pred' not in st.session_state:
    st.session_state.realtime_pred = face_rec.RealTimePred()

# --- Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Registration", "Attendance", "View Records"])

# --- Home Page ---
if page == "Home":
    st.header("Attendance System using Face Recognition")
    st.write("Welcome to the attendance system. Please navigate to the desired page using the sidebar.")
    st.write(
        """
        This application is built using a modern face recognition pipeline. Here's how it works:
        - **Registration**: Captures facial features using the 'InsightFace' model and stores them in a Redis database.
        - **Attendance**: Detects faces in real-time, compares them against the registered database using Cosine Similarity, and logs attendance.
        - **View Records**: Allows you to see all registered users and view the attendance logs.
        """
    )

# --- Registration Page ---
elif page == "Registration":
    st.header("Registration Form")

    registration_form = st.session_state.registration_form

    # Step 1: Collect Person's Name and Role
    person_name = st.text_input(label='Name', placeholder='First & Last Name')
    role = st.selectbox(label='Select Role', options=('Student', 'Teacher'))

    # Step 2: Collect Facial Embeddings
    st.write("Please look at the camera to collect face samples.")


    def video_callback_reg(frame):
        img = frame.to_ndarray(format='bgr24')
        reg_img, embedding = registration_form.get_embedding(img)
        return av.VideoFrame.from_ndarray(reg_img, format='bgr24')


    webrtc_streamer(key='registration', video_frame_callback=video_callback_reg)

    # Step 3: Save Data to Redis
    if st.button('Submit'):
        if registration_form.sample < 20:
            st.warning("Please collect at least 20 samples.")
        elif not person_name or not role:
            st.warning("Please enter a name and select a role.")
        else:
            status, message = registration_form.save_data_redis(person_name, role)
            if status == 'success':
                st.success(message)
            else:
                st.error(message)

# --- Attendance Page ---
elif page == "Attendance":
    st.header("Real-Time Attendance")

    with st.spinner('Retrieving data from database...'):
        redis_face_db = face_rec.retrieve_data(name='academy:register')
        if redis_face_db.empty:
            st.warning("No users registered yet. Please register users on the Registration page.")
        else:
            st.success("Database loaded successfully.")

    st.write("The system will automatically log attendance every 30 seconds.")


    def video_callback_pred(frame):
        img = frame.to_ndarray(format="bgr24")
        pred_img = st.session_state.realtime_pred.face_prediction(img, redis_face_db)
        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


    if not redis_face_db.empty:
        webrtc_streamer(
            key="RealTimePrediction",
            video_frame_callback=video_callback_pred
        )
    else:
        st.write("Cannot start prediction without registered users.")