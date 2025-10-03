📸 Real-Time Face Recognition Attendance System
This project is a comprehensive attendance system that uses real-time face recognition. It is built with Python using the Streamlit framework for the user interface and the powerful InsightFace library for state-of-the-art face detection and recognition.

✨ Features
🖥️ User-Friendly Web Interface: A clean and simple UI built with Streamlit.

📹 Real-Time Face Detection: Captures and processes video streams from a webcam in real time.

🎯 Accurate Face Recognition: Uses the buffalo_l model from InsightFace for high-accuracy facial recognition.

🧑‍🏫 User Registration: Easily register new users (e.g., Students, Teachers) by capturing their facial embeddings.

✍️ Automatic Attendance Logging: Detects known individuals and logs their attendance with a timestamp.

☁️ Cloud Database: Utilises a Redis cloud database for storing facial data and attendance logs, making it accessible from anywhere.

🛠️ Tech Stack
Backend: Python 🐍

Frontend: Streamlit 🎈

Face Recognition: InsightFace, OpenCV

Database: Redis ⚡

Libraries: Streamlit-Webrtc, Pandas, NumPy

🚀 Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
🐍 Python 3.8+: Make sure you have Python installed.

☁️ Redis Cloud Account: This project uses a cloud-hosted Redis instance for data storage.

Sign up for a free account at Redis Enterprise Cloud.

Create a new subscription and database.

Note down your database Hostname, Port, and Password.



2. Download the buffalo_l Model
The InsightFace buffalo_l model is required for face analysis. It is too large to be included in the repository.

🔗 Download Link: https://drive.google.com/drive/folders/1E5eOEcBT603R2t2gfFYuya_ONtNwl4bz?usp=sharing

📍 Placement:

After downloading, you will have several model files (e.g., det_10g.onnx, w600k_r50.onnx, etc.).

In the root directory of the project, create a new folder named .insightface.

Inside .insightface, create another folder named models.

Inside models, create a final folder named buffalo_l.

Place all the downloaded model files inside this buffalo_l folder.

The final directory structure should look like this:

your-project-root/
├── .insightface/
│   └── models/
│       └── buffalo_l/
│           ├── 1k3d68.onnx
│           ├── 2d106det.onnx
│           ├── det_10g.onnx
│           ├── genderage.onnx
│           ├── r100-arcface-ms1m-refine-v2.onnx
│           └── w600k_r50.onnx
├── app.py
├── face_rec_merged.py
└── ... other files

3. Install Dependencies
Install all the required Python libraries using the provided requirements.txt file.

pip install -r requirements.txt

4. Configure the Database
Open the face_rec_merged.py file and update the Redis connection details with your own credentials from your Redis Cloud account.

# --- Configuration and Initialization ---

# Connect to Redis Client
# Replace with your actual Redis credentials
HOSTNAME = 'your-redis-hostname'
PORTNUMBER = your-redis-port
PASSWORD = 'your-redis-password'

▶️ How to Run the Application
Open your terminal in the project's root directory.

Run the following command:

streamlit run app.py

The application will open in your web browser.

➡️ Application Workflow
Registration: Navigate to the "Registration" page from the sidebar. Enter the person's name and role, and look at the camera to collect face samples. Click "Submit" to save.

Attendance: Go to the "Attendance" page. The system will start the webcam and automatically detect registered individuals, logging their attendance every 30 seconds.

🗂️ File Structure
app.py: The main Streamlit application file. It handles the UI and page navigation.

face_rec_merged.py: The backend logic for the application. It contains all the code for face detection, recognition, and database interactions.

requirements.txt: A list of all Python dependencies for the project.

face_rec_merged.py: The backend logic for the application. It contains all the code for face detection, recognition, and database interactions.

requirements.txt: A list of all Python dependencies for the project.
