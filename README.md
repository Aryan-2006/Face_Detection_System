📸 Real-Time Face Recognition Attendance System
This project is a comprehensive attendance system that uses real-time face recognition. It is built with Python using the Streamlit framework for the user interface and the powerful InsightFace library for state-of-the-art face detection and recognition.

✨ Features
🖥️ User-Friendly Web Interface: A clean and simple UI built with Streamlit.

📹 Real-Time Face Detection: Captures and processes video streams from a webcam in real time.

🎯 Accurate Face Recognition: Uses the buffalo_l model from InsightFace for high-accuracy facial recognition.

🧑‍🏫 User Registration: Easily register new users (e.g., Students, Teachers) by capturing their facial embeddings.

✍️ Automatic Attendance Logging: Detects known individuals and logs their attendance with a timestamp.

💾 Flexible Database Support: Easily configurable to work with a local MongoDB or MySQL server.

🛠️ Tech Stack
Backend: Python 🐍

Frontend: Streamlit 🎈

Face Recognition: InsightFace, OpenCV

Database: MongoDB / MySQL (configurable)

Libraries: Streamlit-Webrtc, Pandas, NumPy

🚀 Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
🐍 Python 3.8+: Make sure you have Python installed.

🗄️ MongoDB or MySQL: Install and run a local instance of your chosen database.

Install MongoDB Community Edition

[suspicious link removed]

2. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>

3. Download the buffalo_l Model
The InsightFace buffalo_l model is required for face analysis. It is too large to be included in the repository.

🔗 Download Link: Download buffalo_l Model Files from Google Drive

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

4. Install Dependencies
Install all the required Python libraries using the provided requirements.txt file.

pip install -r requirements.txt

5. Configure the Database
Open the face_rec_merged.py file and update the database connection details to point to your local MongoDB or MySQL instance. (Instructions for this are in the code comments).

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
