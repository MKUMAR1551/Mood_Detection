🎭 AI Facial Mood Detection System
A high-accuracy, real-time emotion recognition application built with OpenCV and Python. This system uses computer vision techniques to detect faces and classify emotions (Happy, Sad, Angry, etc.) via a webcam feed. It features a modern Dark Mode GUI, automatic data augmentation, and advanced lighting correction.

✨ Key Features
🧠 Advanced LBPH Algorithm: Uses Local Binary Patterns Histograms for robust face recognition that requires less training data than deep learning models.

💡 High-Accuracy Preprocessing: Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve detection in poor lighting conditions.

🔄 Smart Data Augmentation: Automatically quadruples the training dataset by generating flipped and rotated versions of every image (-10°/+10° tilts).

🖥️ Modern Dashboard UI: A custom Dark Theme interface with animated rounded buttons, live system logs, and a status dashboard.

📈 Performance Analytics: Built-in simulation tool to generate Learning Curves (Accuracy vs. Data Size) using Matplotlib.

🚀 Automatic Dataset Handling: Automatically downloads the CK+48 dataset from Kaggle if not found locally.

🛠️ Installation
1. Clone the Repository
Bash

git clone https://github.com/MKUMAR1551/Mood_Detection.git
cd Mood_Detection
2. Install Dependencies
You need Python installed. Run the following command to install the required libraries:

Bash

pip install opencv-python numpy matplotlib kagglehub
(Note: tkinter usually comes pre-installed with Python. If you get an error, you may need to install python-tk depending on your OS).

🚀 Usage
1. Run the Application
Start the main dashboard by running the UI file:

Bash

python mood_detection_ui.py
2. First-Time Setup (Training)
Since the trained model file (mood_model.yml) is too large for GitHub, you must train it locally once:

Click the "⚡ Retrain Model" button on the dashboard.

The system will automatically download the dataset (if missing), process images, and train the model.

Wait for the status to turn Green (READY).

3. Start Detection
Click "START CAMERA" to launch the webcam window.

Green Box: High confidence prediction (>90%).

Orange Box: Lower confidence prediction.

Press 'q' or the "STOP CAMERA" button to end the session.

📂 Project Structure
Mood_Detection/
│
├── mood_detection_ui.py      # MAIN ENTRY POINT: Handles the GUI, buttons, and user interaction
├── mood_detection_logic.py   # BACKEND: Handles image processing, math, and training algorithms
├── haarcascade_...xml        # Pre-trained face detector (Downloaded automatically)
├── .gitignore                # Tells Git to ignore the large model file
├── README.md                 # Project documentation
└── CK+48/                    # Dataset folder (Downloaded automatically)
📊 How It Works
Face Detection: Uses Haar Cascades to locate faces in the video frame.

Preprocessing:

Converts the face to Grayscale.

Applies CLAHE to fix shadows and contrast.

Resizes to a standard 200x200 pixel format.

Feature Extraction: The LBPH recognizer analyzes the texture of the face (wrinkles, mouth shape, eye openness) to determine the emotion.

Prediction: The model compares the live face against the trained patterns and outputs the mood with the highest confidence score.

⚠️ Troubleshooting
"Model not found" Error: The mood_model.yml file is excluded from the repository because it exceeds GitHub's file size limit (400MB+). You must click "Retrain Model" the first time you run the app to generate this file on your own machine.

Camera not opening: Ensure no other app (Zoom, Teams) is using your webcam. If you have multiple cameras, you may need to change cv2.VideoCapture(0) to cv2.VideoCapture(1) in the mood_detection_logic.py file.

📜 License
This project is open-source and available for educational purposes.
