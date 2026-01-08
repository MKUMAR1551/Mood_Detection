import cv2
import os
import numpy as np
import random
import urllib.request
import matplotlib.pyplot as plt
import kagglehub # Added to find the dataset automatically

# --- CONFIGURATION ---
SUBJECTS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

def download_haar_if_missing():
    if not os.path.exists(HAAR_CASCADE_FILE):
        print(f"Downloading {HAAR_CASCADE_FILE}...")
        try:
            urllib.request.urlretrieve(HAAR_URL, HAAR_CASCADE_FILE)
        except:
            print("Error: Please download the Haar Cascade XML manually.")
            exit()

def detect_face_roi(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h]

def load_all_data(data_folder_path):
    if not os.path.exists(data_folder_path):
        print(f"Error: Folder '{data_folder_path}' not found.")
        return [], []

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    faces = []
    labels = []
    
    print(f"Scanning {data_folder_path}...")
    print("Loading images... (This may take a minute)")
    
    dirs = os.listdir(data_folder_path)
    for dir_name in dirs:
        if dir_name in SUBJECTS:
            label = SUBJECTS.index(dir_name)
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            
            if not os.path.isdir(subject_dir_path):
                continue
            
            # Count images to ensure folder isn't empty
            image_names = [f for f in os.listdir(subject_dir_path) if not f.startswith(".")]
            if len(image_names) == 0:
                print(f"Warning: '{dir_name}' folder is empty!")
                continue

            for image_name in image_names:
                image_path = os.path.join(subject_dir_path, image_name)
                image = cv2.imread(image_path)
                
                if image is None: continue

                face = detect_face_roi(image, face_cascade)
                if face is not None:
                    face = cv2.resize(face, (200, 200))
                    faces.append(face)
                    labels.append(label)
    return faces, labels

def get_accuracy(model, faces, labels):
    correct = 0
    count = len(faces)
    for i in range(count):
        pred_label, conf = model.predict(faces[i])
        if pred_label == labels[i]:
            correct += 1
    if count == 0: return 0
    return (correct / count) * 100

def show_performance_graph(train_acc, test_acc):
    labels = ['Training Accuracy', 'Testing Accuracy']
    values = [train_acc, test_acc]
    colors = ['#4CAF50', '#FF5722'] 

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    plt.ylim(0, 110)
    plt.ylabel('Accuracy Percentage (%)')
    plt.title('Model Evaluation: Overfitting vs Underfitting Check')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 2, 
                 f'{round(height, 2)}%', ha='center', fontsize=12, fontweight='bold')

    diff = train_acc - test_acc
    if train_acc < 60:
        diagnosis = "Diagnosis: UNDERFITTING (Model is struggling to learn)"
    elif train_acc > 90 and test_acc < 65:
        diagnosis = "Diagnosis: OVERFITTING (Model is memorizing data)"
    elif diff < 15 and train_acc > 70:
        diagnosis = "Diagnosis: GOOD FIT (Model generalizes well)"
    else:
        diagnosis = "Diagnosis: MODERATE FIT"

    plt.figtext(0.5, 0.01, diagnosis, ha="center", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    print("Opening Graph window...")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    download_haar_if_missing()
    
    # 1. FIND DATASET AUTOMATICALLY
    print("Locating dataset via KaggleHub...")
    path = kagglehub.dataset_download("shawon10/ckplus")
    
    # Check for 'CK+48' subfolder inside the download
    possible_subfolder = os.path.join(path, "CK+48")
    if os.path.exists(possible_subfolder):
        DATA_DIR = possible_subfolder
    else:
        DATA_DIR = path
    
    print(f"Dataset found at: {DATA_DIR}")

    # 2. Load & Shuffle
    faces, labels = load_all_data(DATA_DIR)
    
    if len(faces) == 0: 
        print("Error: No faces found. Check if the dataset downloaded correctly.")
        exit()
    
    combined = list(zip(faces, labels))
    random.shuffle(combined)
    faces[:], labels[:] = zip(*combined)

    # 3. Split (80/20)
    split_idx = int(len(faces) * 0.8)
    train_faces, test_faces = faces[:split_idx], faces[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    print(f"Training on {len(train_faces)} images. Testing on {len(test_faces)} images.")

    # 4. Train Temp Model
    print("Training model for evaluation...")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(train_faces, np.array(train_labels))

    # 5. Calculate Accuracy
    print("Calculating accuracy...")
    train_acc = get_accuracy(model, train_faces, train_labels)
    test_acc = get_accuracy(model, test_faces, test_labels)

    # 6. Text Report & Graph
    print(f"\nTraining Accuracy: {train_acc:.2f}%")
    print(f"Testing Accuracy:  {test_acc:.2f}%")
    
    show_performance_graph(train_acc, test_acc)