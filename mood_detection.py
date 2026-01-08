# mood_detection.py
import cv2
import os
import numpy as np
import kagglehub
import urllib.request
import random

# --- CONFIGURATION ---
SUBJECTS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
MODEL_FILE = "mood_model.yml"

# Accepting the UI's log window
def download_haar_if_missing(log_func=print):
    if not os.path.exists(HAAR_CASCADE_FILE):
        log_func(f"[Logic] Downloading {HAAR_CASCADE_FILE}...")
        try:
            urllib.request.urlretrieve(HAAR_URL, HAAR_CASCADE_FILE)
        except Exception as e:
            log_func(f"[Error] Downloading file: {e}")

def process_image(img, face_cascade):
    """
    Advanced Processing: Uses CLAHE for superior lighting correction.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # ACCURACY BOOST: Use CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    height, width = gray.shape
    
    # Smart Check (CK+48)
    if width < 100:
        return gray, (0, 0, width, height)
    
    # Webcam Detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def augment_data(image, label):
    """
    Generates 4 variations (Original, Flip, Rotate Left, Rotate Right).
    """
    augmented_faces = []
    augmented_labels = []
    rows, cols = image.shape
    
    # 1. Original
    augmented_faces.append(image)
    augmented_labels.append(label)
    
    # 2. Flipped
    augmented_faces.append(cv2.flip(image, 1))
    augmented_labels.append(label)
    
    # 3. Rotated Left
    M_left = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    augmented_faces.append(cv2.warpAffine(image, M_left, (cols, rows)))
    augmented_labels.append(label)

    # 4. Rotated Right
    M_right = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1)
    augmented_faces.append(cv2.warpAffine(image, M_right, (cols, rows)))
    augmented_labels.append(label)
        
    return augmented_faces, augmented_labels

# Added 'log_func=print' here
def load_dataset(log_func=print):
    data_folder_path = None
    if os.path.exists("CK+48") and len(os.listdir("CK+48")) > 0:
        data_folder_path = "CK+48"
    else:
        log_func("[Logic] Downloading dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("shawon10/ckplus")
            possible_subfolder = os.path.join(path, "CK+48")
            data_folder_path = possible_subfolder if os.path.exists(possible_subfolder) else path
        except:
            log_func("[Error] Could not download dataset.")
            return [], []

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    faces = []
    labels = []
    
    log_func(f"[Logic] Scanning: {data_folder_path}")
    
    dirs = os.listdir(data_folder_path)
    count = 0
    
    for dir_name in dirs:
        if dir_name in SUBJECTS:
            label = SUBJECTS.index(dir_name)
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            if not os.path.isdir(subject_dir_path): continue
            
            for image_name in os.listdir(subject_dir_path):
                if image_name.startswith("."): continue
                image_path = os.path.join(subject_dir_path, image_name)
                image = cv2.imread(image_path)
                if image is None: continue

                face_roi, _ = process_image(image, face_cascade)
                if face_roi is not None:
                    face_resized = cv2.resize(face_roi, (200, 200))
                    aug_faces, aug_labels = augment_data(face_resized, label)
                    faces.extend(aug_faces)
                    labels.extend(aug_labels)
                    count += 1
                    
    log_func(f"[Logic] Success! {len(faces)} training images loaded.")
    return faces, labels

# Added 'log_func=print' here 
def train_model(faces, labels, log_func=print):
    log_func(f"[Logic] Training LBPH Model on {len(faces)} images...")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    model.save(MODEL_FILE)
    log_func("[Logic] Model saved successfully.")
    return model

# Added 'log_func=print' here
def load_saved_model(log_func=print):
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(MODEL_FILE)
        log_func("[Logic] Saved model loaded.")
        return model
    except:
        return None

# Added 'log_func=print' here 
def simulate_learning_curve_data(faces, labels, log_func=print):
    combined = list(zip(faces, labels))
    random.shuffle(combined)
    shuffled_faces, shuffled_labels = zip(*combined)
    split_idx = int(len(shuffled_faces) * 0.8)
    train_faces, train_labels = shuffled_faces[:split_idx], shuffled_labels[:split_idx]
    val_faces, val_labels = shuffled_faces[split_idx:], shuffled_labels[split_idx:]

    steps = [0.2, 0.5, 0.8, 1.0]
    train_accs = []
    val_accs = []
    valid_steps = []

    for percent in steps:
        size = int(len(train_faces) * percent)
        if size == 0: continue
        
        sub_faces = train_faces[:size]
        sub_labels = train_labels[:size]
        
        log_func(f"[Sim] Training on {int(percent*100)}% ({size} images)...")
        tmp_model = cv2.face.LBPHFaceRecognizer_create()
        try:
            tmp_model.train(sub_faces, np.array(sub_labels))
            
            c_t = sum(1 for i in range(len(sub_faces)) if tmp_model.predict(sub_faces[i])[0] == sub_labels[i])
            c_v = sum(1 for i in range(len(val_faces)) if tmp_model.predict(val_faces[i])[0] == val_labels[i])
            
            train_accs.append((c_t/len(sub_faces))*100)
            val_accs.append((c_v/len(val_faces))*100)
            valid_steps.append(percent)
        except: continue
        
    return valid_steps, train_accs, val_accs

def start_webcam_logic(model):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE) 
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        img_copy = frame.copy()
        face_roi, rect = process_image(frame, face_cascade)
        
        if face_roi is not None:
            face_resized = cv2.resize(face_roi, (200, 200))
            label_id, confidence = model.predict(face_resized)
            
            mood = SUBJECTS[label_id] if label_id < len(SUBJECTS) else "Unknown"
            color = (0, 255, 0) if confidence < 90 else (0, 165, 255)
            
            if rect is not None:
                (x, y, w, h) = rect
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img_copy, f"{mood} ({int(confidence)})", (x, y-10), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        
        cv2.imshow("Mood Detector (Press 'q' to quit)", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()