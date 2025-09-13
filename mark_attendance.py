import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_and_train_faces():
    """Load registered faces and train the recognizer"""
    if not os.path.exists('users.csv'):
        print("No users registered yet. Please register users first using attendance.py")
        return False, [], []

    # Load users
    df_users = pd.read_csv('users.csv')
    images = []
    classNames = []

    print("Loading registered faces...")
    for index, row in df_users.iterrows():
        filename = row['Filename']
        path = f'known_faces/{filename}'
        if not os.path.exists(path):
            print(f"Warning: Image file {filename} not found. Skipping user {row['Name']}.")
            continue
        
        img = cv2.imread(path)
        if img is not None and img.size > 0:
            print(f"Loading image {filename}")
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.ascontiguousarray(img, dtype=np.uint8)
                images.append(img)
                classNames.append(f"{row['Name']} ({row['Roll']})")
            else:
                print(f"Warning: Image {filename} has invalid format. Skipping user {row['Name']}.")
        else:
            print(f"Warning: Image {filename} is empty or could not be loaded. Skipping user {row['Name']}.")

    if not images:
        print("No valid face images found. Please register faces first.")
        return False, [], []

    # Train the recognizer
    faces = []
    labels = []
    
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)} for training")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect faces in the image
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(face_rects) > 0:
            # Use the first detected face
            (x, y, w, h) = face_rects[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))  # Standardize size
            
            faces.append(face_roi)
            labels.append(i)
            print(f"Face extracted and processed for {classNames[i]}")
        else:
            print(f"No face detected in image {i+1} for {classNames[i]}")
    
    if len(faces) > 0:
        print(f"Training recognizer with {len(faces)} faces...")
        recognizer.train(faces, np.array(labels))
        print("Training completed successfully!")
        return True, classNames, images
    else:
        print("No faces found for training!")
        return False, [], []

def mark_present(name, roll):
    """Mark a user as present in presentuser.csv"""
    # Create presentuser.csv if it doesn't exist
    if not os.path.exists('presentuser.csv'):
        df = pd.DataFrame(columns=['Name', 'Roll', 'Date', 'Time', 'Status'])
        df.to_csv('presentuser.csv', index=False)
    
    df = pd.read_csv('presentuser.csv')
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    # Check if user is already marked present today
    if not ((df['Name'] == name) & (df['Date'] == date) & (df['Status'] == 'Present')).any():
        new_entry = pd.DataFrame([[name, roll, date, time, 'Present']], 
                                columns=['Name', 'Roll', 'Date', 'Time', 'Status'])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv('presentuser.csv', index=False)
        print(f"✓ {name} marked PRESENT at {time}")
        return True
    else:
        print(f"→ {name} already marked present today")
        return False

def start_attendance_marking():
    """Start the camera and automatically mark attendance"""
    print("Starting automatic attendance marking system...")
    
    # Load and train faces
    training_success, classNames, images = load_and_train_faces()
    if not training_success:
        print("Failed to load and train faces. Exiting.")
        return
    
    # Start camera
    cap = cv2.VideoCapture(1)  # Try index 0 or 1
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback to index 0
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

    print("Camera started successfully!")
    print("Automatic attendance marking is now active.")
    print("Press 'q' to quit, 's' to show today's attendance")
    
    # Track recently detected faces to avoid spam
    recent_detections = {}
    detection_cooldown = 5  # seconds
    
    while True:
        success, img = cap.read()
        if not success or img is None or img.size == 0:
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = datetime.now()
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Predict using trained recognizer
            label, confidence = recognizer.predict(face_roi)
            
            # Lower confidence means better match (distance-based)
            if confidence < 80:  # Threshold for recognition
                name_roll = classNames[label]
                name, roll = name_roll.split(' (')
                roll = roll.rstrip(')')
                
                # Check cooldown to avoid spam detection
                last_detection = recent_detections.get(name, None)
                if last_detection is None or (current_time - last_detection).seconds >= detection_cooldown:
                    if mark_present(name, roll):
                        recent_detections[name] = current_time
                
                # Draw green rectangle for recognized face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"Conf: {confidence:.1f}", (x + 6, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # Unknown face - draw red rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add status text
        cv2.putText(img, "Automatic Attendance System", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Press 'q' to quit, 's' for today's attendance", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Automatic Attendance Marking', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_todays_attendance()

    cap.release()
    cv2.destroyAllWindows()

def show_todays_attendance():
    """Display today's attendance"""
    if not os.path.exists('presentuser.csv'):
        print("No attendance records found.")
        return
    
    df = pd.read_csv('presentuser.csv')
    today = datetime.now().strftime("%Y-%m-%d")
    todays_attendance = df[df['Date'] == today]
    
    if todays_attendance.empty:
        print(f"No attendance records for today ({today})")
    else:
        print(f"\n=== Today's Attendance ({today}) ===")
        for _, row in todays_attendance.iterrows():
            print(f"{row['Name']} (Roll: {row['Roll']}) - {row['Status']} at {row['Time']}")
        print(f"Total present: {len(todays_attendance)}")
        print("=" * 40)

if __name__ == "__main__":
    print("=== Automatic Attendance Marking System ===")
    print("This system will automatically detect and mark attendance")
    print("for registered users when they appear in the camera.")
    print()
    
    start_attendance_marking()
