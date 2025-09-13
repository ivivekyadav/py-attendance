import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ----------- Face Registration Function -----------

def register_face():
    name = input("Enter Name: ")
    roll = input("Enter Roll Number: ")

    # Create users.csv if it doesn't exist
    if not os.path.exists('users.csv'):
        df = pd.DataFrame(columns=['Name', 'Roll', 'Filename'])
        df.to_csv('users.csv', index=False)

    df = pd.read_csv('users.csv')
    if roll in df['Roll'].values:
        print("Roll number already registered.")
        return

    # Create known_faces directory if it doesn't exist
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')

    cap = cv2.VideoCapture(1)  # Try index 0 or 1 based on your camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Capturing face. Please look at the camera...")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to capture image.")
            break

        cv2.imshow("Register Face - Press 'q' to capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            count += 1
            filename = f'{roll}_{name}.jpg'
            path = f'known_faces/{filename}'
            cv2.imwrite(path, frame)
            print(f"Captured and saved {filename}")
            break

    cap.release()
    cv2.destroyAllWindows()

    new_entry = pd.DataFrame([[name, roll, filename]], columns=['Name', 'Roll', 'Filename'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv('users.csv', index=False)
    print("User registered successfully.")

# ----------- Attendance Function -----------

def start_attendance():
    if not os.path.exists('users.csv'):
        print("No users registered yet. Please register first.")
        return

    df_users = pd.read_csv('users.csv')
    images = []
    classNames = []

    # Load images and convert to RGB
    for index, row in df_users.iterrows():
        filename = row['Filename']
        path = f'known_faces/{filename}'
        if not os.path.exists(path):
            print(f"Warning: Image file {filename} not found. Skipping user {row['Name']}.")
            continue
        img = cv2.imread(path)
        if img is not None and img.size > 0:
            print(f"Loading image {filename} - Shape: {img.shape}, Type: {img.dtype}")
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Ensure the image array is contiguous and in the right format
                img = np.ascontiguousarray(img, dtype=np.uint8)
                images.append(img)
                classNames.append(f"{row['Name']} ({row['Roll']})")
            else:
                print(f"Warning: Image {filename} has invalid format. Skipping user {row['Name']}.")
        else:
            print(f"Warning: Image {filename} is empty or could not be loaded. Skipping user {row['Name']}.")

    if not images:
        print("No valid face images found. Please register first.")
        return

    # Train OpenCV face recognizer
    def trainRecognizer(images, classNames):
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
            return True
        else:
            print("No faces found for training!")
            return False

    # Train the recognizer with the loaded images
    training_success = trainRecognizer(images, classNames)
    if not training_success:
        print("Failed to train face recognizer. Cannot proceed with attendance.")
        return
    print('Face recognizer trained successfully.')

    if not os.path.exists('attendance.csv'):
        pd.DataFrame(columns=['Name', 'Roll', 'Date', 'Time']).to_csv('attendance.csv', index=False)

    def markAttendance(name, roll):
        df = pd.read_csv('attendance.csv')
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        if not ((df['Name'] == name) & (df['Date'] == date)).any():
            new_entry = pd.DataFrame([[name, roll, date, time]], columns=['Name', 'Roll', 'Date', 'Time'])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv('attendance.csv', index=False)
            print(f"{name} marked present at {time}")

    cap = cv2.VideoCapture(1)  # Try index 0 or 1
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting attendance system. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success or img is None or img.size == 0:
            print("Failed to capture frame from camera.")
            continue

        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Predict using trained recognizer
            label, confidence = recognizer.predict(face_roi)
            
            # Lower confidence means better match (distance-based)
            if confidence < 100:  # Threshold for recognition
                name_roll = classNames[label]
                name, roll = name_roll.split(' (')
                roll = roll.rstrip(')')
                
                # Draw rectangle and name
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{name} ({confidence:.1f})", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                markAttendance(name, roll)
            else:
                # Unknown face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Attendance System', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------- Main Menu -----------

def main():
    while True:
        print("\nAttendance System Menu")
        print("1. Register a new face")
        print("2. Start attendance")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            register_face()
        elif choice == '2':
            start_attendance()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
