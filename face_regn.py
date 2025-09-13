import cv2
import os
import pandas as pd

def register_face():
    name = input("Enter Name: ")
    roll = input("Enter Roll Number: ")

    # Create users.csv if it doesn't exist
    if not os.path.exists('users.csv'):
        df = pd.DataFrame(columns=['Name', 'Roll', 'Filename'])
        df.to_csv('users.csv', index=False)

    # Check if roll number already exists
    df = pd.read_csv('users.csv')
    if roll in df['Roll'].values:
        print("Roll number already registered.")
        return

    # Create known_faces directory if not exists
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')

    cap = cv2.VideoCapture(0)
    print("Capturing face. Please look at the camera...")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
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

    # Save to users.csv
    new_entry = pd.DataFrame([[name, roll, filename]], columns=['Name', 'Roll', 'Filename'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv('users.csv', index=False)
    print("User registered successfully.")
