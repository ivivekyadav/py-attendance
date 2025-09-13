# py-attendance
===============================================================================
                    FACE RECOGNITION ATTENDANCE SYSTEM
===============================================================================

OVERVIEW
--------
This is an automated attendance system that uses OpenCV-based face recognition
to detect and mark attendance for registered users. The system includes two
main components: user registration/management and automatic attendance marking.

FEATURES
--------
✓ Face registration with camera capture
✓ OpenCV-based face recognition (no external dependencies like dlib)
✓ Automatic attendance marking when faces are detected
✓ Duplicate prevention (won't mark same person twice per day)
✓ Real-time camera feed with face detection visualization
✓ CSV-based data storage for easy management
✓ Cooldown system to prevent spam detection

SYSTEM REQUIREMENTS
-------------------
- Python 3.7 or higher
- Webcam/Camera (USB or built-in)
- Windows/Linux/macOS

INSTALLATION
------------
1. Clone or download this project
2. Install dependencies:
   pip install -r requirements.txt
3. Run the system:
   python attendance.py  (for full system with menu)
   OR
   python mark_attendance.py  (for automatic attendance only)

FILE STRUCTURE
--------------
attendance.py          - Main system with registration and attendance
mark_attendance.py     - Automatic attendance marking system
requirements.txt       - Python dependencies
users.csv             - Registered users database
attendance.csv        - General attendance records
presentuser.csv       - Daily attendance records
known_faces/          - Directory containing face images
├── [roll]_[name].jpg - Face images for each registered user

HOW TO USE
----------

STEP 1: REGISTER USERS
Run: python attendance.py
- Select option 1 "Register a new face"
- Enter name and roll number
- Look at camera and press 'q' to capture face
- Face image will be saved in known_faces/ directory

STEP 2: START ATTENDANCE
Option A - Manual Attendance System:
- Run: python attendance.py
- Select option 2 "Start attendance"
- System will detect faces and mark attendance in attendance.csv

Option B - Automatic Attendance System:
- Run: python mark_attendance.py
- System automatically detects and marks attendance in presentuser.csv
- Press 'q' to quit, 's' to show today's attendance

CAMERA CONTROLS
---------------
- 'q' key: Quit/Exit
- 's' key: Show today's attendance (in mark_attendance.py)

OUTPUT FILES
------------
users.csv:
- Contains registered user information
- Columns: Name, Roll, Filename

attendance.csv:
- General attendance records from main system
- Columns: Name, Roll, Date, Time

presentuser.csv:
- Daily attendance from automatic system
- Columns: Name, Roll, Date, Time, Status

TROUBLESHOOTING
---------------

Camera Issues:
- If camera doesn't open, try changing camera index in code (0 or 1)
- Ensure no other application is using the camera

Face Recognition Issues:
- Ensure good lighting when registering faces
- Face should be clearly visible and centered
- System uses confidence threshold of 80-100 for recognition

Installation Issues:
- If opencv installation fails, try: pip install opencv-python --upgrade
- For Windows users, ensure Visual C++ redistributables are installed

No Face Detected:
- Check lighting conditions
- Ensure face is properly positioned in camera view
- Try re-registering the face with better image quality

TECHNICAL DETAILS
-----------------
- Uses OpenCV's Haar Cascade for face detection
- LBPH (Local Binary Pattern Histogram) for face recognition
- Confidence threshold: <80 for recognition, <100 for detection
- Face images standardized to 200x200 pixels
- 5-second cooldown between duplicate detections

CONFIGURATION
-------------
You can modify these settings in the code:

Camera Index:
- Change cv2.VideoCapture(1) to cv2.VideoCapture(0) if needed

Recognition Threshold:
- Modify confidence threshold in face recognition section
- Lower values = stricter recognition
- Higher values = more lenient recognition

Face Image Size:
- Change resize dimensions in face processing section

LIMITATIONS
-----------
- Works best with good lighting conditions
- Single face per person registration
- Requires clear, frontal face view for best results
- Performance depends on camera quality

SUPPORT
-------
For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure camera is working with other applications
4. Check file permissions for CSV files

VERSION HISTORY
---------------
v1.0 - Initial release with face_recognition library
v2.0 - Migrated to OpenCV-only solution for better compatibility
v2.1 - Added automatic attendance marking system
v2.2 - Added comprehensive documentation and requirements.txt

===============================================================================
                            END OF DOCUMENTATION
===============================================================================
