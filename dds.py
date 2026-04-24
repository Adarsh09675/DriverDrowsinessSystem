import cv2
import dlib
from scipy.spatial import distance
import time
import os
import winsound  # Specific to Windows for the alarm beep

# --------- Constants ----------
EYE_AR_THRESH = 0.25       # EAR threshold to detect closed eyes
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames eyes must be below threshold
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# --------- Counters ----------
COUNTER = 0

# --------- Load Models ----------
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found!")
    print("Please ensure the file is in the project directory.")
    exit()

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# --------- Eye Landmark Indices ----------
# dlib 68-point landmarks: left eye is 36-41, right eye is 42-47
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# --------- Helper Functions ----------
def eye_aspect_ratio(eye):
    """Compute the Eye Aspect Ratio (EAR)"""
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_list(shape):
    """Convert dlib shape object to a list of (x, y) coordinates"""
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

# --------- Start Video Stream ----------
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0) # Warmup

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    frame = cv2.flip(frame, 1) # Flip for mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for face in faces:
        shape = predictor(gray, face)
        shape_list = shape_to_list(shape)

        left_eye = [shape_list[i] for i in LEFT_EYE]
        right_eye = [shape_list[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye contours
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # UI Overlay
        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check if driver is drowsy
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                # Sound alarm
                winsound.Beep(1000, 100) # Frequency, Duration
        else:
            COUNTER = 0
            cv2.putText(frame, "Status: Awake", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    
    # Check if the window was closed via the "X" button
    if cv2.getWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Exit on 'q' or 'ESC'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
