import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('drowsiness_detection.h5')

# Define drowsiness detection function
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.astype('float') / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=3)
        prediction = model.predict(face_roi)[0]
        
        if prediction < 0.5:
            label = 'Awake'
            color = (0, 255, 0)
        else:
            label = 'Drowsy'
            color = (0, 0, 255)
            
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
    return frame

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_drowsiness(frame)
    
    cv2.imshow('Driver Drowsiness Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
