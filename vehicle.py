import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

min_width_rect = 80
min_height_rect = 80
count_line_position = 550

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

detect = []
offset = 6
counter = 0

def center_handle(x,y,w,h):
    cx = int(x + w/2)
    cy = int(y + h/2)
    return cx, cy

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255,127,0), 3)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_width_rect and h >= min_height_rect:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Vehicle "+str(counter), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,244,0), 2)
            center = center_handle(x,y,w,h)
            detect.append(center)
            cv2.circle(frame, center, 4, (0,0,255), -1)

    for (x,y) in detect[:]: # Iterate over a copy to avoid removal issues
        if count_line_position-offset < y < count_line_position+offset:
            counter += 1
            cv2.line(frame, (25,count_line_position), (1200,count_line_position), (0,127,255), 3)
            detect.remove((x,y))
            print("Vehicle Counter: " + str(counter))

    cv2.putText(frame, "VEHICLE COUNTER: " + str(counter), (450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    cv2.imshow('Vehicle Detection', frame)

    # Check if the window was closed via the "X" button
    if cv2.getWindowProperty('Vehicle Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 or key == 13: # q, ESC, or Enter to exit
        break

cap.release()
cv2.destroyAllWindows()
