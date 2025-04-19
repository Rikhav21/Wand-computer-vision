import cv2
import numpy as np
import os
import uuid
def record():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        time = cv2.getTickCount()
        if (time - start_time) / cv2.getTickFrequency() > 2:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def reds(video_path):
    cap = cv2.VideoCapture(video_path)
    red = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            bigge = max(contours, key=cv2.contourArea)
            M = cv2.moments(bigge)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                ped.append((x, y))
        for i in range(1, len(red)):
            cv2.line(frame, red[i-1], red[i], (0, 0, 255), thickness=5)
        cv2.imshow("Red Path", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    imagify(red)
    cap.release()
    cv2.destroyAllWindows()
def imagify(red):
    os.makedirs("None", exist_ok=True)
    filename = os.path.join("None", str(uuid.uuid4())[:8] + ".png")
    blakx = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, len(red)):
        cv2.line(blakx, red[i-1], red[i], (255, 255, 255), thickness=5)
    cv2.imwrite(filename, blakx)

record()
reds("output.avi")
