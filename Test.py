import cv2
import numpy as np
from keras.models import load_model
import time
import screen_brightness_control
SET_MONITOR_BRIGHTNESS = 0x1007CC
def track_red_light():
    cap = cv2.VideoCapture(0)
    reds = []
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lows = np.array([0, 100, 100])
        ups = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lows, ups)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            begge = max(contours, key=cv2.contourArea)
            M = cv2.moments(begge)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                reds.append((centroid_x, centroid_y))
        for i in range(1, len(reds)):
            cv2.line(frame, reds[i - 1], reds[i], (0, 0, 255), thickness=5)
        cv2.imshow("Red Path", frame)
        if time.time() - start_time > 2:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return reds
def clss(reds):
    black_background = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, len(reds)):
        cv2.line(black_background, reds[i - 1], reds[i], (255, 255, 255), thickness=5)
    input = cv2.resize(black_background, (100, 100))
    input = np.expand_dims(input, axis=0)
    model = load_model("classify.keras")
    prediction = model.predict(input)
    classes = ['lumos', 'nox', 'None']
    inx = np.argmax(prediction)
    result = classes[inx]
    return result
def change(classification_result):
    if classification_result == 'lumos':
        screen_brightness_control.set_brightness(100)
    elif classification_result == 'nox':
        screen_brightness_control.set_brightness(20)
    else:
        return
print("Tracking...")
reds = track_red_light()
print("Classifying ...")
classification_result = clss(reds)
print("Classification :", classification_result)
print("Adjusting ...")
change(classification_result)