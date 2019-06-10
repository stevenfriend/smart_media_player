import cv2
import keyboard

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

prev_state = False
curr_state = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    prev_state = curr_state
    curr_state = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_face = frame[y : y + h, x : x + w]
        cropped_face_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(cropped_face_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            curr_state = True
            cv2.rectangle(cropped_face, (ex, ey),
                (ex + ew, ey + eh), (0, 255, 0), 2)
        if prev_state == False and curr_state == True:
            keyboard.send('ctrl+shift+a')
        if prev_state == True and curr_state == False:
            keyboard.send('ctrl+shift+q')

    cv2.imshow('Face detection', frame)

    key_e = cv2.waitKey(10)
    win_e = cv2.getWindowProperty('Face detection', 1)
    if key_e == ord('q') or win_e == -1:
        cap.release()
        cv2.destroyAllWindows()
        break
