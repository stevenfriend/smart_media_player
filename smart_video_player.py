import cv2

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_face = frame[y : y + h, x : x + w]
        cropped_face_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(cropped_face_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            curr_sate = True
            cv2.rectangle(cropped_face, (ex, ey),
                (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Face detection', frame)

    key_e = cv2.waitKey(10)
    win_e = cv2.getWindowProperty('Face detection', 1)
    if key_e == ord('q') or win_e == -1:
        cap.release()
        cv2.destroyAllWindows()
        break
