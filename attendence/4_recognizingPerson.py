
import cv2
import imutils
import pickle
import time
import os
import numpy as np

# Load face detector and embedder
print("[INFO] loading face detector...")
protoPath = r"C:\Attendence sysytem\model\deploy.prototxt"
modelPath = r"C:\Attendence sysytem\model\res10_300x300_ssd_iter_140000.caffemodel"
embedderPath = r"C:\Attendence sysytem\archive\nn4.small2.v1.t7"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch(embedderPath)

# Load the recognizer and label encoder
recognizerPath = r"C:\Attendence sysytem\output\recognizer.pickle"
lePath = r"C:\Attendence sysytem\output\le.pickle"

recognizer = pickle.loads(open(recognizerPath, "rb").read())
le = pickle.loads(open(lePath, "rb").read())

# Start video stream
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

conf_threshold = 0.5

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Prepare input blob for face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Get face embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Perform classification
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break

cam.release()
cv2.destroyAllWindows()
