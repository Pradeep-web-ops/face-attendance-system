from collections.abc import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
import datetime

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# Paths
embeddingFile = "output/embeddings.pickle"
embeddingModel = r"C:\Attendence sysytem\archive\nn4.small2.v1.t7"
recognizerPath = r"C:\Attendence sysytem\output\recognizer.pickle"
labelEncFile = "output/le.pickle"
prototxt = r"C:\Attendence sysytem\model\deploy.prototxt"
model = r"C:\Attendence sysytem\model/res10_300x300_ssd_iter_140000.caffemodel"

conf_threshold = 0.5

# Load models
print("[INFO] loading face detector and recognizer...")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerPath, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

# Start recognition loop
while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
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

            # Extract face embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Search roll number from CSV
            Roll_Number = "Unknown"
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    if name in row:
                        Index = row.index(name)
                        Roll_Number = row[Index + 1]
                        break

            # Save to attendance_log.csv with date and time
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            with open("attendance_log.csv", "a", newline="") as logfile:
                logwriter = csv.writer(logfile)
                logwriter.writerow([name, Roll_Number, date_str, time_str])

            # Draw bounding box and label
            text = "{} ({}): {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Attendance Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
