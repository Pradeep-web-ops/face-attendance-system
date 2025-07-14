from collections import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

embeddingFile = "output/embeddings.pickle"
embeddingModel = r"C:\Attendence sysytem\archive\nn4.small2.v1.t7"
recognizerPath = r"C:\Attendence sysytem\output\recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

print("[INFO] loading face detector...")
prototxt = r"C:\Attendence sysytem\model\deploy.prototxt"
model = r"C:\Attendence sysytem\model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerPath, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

ROLL_NUMBER = ""
box =[]
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

conf_threshold = 0.5

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

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

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            with open ('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    print("Box", box)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        print(name)
                listString = str(box)
##                    print(box)
                if name in listString:
                    singleList = list(flatten(box))
                    listlen = len(singleList)
                    print("listlen",listlen)
                    Index = singleList.index(name)
                    print("Index",Index)
                    name = singleList[Index]
                    Roll_Number = singleList[Index + 1]
                    print(Roll_Number)

##            text = "{}  :  {}   :  {: . 2f}%".format(name, Roll_Number, proba * 100)
##            y  = startY - 10 if startY - 10 > 10 else startY + 10
##            cv2.rectangle(frame, (startX, startY), (endX, endY),
##                          (0, 0, 255),  2)
##            cv2.putText(frame, text, (startX, y),
##                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),  2)

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0XFF
    if key == 27:
        break
                
cam.release()
cv2.destroyAllWindows()








