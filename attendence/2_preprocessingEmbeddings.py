
from imutils import paths
import numpy as np
import pickle
import cv2
import os
import imutils  # missing import if using imutils.resize()

# Paths
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"
embeddingModel = r"C:\Attendence sysytem\archive\nn4.small2.v1.t7"  # check file name
prototxt = r"C:\Attendence sysytem\model\deploy.prototxt"
model = r"C:\Attendence sysytem\model/res10_300x300_ssd_iter_140000.caffemodel"

# Load models
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Init data
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5  # Confidence threshold

for (i, imagePath) in enumerate(imagePaths):
    print("Preprocessing image {}/{}".format(i + 1, len(imagePaths)))

    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Failed to load image: {imagePath}")
        continue

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Proceed if any face is detected
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]  # FIXED: typo (.) changed to (,)

            if fW < 20 or fH < 20:
                continue

            # Get embedding
            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False
            )
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print(f"[INFO] Total embeddings extracted: {total}")

# Save embeddings
print(f"[INFO] Saving embeddings to {embeddingFile}")
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Process Completed")
