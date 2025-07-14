
import cv2
import os
import csv
import time
import imutils

# Load Haar cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    print("Error: Could not load cascade classifier.")
    exit()

# Input
Name = input("Enter your name: ")
Roll_Number = input("Enter your Roll Number: ")

# Dataset folder
dataset_path = 'dataset'
user_path = os.path.join(dataset_path, Name)

# Create user folder if it doesn't exist
os.makedirs(user_path, exist_ok=True)
print(f"Folder created at: {user_path}")

# Add to CSV
if not os.path.exists('student.csv'):
    with open('student.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Roll Number'])

# Append new entry
with open('student.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([Name, Roll_Number])
    print("Student info saved.")

# Start camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started. Capturing face images...")
time.sleep(2.0)
total = 0

while total < 50:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        img_filename = os.path.join(user_path, f"{str(total).zfill(5)}.png")
        cv2.imwrite(img_filename, face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        total += 1
        print(f"Captured {total}/50")

        if total >= 50:
            break

    cv2.imshow("Capturing Faces - Press 'q' to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Face data collection completed.")



