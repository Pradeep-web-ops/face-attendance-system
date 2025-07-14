# 📸 Face Recognition Based Attendance System

A real-time face recognition attendance system using Python, OpenCV, Deep Learning (FaceNet), and Machine Learning (SVM). It supports both GUI (Tkinter) and Web-based (Streamlit) user interfaces.

---

## 📁 Project Folder Structureattendence/
├── app.py # Streamlit Web App
├── gui_attendance_system.py # Tkinter GUI Application
├── 1_datasetCreation.py # Capture face images & create dataset
├── 2_preprocessingEmbeddings.py # Generate embeddings using FaceNet
├── 3_trainingFaceML.py # Train face recognition model
├── 5_recognizationPersonwithCSVDatabse.py # Recognize faces & log attendance
├── student.csv # Student name and roll number mapping
├── attendance_log.csv # Attendance record with timestamp
├── requirements.txt # All required Python libraries
├── archive/
│ └── nn4.small2.v1.t7 # FaceNet embedding model
├── model/
│ ├── deploy.prototxt
│ └── res10_300x300_ssd_iter_140000.caffemodel # Face detection model
├── output/
│ ├── embeddings.pickle
│ ├── recognizer.pickle
│ └── le.pickle
├── dataset/
│ └── [name]/*.png # Captured face images


## ✨ Features

- Real-time face detection and recognition using webcam 🎥
- Face embedding with pretrained FaceNet model
- Identity classification using SVM classifier
- Attendance recorded with Name, Roll No, Date, Time
- CSV file attendance storage
- Simple & intuitive UI:  
  - ✅ GUI using Tkinter  
  - ✅ Web app using Streamlit

---

## ⚙️ Technologies Used

- **Python**
- **OpenCV**
- **FaceNet (Torch model)**
- **Scikit-learn (SVM)**
- **Streamlit** (Web UI)
- **Tkinter** (Desktop GUI)
- **NumPy, Pandas, Pillow, CSV, Pickle**

---

## 🚀 How to Run

### ▶️ Run the GUI App:
```bash
python attendence/gui_attendance_system.py