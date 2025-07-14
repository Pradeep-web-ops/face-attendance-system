import streamlit as st
import os
import cv2
import numpy as np
import pickle
import csv
import datetime
from PIL import Image
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Face Attendance System", layout="wide")

# === Sidebar ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1087/1087815.png", width=80)
st.sidebar.title("🎓 Smart Attendance")
st.sidebar.markdown("Made by **Pradeep** 💻")
st.sidebar.info("Upload a face image to record attendance.")
st.sidebar.markdown("---")

# === Title ===
st.markdown("<h1 style='text-align: center;'>📸 Face Recognition Attendance System</h1>", unsafe_allow_html=True)

# === Load Models ===
@st.cache_resource
def load_models():
    detector = cv2.dnn.readNetFromCaffe("model/deploy.prototxt", "model/res10_300x300_ssd_iter_140000.caffemodel")
    embedder = cv2.dnn.readNetFromTorch("archive/nn4.small2.v1.t7")
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
    return detector, embedder, recognizer, le

detector, embedder, recognizer, le = load_models()

# === Upload Image ===
uploaded_file = st.file_uploader("📤 Upload a face image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    frame = np.array(image.convert('RGB'))

    st.image(frame, caption="🖼️ Uploaded Image", use_column_width=True)

    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    found = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            found = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Get Roll No
            Roll_Number = "Unknown"
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    if name in row:
                        Roll_Number = row[row.index(name)+1]
                        break

            st.success(f"✅ Detected: **{name}** ({Roll_Number}) with **{proba*100:.2f}%** confidence")

            # Save to attendance_log.csv
            now = datetime.datetime.now()
            try:
                with open("attendance_log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, Roll_Number, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
            except PermissionError:
                st.error("⚠️ Please close attendance_log.csv before trying again.")
            break

    if not found:
        st.warning("⚠️ No face detected or confidence too low")

# === Attendance Viewer ===
st.markdown("---")
st.subheader("📄 View Attendance Log")

if os.path.exists("attendance_log.csv"):
    with open("attendance_log.csv", "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if rows:
            df = pd.DataFrame(rows, columns=["Name", "Roll No", "Date", "Time"])
            st.dataframe(df, use_container_width=True)

            # ✅ Download CSV
            csv_data = StringIO()
            df.to_csv(csv_data, index=False)
            st.download_button("⬇️ Download Attendance CSV", csv_data.getvalue(),
                               file_name="attendance_log.csv", mime="text/csv")

            # 🧹 Clear Button
            if st.button("🧹 Clear All Logs"):
                open("attendance_log.csv", "w").close()
                st.warning("🧾 Attendance log cleared!")
                st.rerun()

            # === 📊 Charts ===
            st.markdown("### 📊 Attendance Summary")

            count_df = df["Name"].value_counts().reset_index()
            count_df.columns = ["Name", "Count"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📌 Bar Chart")
                st.bar_chart(data=count_df.set_index("Name"))

            with col2:
                st.markdown("#### 🥧 Pie Chart")
                fig, ax = plt.subplots()
                ax.pie(count_df["Count"], labels=count_df["Name"], autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

        else:
            st.info("No records yet.")
else:
    st.info("📁 attendance_log.csv not found.")
