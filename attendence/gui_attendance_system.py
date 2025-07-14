import tkinter as tk
from tkinter import messagebox
import os
import subprocess
import cv2
import PIL.Image, PIL.ImageTk
import threading

# === Admin Credentials ===
ADMIN_USERNAME = "pradeep"
ADMIN_PASSWORD = "ikdp2314"

# === Main Attendance GUI Class ===
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Face Recognition Attendance System")
        self.root.geometry("800x650")
        self.root.configure(bg="#f0f0f0")

        title = tk.Label(self.root, text="📸 Face Recognition Attendance System", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        title.pack(pady=10)

        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        btn_frame = tk.Frame(root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        btn1 = tk.Button(btn_frame, text="➕ Register New Face", font=("Arial", 14), bg="#4CAF50", fg="white", command=self.register_face)
        btn1.grid(row=0, column=0, padx=10, pady=10, ipadx=10, ipady=5)

        btn2 = tk.Button(btn_frame, text="🤖 Start Attendance", font=("Arial", 14), bg="#2196F3", fg="white", command=self.start_recognition)
        btn2.grid(row=0, column=1, padx=10, pady=10, ipadx=10, ipady=5)

        btn3 = tk.Button(btn_frame, text="📄 View Attendance CSV", font=("Arial", 14), bg="#FF5722", fg="white", command=self.open_csv)
        btn3.grid(row=0, column=2, padx=10, pady=10, ipadx=10, ipady=5)

        footer = tk.Label(root, text="Made by Pradeep 💻", bg="#f0f0f0", font=("Helvetica", 10))
        footer.pack(side="bottom", pady=20)

        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
        self.root.after(10, self.update_video)

    def register_face(self):
        threading.Thread(target=lambda: self.run_script("1_datasetCreation.py")).start()

    def start_recognition(self):
        threading.Thread(target=lambda: self.run_script("5_recognizationPersonwithCSVDatabse.py")).start()

    def open_csv(self):
        try:
            os.startfile("attendance_log.csv")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_script(self, script_name):
        try:
            subprocess.run(["python", script_name])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# === Admin Login Window ===
def show_login():
    login_window = tk.Tk()
    login_window.title("🔐 Admin Login")
    login_window.geometry("400x250")
    login_window.configure(bg="#f0f0f0")

    tk.Label(login_window, text="Admin Login", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)

    tk.Label(login_window, text="Username", font=("Arial", 12), bg="#f0f0f0").pack()
    username_entry = tk.Entry(login_window, font=("Arial", 12))
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password", font=("Arial", 12), bg="#f0f0f0").pack()
    password_entry = tk.Entry(login_window, show="*", font=("Arial", 12))
    password_entry.pack(pady=5)

    def validate_login():
        username = username_entry.get()
        password = password_entry.get()
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            login_window.destroy()
            root = tk.Tk()
            app = AttendanceApp(root)
            root.mainloop()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")

    tk.Button(login_window, text="Login", font=("Arial", 12), bg="#4CAF50", fg="white", command=validate_login).pack(pady=20)

    login_window.mainloop()

# === Start Here ===
if __name__ == "__main__":
    show_login()
