import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import joblib
import os
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import time
import subprocess  # ✅ for launching main_menu.py

# --- SETTINGS ---
MODEL_DIR = "newmodels"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
FEATURES_TXT = "mic_features.txt"

feature_names = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "PPE", "Intensity", "Final PointProcess"
]

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Classifier label logic
def interpret_updrs(value):
    if value < 20:
        return "🟢 Healthy"
    elif value < 40:
        return "🟡 Mild Parkinsonism"
    elif value < 60:
        return "🟠 Moderate Parkinsonism"
    else:
        return "🔴 Severe Parkinsonism"

# Record only and save as WAV
def record_voice_only():
    def record():
        file_path = "mic_input.wav"
        recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        for i in range(RECORD_SECONDS, 0, -1):
            countdown_label.config(text=f"⏳ Recording... {i} seconds left")
            time.sleep(1)
        sd.wait()
        countdown_label.config(text="✅ Recording complete")
        write(file_path, SAMPLE_RATE, recording)
        messagebox.showinfo("Saved", f"Recording saved as {file_path}. Please upload features to proceed.")
    threading.Thread(target=record).start()

# Predict from .txt file
def predict_from_txt_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    try:
        with open(file_path, 'r') as f:
            values = list(map(float, f.read().strip().split()))
            if len(values) != 18:
                raise ValueError("Expected 18 features in the text file.")
            scaled = scaler.transform([values])
            prediction = model.predict(scaled)[0]
            label = interpret_updrs(prediction)
            result_window = tk.Toplevel()
            result_window.title("Prediction Result")
            result_window.geometry("300x150")
            result_window.update_idletasks()
            x = (result_window.winfo_screenwidth() - result_window.winfo_reqwidth()) // 2
            y = (result_window.winfo_screenheight() - result_window.winfo_reqheight()) // 2
            result_window.geometry(f"+{x}+{y}")
            tk.Label(result_window, text=f"Predicted motor_UPDRS: {prediction:.2f}\nStatus: {label}", font=("Arial", 12)).pack(pady=30)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read or predict: {e}")

# Manual input
def predict_from_text_input():
    try:
        values = [float(e.get()) for e in manual_entries]
        if len(values) != 18:
            raise ValueError("Exactly 18 features are required.")
        scaled = scaler.transform([values])
        prediction = model.predict(scaled)[0]
        label = interpret_updrs(prediction)
        result_window = tk.Toplevel()
        result_window.title("Prediction Result")
        result_window.geometry("300x150")
        result_window.update_idletasks()
        x = (result_window.winfo_screenwidth() - result_window.winfo_reqwidth()) // 2
        y = (result_window.winfo_screenheight() - result_window.winfo_reqheight()) // 2
        result_window.geometry(f"+{x}+{y}")
        tk.Label(result_window, text=f"Predicted motor_UPDRS: {prediction:.2f}\nStatus: {label}", font=("Arial", 12)).pack(pady=30)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Function to return to main menu
def back_to_main_menu():
    root.destroy()
    subprocess.Popen(["python", "main_menu.py"])

# GUI
root = tk.Tk()
root.title("Parkinson's Voice Predictor")
root.state('zoomed')  # Windows maximized view
root.protocol("WM_DELETE_WINDOW", back_to_main_menu)  # ✅ Return to main menu when closing

# Layout
outer_frame = tk.Frame(root)
outer_frame.pack(expand=True, fill='both')

canvas = tk.Canvas(outer_frame)
scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

scrollable_frame = tk.Frame(canvas)

content_frame = tk.Frame(scrollable_frame)
content_frame.pack(expand=True)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((650, 0), window=scrollable_frame, anchor="n")

frame = content_frame

tk.Label(frame, text="", height=2).pack()

label = tk.Label(frame, text="Parkinson's UPDRS Predictor\nUpload, Record or Manually Enter Features", font=("Arial", 18))
label.pack(pady=10)

upload_btn = tk.Button(frame, text="📁 Upload Voice File (.wav)", font=("Arial", 12), command=lambda: messagebox.showinfo("Note", "Feature extraction is disabled. Upload a .txt file instead."))
upload_btn.pack(pady=5)

record_btn = tk.Button(frame, text="🎙️ Record Voice (WAV only)", font=("Arial", 12), command=record_voice_only)
record_btn.pack(pady=5)

countdown_label = tk.Label(frame, text="", font=("Arial", 12))
countdown_label.pack(pady=5)

tk.Label(frame, text="Manual Feature Input (18 values)", font=("Arial", 12)).pack(pady=5)
manual_entries = []

input_frame = tk.Frame(frame)
input_frame.pack(pady=5)

left_column = tk.Frame(input_frame)
right_column = tk.Frame(input_frame)
left_column.pack(side=tk.LEFT, padx=20)
right_column.pack(side=tk.LEFT, padx=20)

for i, name in enumerate(feature_names):
    col = left_column if i < 9 else right_column
    row = tk.Frame(col)
    row.pack(pady=1)
    label = tk.Label(row, text=name, font=("Arial", 9), width=18, anchor='w')
    label.pack(side=tk.LEFT)
    entry = tk.Entry(row, width=8, font=("Arial", 10))
    entry.pack(side=tk.LEFT)
    manual_entries.append(entry)

predict_manual_btn = tk.Button(frame, text="📊 Predict from Manual Input", font=("Arial", 12), command=predict_from_text_input)
predict_manual_btn.pack(pady=5)

load_txt_btn = tk.Button(frame, text="📄 Predict from .txt File (18 values)", font=("Arial", 12), command=predict_from_txt_file)
load_txt_btn.pack(pady=5)

# ✅ Back to main menu button
back_btn = tk.Button(frame, text="⬅️ Back to Main Menu", font=("Arial", 12), command=back_to_main_menu)
back_btn.pack(pady=5)

quit_btn = tk.Button(frame, text="❌ Quit", font=("Arial", 12), command=root.destroy)
quit_btn.pack(pady=10)

root.mainloop()
