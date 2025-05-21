import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import numpy as np
import os
import pandas as pd
import subprocess
import sys

# --- Load model and scaler ---
MODEL_DIR = r"C:\Users\anasr\Desktop\machine learnng\alz\alz_models"
regressor = joblib.load(os.path.join(MODEL_DIR, "stacked_regressor.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

feature_names = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", "AlcoholConsumption",
    "PhysicalActivity", "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers",
    "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension",
    "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "MemoryComplaints",
    "BehavioralProblems", "ADL", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness"
]

# --- Window Setup ---
root = tk.Tk()
root.title("🧠 Alzheimer's Risk Prediction")
root.state('zoomed')  # Fullscreen for Windows

def return_to_main_menu():
    root.destroy()
    if getattr(sys, 'frozen', False):
        subprocess.Popen(["main_menu.py"])
    else:
        subprocess.Popen([sys.executable, "main_menu.py"])

def on_close():
    return_to_main_menu()

root.protocol("WM_DELETE_WINDOW", on_close)

# --- Title ---
header = tk.Label(root, text="Alzheimer's Risk Probability Predictor", font=("Helvetica", 20, "bold"), fg="darkblue")
header.pack(pady=10)

# --- Layout Frames ---
main_frame = ttk.Frame(root)
main_frame.pack(padx=20, pady=10, expand=True)

left_frame = ttk.Frame(main_frame)
right_frame = ttk.Frame(main_frame)

left_frame.grid(row=0, column=0, padx=25, sticky="n")
right_frame.grid(row=0, column=1, padx=25, sticky="n")

entries = {}
mid_index = len(feature_names) // 2
left_features = feature_names[:mid_index]
right_features = feature_names[mid_index:]

for i, name in enumerate(left_features):
    ttk.Label(left_frame, text=name, width=25).grid(row=i, column=0, sticky="w", pady=2)
    e = ttk.Entry(left_frame, width=20)
    e.grid(row=i, column=1, pady=2)
    entries[name] = e

for i, name in enumerate(right_features):
    ttk.Label(right_frame, text=name, width=25).grid(row=i, column=0, sticky="w", pady=2)
    e = ttk.Entry(right_frame, width=20)
    e.grid(row=i, column=1, pady=2)
    entries[name] = e

# --- Button Functions ---
def predict():
    try:
        values = [float(entries[name].get()) for name in feature_names]
        features = np.array(values).reshape(1, -1)
        scaled = scaler.transform(features)
        prob = regressor.predict(scaled)[0]

        result = f"Predicted Risk Probability: {prob:.3f}\n"
        if prob >= 0.7:
            result += "🔴 High Risk of Alzheimer's"
        elif prob >= 0.4:
            result += "🟠 Moderate Risk"
        else:
            result += "🟢 Low Risk"

        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Input Error", f"Please enter valid numerical values.\n\n{e}")

def load_from_file():
    path = filedialog.askopenfilename(filetypes=[("Text/CSV Files", "*.txt *.csv")])
    if not path:
        return
    try:
        if path.endswith(".txt"):
            with open(path, "r") as f:
                values = [float(x.strip()) for x in f.readlines()]
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            if df.shape[0] > 1:
                df = df.iloc[0:1]
            values = df.iloc[0].tolist()
        else:
            raise ValueError("Unsupported file type")

        if len(values) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} values, got {len(values)}")

        for name, val in zip(feature_names, values):
            entries[name].delete(0, tk.END)
            entries[name].insert(0, str(val))

        messagebox.showinfo("Loaded", "Values loaded successfully from file.")
    except Exception as e:
        messagebox.showerror("File Error", f"Error loading file:\n\n{e}")

# --- Buttons ---
button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

ttk.Button(button_frame, text="📂 Load from File", command=load_from_file).pack(side=tk.LEFT, padx=15)
ttk.Button(button_frame, text="🧠 Predict Risk", command=predict).pack(side=tk.LEFT, padx=15)
ttk.Button(button_frame, text="🔙 Back to Main Menu", command=return_to_main_menu).pack(side=tk.LEFT, padx=15)

root.mainloop()
