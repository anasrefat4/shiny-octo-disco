import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import joblib
import json
import os
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
import parselmouth
from parselmouth.praat import call
import subprocess

# Load feature list
model_dir = "modelsp"  # Folder where all model files are stored

with open(os.path.join(model_dir, "feature_list.json"), "r") as f:
    feature_names = json.load(f)

# Load scaler
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

# Define model file paths
model_paths = {
    "Random Forest": os.path.join(model_dir, "rf_classifier.pkl"),
    "Logistic Regression": os.path.join(model_dir, "logreg_classifier.pkl"),
    "SVM": os.path.join(model_dir, "svm_classifier.pkl"),
    "KNN": os.path.join(model_dir, "knn_classifier.pkl"),
    "AdaBoost": os.path.join(model_dir, "adaboost_classifier.pkl"),
    "MLP (Keras)": os.path.join(model_dir, "parkinsons_mlp_model.h5")
}

# Extract features from a WAV file
def extract_features_from_wav(path):
    snd = parselmouth.Sound(path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    hnr_obj = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(hnr_obj, "Get mean", 0, 0)

    return {
        "Jitter(%)": call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
        "Jitter(Abs)": call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3),
        "Jitter:RAP": call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
        "Jitter:PPQ5": call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
        "Jitter:DDP": 3 * call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
        "Shimmer": call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "Shimmer(dB)": call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "Shimmer:APQ3": call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "Shimmer:APQ5": call([snd, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "Shimmer:APQ11": call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "Shimmer:DDA": 3 * call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "NHR": 1 / (hnr + 1e-6),
        "HNR": hnr,
        "RPDE": 0.81, "DFA": 1.34, "PPE": 0.005, "age": 65.0, "sex": 1.0
    }

# Prediction logic
def predict(model_name, inputs):
    model_path = model_paths[model_name]
    if "mlp" in model_path:
        model = load_model(model_path)
    else:
        model = joblib.load(model_path)

    scaled_input = scaler.transform([inputs])
    if "mlp" in model_path:
        prob = model.predict(scaled_input)[0][0]
    else:
        prob = model.predict_proba(scaled_input)[0][1]

    result = "Likely Parkinson’s" if prob >= 0.5 else "Likely Healthy"
    return result, prob

# Main App GUI
class ParkinsonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parkinson's Predictor")
        self.geometry("800x600")
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle close button

    def create_widgets(self):
        self.model_var = tk.StringVar(value="Random Forest")
        tk.Label(self, text="Select Model:").pack()
        ttk.Combobox(self, textvariable=self.model_var, values=list(model_paths.keys())).pack()

        self.entries = {}
        frame = tk.Frame(self)
        frame.pack()
        for i, feature in enumerate(feature_names):
            tk.Label(frame, text=feature).grid(row=i, column=0, sticky="w")
            entry = tk.Entry(frame)
            entry.grid(row=i, column=1)
            self.entries[feature] = entry

        tk.Button(self, text="Predict from Manual Input", command=self.manual_predict).pack(pady=10)
        tk.Button(self, text="Upload and Predict from Voice File", command=self.upload_voice).pack(pady=5)
        tk.Button(self, text="🎙️ Record from Microphone", command=self.record_voice).pack(pady=5)
        tk.Button(self, text="⬅️ Back to Main Menu", command=self.back_to_main_menu).pack(pady=10)

        self.result_label = tk.Label(self, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def manual_predict(self):
        try:
            input_vals = [float(self.entries[f].get()) for f in feature_names]
            label, prob = predict(self.model_var.get(), input_vals)
            self.result_label.config(text=f"{label} ({prob*100:.2f}%)")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def upload_voice(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filepath:
            self.predict_from_path(filepath)

    def record_voice(self):
        fs = 16000
        duration = 5
        try:
            messagebox.showinfo("Recording", "Recording will start. Speak now...")
            sd.default.device = (2, None)
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            filename = "recorded_voice.wav"
            write(filename, fs, recording)
            messagebox.showinfo("Done", f"Recording saved as {filename}")
            self.predict_from_path(filename)
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))

    def predict_from_path(self, path):
        try:
            features = extract_features_from_wav(path)
            input_vals = [features[f] for f in feature_names]
            label, prob = predict(self.model_var.get(), input_vals)
            self.result_label.config(text=f"{label} ({prob*100:.2f}%)")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def back_to_main_menu(self):
        self.destroy()
        subprocess.Popen(["python", "main_menu.py"])

    def on_closing(self):
        self.destroy()
        subprocess.Popen(["python", "main_menu.py"])

# Run the app
if __name__ == "__main__":
    app = ParkinsonApp()
    app.state('zoomed')  # Fullscreen
    app.mainloop()
