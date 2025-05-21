import tkinter as tk
from tkinter import filedialog, Toplevel, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import subprocess

# === Load trained models and tools ===
model_dir = "models"
voting_model = joblib.load(os.path.join(model_dir, "voting_model.pkl"))
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
pca = joblib.load(os.path.join(model_dir, "pca.pkl"))

model_accuracy = 0.74
IMG_SIZE = 224

# === Build feature extractor once ===
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# === Prediction Function ===
def classify_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = feature_extractor.predict(image)
    features = scaler.transform(features)
    features = pca.transform(features)

    prediction = voting_model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# === Run prediction with loading popup ===
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        loading = Toplevel(root)
        loading.title("Please Wait")
        loading.configure(bg="#ffffff")

        width, height = 300, 100
        loading.update_idletasks()
        screen_width = loading.winfo_screenwidth()
        screen_height = loading.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        loading.geometry(f"{width}x{height}+{x}+{y}")

        Label(loading, text="⏳ Predicting...", font=("Arial", 12), bg="#ffffff").pack(pady=30)
        loading.update()

        root.after(100, lambda: run_prediction(file_path, loading))

def run_prediction(file_path, loading_window):
    label = classify_image(file_path)
    loading_window.destroy()
    result_label.config(text=f"Predicted Tumor Type: {label}")

# === Return to main menu
def back_to_main_menu():
    root.destroy()
    subprocess.Popen(["python", "main_menu.py"])

# === Create Window ===
root = tk.Tk()
root.title("Brain Tumor Classification")
root.state('zoomed')
root.configure(bg="#f4f4f4")

# === Layout frames
top_frame = tk.Frame(root, bg="#f4f4f4")
top_frame.pack(pady=20)

bottom_frame = tk.Frame(root, bg="#f4f4f4")
bottom_frame.pack(pady=10)

# === Title and Accuracy
tk.Label(top_frame, text="🧠 Brain Tumor Classifier (Voting Model)", font=("Arial", 22, "bold"), bg="#f4f4f4", fg="#333").pack(pady=5)
tk.Label(top_frame, text=f"Model Accuracy: {model_accuracy*100:.2f}%", font=("Arial", 14), bg="#f4f4f4", fg="#444").pack()

# === Upload button
tk.Button(bottom_frame, text="📁 Upload MRI Image", command=load_image,
          font=("Arial", 14), width=30, bg="#007acc", fg="white", activebackground="#005f99").pack(pady=20)

# === Image preview
panel = tk.Label(bottom_frame, bg="#f4f4f4")
panel.pack()

# === Prediction label
result_label = tk.Label(bottom_frame, text="Predicted Tumor Type: ", font=("Arial", 16), bg="#f4f4f4", fg="#111")
result_label.pack(pady=20)

# === Back button
tk.Button(bottom_frame, text="⬅ Back to Main Menu", command=back_to_main_menu,
          font=("Arial", 12), width=25, bg="#999", fg="white", activebackground="#666").pack(pady=10)

# === Start GUI
root.mainloop()
