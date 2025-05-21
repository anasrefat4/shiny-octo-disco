import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
import os
import subprocess
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image

# Load DenseNet169 feature extractor
base_model = DenseNet169(weights='imagenet', include_top=False, pooling='avg')

# Class map (must match your training labels)
class_map = {
    0: 'Basal Cell Carcinoma',
    1: 'Dermatofibroma',
    2: 'Melanoma',
    3: 'Nevus',
    4: 'Pigmented Benign Keratosis',
    5: 'Squamous Cell Carcinoma',
    6: 'Vascular Lesion'
}

# Load all models
models = {
    "Random Forest": joblib.load("modelsskin\model_rf_7class.pkl"),
    "SVM": joblib.load("modelsskin\model_svm_7class.pkl"),
    "KNN": joblib.load("modelsskin\model_knn_7class.pkl"),
    "XGBoost": joblib.load("modelsskin\model_xgb_7class.pkl"),
    "Voting Classifier": joblib.load("modelsskin\model_voting_7class.pkl")
}

# Extract features from image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    feat = base_model.predict(x)
    return feat.flatten()

# GUI
class SkinCancerApp:
    def __init__(self, master):
        
        self.master = master
        self.master.title("Skin Cancer Classifier (7-Class)")
        self.master.geometry("600x500")
        self.master.state('zoomed')
        # Title
        tk.Label(master, text="🧬 Skin Cancer Classifier", font=("Arial", 18, "bold")).pack(pady=10)

        self.label = tk.Label(master, text="Choose an image:", font=("Arial", 14))
        self.label.pack(pady=5)

        self.img_label = tk.Label(master)
        self.img_label.pack()

        self.upload_btn = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)

        self.model_var = tk.StringVar()
        self.model_var.set("Voting Classifier")
        self.model_menu = tk.OptionMenu(master, self.model_var, *models.keys())
        self.model_menu.pack(pady=5)

        self.predict_btn = tk.Button(master, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=10)

        # Loading label
        self.loading_label = tk.Label(master, text="", font=("Arial", 12), fg="green")
        self.loading_label.pack()

        self.result_label = tk.Label(master, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Back to main menu button
        self.back_btn = tk.Button(master, text="⬅ Back to Main Menu", command=self.back_to_main_menu)
        self.back_btn.pack(pady=5)

        self.file_path = None

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.file_path = path
            img = Image.open(path)
            img = img.resize((200, 200))
            tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=tk_img)
            self.img_label.image = tk_img
            self.result_label.config(text="")
            self.loading_label.config(text="")

    def predict(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        self.loading_label.config(text="🔄 Predicting...")
        self.master.update_idletasks()

        features = extract_features(self.file_path).reshape(1, -1)
        model_name = self.model_var.get()
        model = models[model_name]
        pred = model.predict(features)[0]
        label = class_map[pred]

        self.loading_label.config(text="")
        self.result_label.config(text=f"Prediction: {label}", fg="blue")

    def back_to_main_menu(self):
        self.master.destroy()
        subprocess.Popen(["python", "main_menu.py"])

# Run app
root = tk.Tk()

app = SkinCancerApp(root)
root.mainloop()
