import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import joblib
import subprocess
from tensorflow.keras.models import load_model

# Load all models from folder
keras_model = load_model("modelsheart/keras_model.h5")
logreg_model = joblib.load("modelsheart/logistic_regression.pkl")
rf_model = joblib.load("modelsheart/random_forest.pkl")
svm_model = joblib.load("modelsheart/svm_model.pkl")
gb_model = joblib.load("modelsheart/gradient_boosting.pkl")
scaler = joblib.load("modelsheart/scaler.pkl")

models = {
    "Keras Neural Network": keras_model,
    "Logistic Regression": logreg_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "Gradient Boosting": gb_model
}

df = None
feature_names = []
entries = []

def load_csv():
    global df, feature_names
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        df = pd.read_csv(file_path)

        # Drop target column if present
        drop_cols = [col for col in df.columns if col.strip().lower() == 'num']
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        feature_names.clear()
        feature_names.extend(df.columns.tolist())
        update_fields()
        row_slider.config(to=len(df) - 1)
        messagebox.showinfo("Loaded", f"{len(df)} rows loaded successfully with {len(feature_names)} features.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

def update_fields():
    for widget in frame.winfo_children():
        widget.destroy()
    entries.clear()
    for name in feature_names:
        row = tk.Frame(frame, bg="#f0f0f0")
        tk.Label(row, text=f"{name}: ", width=20, anchor="w", bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT)
        entry = tk.Entry(row, width=20, font=("Arial", 10))
        entry.pack(side=tk.RIGHT)
        row.pack(pady=2)
        entries.append(entry)

def fill_from_row(index):
    if df is None:
        messagebox.showwarning("Warning", "Please load a CSV file first.")
        return
    if index >= len(df):
        return
    row_data = df.iloc[index]
    for i, name in enumerate(feature_names):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, str(row_data[name]))

def predict():
    try:
        input_data = [float(entry.get()) for entry in entries]
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        model_name = model_var.get()
        model = models[model_name]

        if model_name == "Keras Neural Network":
            prob = model.predict(scaled_input)[0][0]
        else:
            prob = model.predict_proba(scaled_input)[0][1]

        result = "🔴 CHD Detected" if prob > 0.5 else "🟢 No CHD"
        result_label.config(text=f"{result}\nProbability: {prob:.2f}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Could not make prediction:\n{e}")

def open_main_menu():
    try:
        subprocess.Popen(["python", "main_menu.py"])
        root.destroy()  # Close current window
    except Exception as e:
        messagebox.showerror("Error", f"Could not open main_menu.py:\n{e}")

# ========== GUI Setup ==========

root = tk.Tk()
root.title("Heart Disease Prediction System")
root.state("zoomed")  # Open in full-screen maximized window
root.configure(bg="#f0f0f0")

# Ensure app closes completely when red X is clicked
root.protocol("WM_DELETE_WINDOW", root.destroy)

# Title
tk.Label(root, text="🫀 Coronary Heart Disease Predictor", font=("Helvetica", 24, "bold"),
         bg="#f0f0f0", fg="#222").pack(pady=20)

# CSV Section
tk.Button(root, text="📂 Load Patient CSV", command=load_csv,
          bg="#4CAF50", fg="white", font=("Arial", 11), width=25).pack(pady=5)

tk.Label(root, text="Select Patient Row:", bg="#f0f0f0", font=("Arial", 11)).pack()
row_var = tk.IntVar()
row_slider = tk.Scale(root, from_=0, to=0, orient=tk.HORIZONTAL, variable=row_var,
                      command=lambda val: fill_from_row(int(val)), length=800, bg="#f0f0f0")
row_slider.pack(pady=5)

# Scrollable Entry Section
canvas = tk.Canvas(root, height=350, bg="#f0f0f0", highlightthickness=0)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas, bg="#f0f0f0")
frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)
canvas.pack(side="left", fill="both", expand=True, padx=10)
scroll_y.pack(side="right", fill="y")

# Model Selection
tk.Label(root, text="Select Model:", bg="#f0f0f0", font=("Arial", 11)).pack(pady=10)
model_var = tk.StringVar(value="Keras Neural Network")
model_menu = ttk.Combobox(root, textvariable=model_var, values=list(models.keys()),
                          state="readonly", font=("Arial", 10), width=30)
model_menu.pack(pady=5)

# Predict Button
tk.Button(root, text="🔍 Predict", command=predict,
          bg="#007BFF", fg="white", font=("Arial", 12, "bold"), width=20).pack(pady=15)

# Result Display
result_label = tk.Label(root, text="", font=("Arial", 18), bg="#f0f0f0")
result_label.pack(pady=10)

# Go to Main Menu Button
tk.Button(root, text="🏠 Go to Main Menu", command=open_main_menu,
          bg="#6c757d", fg="white", font=("Arial", 11), width=25).pack(pady=10)
            


root.mainloop()
