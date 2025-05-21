🧠 MedAI-Dx: Multi-Disease AI Diagnostic System

MedAI-Dx is an intelligent, offline-capable diagnostic system that uses artificial intelligence to predict the presence of five major diseases:

Brain Tumors

Alzheimer's Disease

Heart Disease

Skin Cancer

Parkinson’s Disease

Each module in MedAI-Dx is designed with disease-specific models, datasets, and preprocessing logic to ensure accurate and robust predictions. The system integrates all models into a user-friendly graphical interface (GUI) built with Tkinter.

📁 Project Structure

MedAI-Dx/
│
├── main_menu.py                 # Launchpad for accessing each disease module
├── modelsheart/                # Folder containing saved model files
│   ├── keras_model.h5
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm_model.pkl
│   ├── gradient_boosting.pkl
│   ├── scaler.pkl
│   └── ...
├── brain_module.py             # Brain tumor classifier (MRI-based)
├── alzheimer_module.py         # Alzheimer’s MRI classifier with PCA
├── heart_module.py             # Heart disease predictor (tabular data)
├── skin_module.py              # Skin cancer classifier (image)
├── parkinson_module.py         # Parkinson's predictor (voice features)
├── processed_cleveland.csv     # Sample dataset for heart disease
├── runheart.bat.bat            # Batch file to launch heart GUI
├── README.md                   # This file
└── requirements.txt            # Python dependencies

🚀 How to Run

🔧 Requirements

Python 3.8+

TensorFlow

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

Pillow, OpenCV (for image processing)

sounddevice, scipy (for voice input)

Install dependencies:

pip install -r requirements.txt

🖥️ Launch the GUI

python main_menu.py

Or use any of the provided batch files like runheart.bat.bat to launch a specific module directly.

📊 Evaluation Metrics

All models are evaluated using:

Accuracy

Precision, Recall, F1-score

ROC AUC (for classifiers)

R² Score (for Parkinson's regression)

Results are visualized using ROC curves, confusion matrices, and comparative bar charts.


🧲 Sample Datasets Used

Brain Tumor: Brain MRI Dataset (Kaggle)

Alzheimer’s: Alzheimer’s MRI Dataset

Heart Disease: UCI Cleveland Heart Dataset

Skin Cancer: HAM10000 Dataset

Parkinson’s: UCI Parkinson’s Voice Dataset

🛠️ Features

🌟 Multi-model integration

🧠 Disease-specific optimization

🪪 Real-time diagnosis from local files

🖼️ Clean, Tkinter-based interface

🧹 Modular design for easy extension

✅ Offline-capable (no API dependency)

🙌 Authors

Anas Refaat

Kareem Abo Alazam

Yousef Wael

Mohammed Tarek

Alaa Adel

📄 License

This project is intended for educational and research use only.

📌 Acknowledgments

UCI Machine Learning Repository

Kaggle Datasets

Scikit-learn, TensorFlow, Keras

