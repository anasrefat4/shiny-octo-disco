# MedAI-Dx: Local Deployment and Runtime Instructions

## 📦 Overview

MedAI-Dx is a local, modular machine learning diagnostic platform for detecting five diseases:

* Brain Tumors
* Alzheimer's Disease
* Heart Disease
* Skin Cancer
* Parkinson’s Disease

Each module is a standalone script using pre-trained models saved in `/modelsheart/`. A Tkinter GUI serves as the entry point for interactive testing.

---

## 🗂️ Folder Structure

```
MedAI-Dx/
├── main_menu.py                  # Entry GUI for all modules
├── brain_module.py               # Brain tumor pipeline
├── alzheimer_module.py           # Alzheimer's pipeline
├── heart_module.py               # Heart disease predictor
├── skin_module.py                # Skin cancer classifier
├── parkinson_module.py           # Parkinson's predictor
├── processed_cleveland.csv       # Heart dataset (sample)
├── modelsheart/                  # Folder containing all saved models
│   ├── keras_model.h5
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm_model.pkl
│   ├── gradient_boosting.pkl
│   └── scaler.pkl
├── runheart.bat.bat              # Batch launcher for heart module
├── README.md
└── requirements.txt              # Python dependencies
```

---

## ⚙️ System Requirements

* Python 3.8+
* OS: Windows/Linux
* Disk: At least 2 GB free space for models and test data

### Required Python Packages

Install all dependencies:

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn opencv-python sounddevice
```

---

## 🚀 Running the System

### Launch Full GUI:

```bash
python main_menu.py
```

### Run a Specific Module:

```bash
python heart_module.py
python brain_module.py
```

Or via batch:

```bash
./runheart.bat.bat
```

> Ensure that `modelsheart/` folder exists in the same directory. Models are loaded using relative paths.

---

## 🤖 Models Summary

| Module        | Model Type       | File Path                             |
| ------------- | ---------------- | ------------------------------------- |
| Brain Tumor   | VotingClassifier | Used via DenseNet169 features         |
| Alzheimer's   | VotingClassifier | DenseNet169 + PCA + SVM/KNN/RF        |
| Heart Disease | Classical ML     | `logistic_regression.pkl`, `rf`, `gb` |
| Skin Cancer   | CNN + DNN        | Loaded via `keras_model.h5`           |
| Parkinson’s   | SVM              | `svm_model.pkl` with `scaler.pkl`     |

---

## 🔍 Notes for Debugging

* If models fail to load, verify `.pkl` and `.h5` files exist in `/modelsheart/`
* If GUI fails, check Python Tkinter installation
* For TensorFlow issues, use:

```bash
pip install tensorflow --upgrade
```

---

## 🧑‍💻 Authors

* Anas Refaat, Kareem Abo Alazam, Yousef Wael, Mohammed Tarek, Alaa Adel

---

## 📄 License

For academic and research use only.
