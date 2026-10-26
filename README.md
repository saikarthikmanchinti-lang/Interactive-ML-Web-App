# 🔬 Interactive Machine Learning Web Application (`Interactive_ML_Web_App`)

A beginner-friendly, interactive Machine Learning web application built with **Python**, **Streamlit**, **Scikit-Learn**, **Pandas**, and **NumPy** for a college assignment.

This application uses a **Decision Tree Classifier** trained on Scikit-Learn's **Breast Cancer Wisconsin (Diagnostic)** dataset to classify tumors as **Benign** (non-cancerous) or **Malignant** (cancerous) based on cell nucleus characteristics.

---

## 📌 Project Overview

- **Dataset:** Scikit-Learn Breast Cancer Wisconsin (569 samples, 30 features, 2 classes)
- **Model:** `DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)`
- **Evaluation Metrics:**
  - **Test Accuracy:** `93.86%`
  - **Precision:** `95.77%`
  - **Recall:** `94.44%`
  - **F1-Score:** `95.10%`
- **User Interface:** Streamlit interactive web dashboard with categorized sliders, one-click clinical presets, real-time prediction output, confidence probability bars, and model explanations.

---

## 📁 Project Directory Structure

```text
Interactive_ML_Web_App/
│
├── app.py                      # Main Streamlit web application
├── train_model.py              # Script to train, evaluate, and export the model
├── requirements.txt            # Required Python libraries
├── README.md                   # Complete documentation and run guide
└── models/
    ├── decision_tree_model.joblib  # Trained Decision Tree classifier
    └── metadata.json               # Model metrics, feature statistics, and presets
```

---

## 🚀 Quick Start Guide (Windows)

### Prerequisites
Make sure **Python 3.10+** is installed on your computer.

### Step 1: Install Required Libraries
Open your Windows Terminal or Command Prompt (`cmd` or PowerShell) in this project folder and run:

```bash
pip install -r requirements.txt
```

*(Alternatively, if using the Python Launcher: `py -m pip install -r requirements.txt`)*

### Step 2: (Optional) Re-train the Model
The pre-trained model from Project 04 is already saved in the `models/` folder. If you ever want to re-train the model from scratch, simply run:

```bash
python train_model.py
```

### Step 3: Launch the Streamlit Web Application
Run the following exact command:

```bash
streamlit run app.py
```

*(Or via Python module: `python -m streamlit run app.py` / `py -m streamlit run app.py`)*

The web app will automatically open in your default browser at:
👉 **`http://localhost:8501`**

---

## 🌟 Application Features & Walkthrough

1. **Preset Selection (Sidebar):**
   - **Sample Benign (Harmless):** Automatically loads real cell measurements from a benign patient sample.
   - **Sample Malignant (Cancerous):** Automatically loads real cell measurements from a malignant patient sample.
   - **Dataset Average:** Sets all sliders to the dataset mean.

2. **Categorized Feature Sliders:**
   - **Tab 1: Mean Cell Features** (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension).
   - **Tab 2: Standard Error (SE) Features** (Variation/uncertainty in cell measurements).
   - **Tab 3: Worst (Largest) Features** (Most extreme cell characteristics observed).

3. **Prediction & Clear Result Message:**
   - Clicking **"Predict Tumor Diagnosis"** triggers immediate model inference.
   - Displays a clear color-coded alert:
     - 🟢 **Benign (Non-Cancerous):** Cells are uniform, non-invasive, and non-harmful.
     - 🔴 **Malignant (Cancerous):** Cells exhibit irregularities and potential malignancy.

4. **Prediction Probabilities:**
   - Displays class confidence percentages (e.g., `Benign: 99.2%`, `Malignant: 0.8%`).
   - Visual progress bars show prediction certainty.

5. **Educational Model Explanation:**
   - Explains how **Decision Trees** split data using sequential if-else rules.
   - Details **Gini Impurity** and why tree depth was restricted to `max_depth=4` to avoid overfitting while preserving high interpretability.
   - Highlights the top features influencing decisions (`worst radius`, `worst concave points`, `texture error`).

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Core programming language |
| **Streamlit** | >= 1.30.0 | Interactive web application framework |
| **Scikit-Learn** | >= 1.3.0 | Dataset loading, Decision Tree model, evaluation |
| **Pandas** | >= 2.0.0 | Data handling and tabular display |
| **NumPy** | >= 1.24.0 | Numerical calculations |
| **Joblib** | >= 1.3.0 | Model serialization and disk storage |

---

## ❓ Troubleshooting on Windows

- **Command `streamlit` is not recognized:**
  Use the Python module syntax instead:
  ```bash
  python -m streamlit run app.py
  ```
  or with the Windows launcher:
  ```bash
  py -m streamlit run app.py
  ```
- **Port 8501 already in use:**
  Specify a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **Permission errors with script execution in PowerShell:**
  Run PowerShell as Administrator and enable script execution:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

---

## 🎓 Academic Integrity Note
This project was created for educational purposes as part of a college assignment. Automated submission is disabled; all work is available in the local directory for your personal review and submission.
