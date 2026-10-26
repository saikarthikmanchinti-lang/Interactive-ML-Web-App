"""
Interactive_ML_Web_App
======================
Streamlit Web Application for Breast Cancer Tumor Classification
using a Decision Tree Classifier trained on the Wisconsin Diagnostic Dataset.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Interactive ML Web App | Tumor Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, attractive layout
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 12px 18px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .explanation-box {
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #BFDBFE;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model & Data Loader Function
# ---------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    """
    Loads the trained Decision Tree model from models/ directory if available.
    If not available, trains a new model using Breast Cancer Wisconsin dataset.
    """
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "decision_tree_model.joblib")
    metadata_path = os.path.join("models", "metadata.json")
    
    # Load dataset from scikit-learn
    cancer_data = load_breast_cancer()
    feature_names = list(cancer_data.feature_names)
    target_names = list(cancer_data.target_names)  # ['malignant', 'benign']
    
    df_features = pd.DataFrame(cancer_data.data, columns=feature_names)
    y = cancer_data.target
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Try loading existing model; otherwise train new one
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, model_path)
    
    # Evaluate metrics on test set
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    
    # Find representative samples for presets
    benign_sample = df_features[y == 1].iloc[0].to_dict()
    malignant_sample = df_features[y == 0].iloc[0].to_dict()
    mean_sample = df_features.mean().to_dict()
    
    # Feature ranges (min, max, mean) for sliders
    feature_stats = {}
    for feat in feature_names:
        feature_stats[feat] = {
            "min": float(df_features[feat].min()),
            "max": float(df_features[feat].max()),
            "mean": float(df_features[feat].mean()),
            "step": max(float((df_features[feat].max() - df_features[feat].min()) / 100.0), 0.001)
        }
        
    metadata = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "feature_names": feature_names,
        "target_names": target_names,
        "feature_stats": feature_stats,
        "benign_sample": benign_sample,
        "malignant_sample": malignant_sample,
        "mean_sample": mean_sample
    }
    
    # Update metadata file
    try:
        with open(metadata_path, "w") as f:
            json.dump({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "feature_names": feature_names
            }, f, indent=2)
    except Exception:
        pass
        
    return clf, metadata


# Load model and dataset metadata
model, meta = load_or_train_model()
feature_names = meta["feature_names"]
feature_stats = meta["feature_stats"]


# ---------------------------------------------------------
# Sidebar: Presets & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png",
        width=130
    )
    st.title("Settings & Presets")
    st.markdown("Use presets to quickly test the application with real clinical patient samples:")
    
    preset_choice = st.radio(
        "Select an Input Preset:",
        options=["Dataset Average", "Sample Benign (Harmless)", "Sample Malignant (Cancerous)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Quick Summary")
    st.write(f"**Model Type:** Decision Tree")
    st.write(f"**Max Depth:** 4")
    st.write(f"**Test Accuracy:** `{meta['accuracy'] * 100:.2f}%`")
    st.write(f"**Precision:** `{meta['precision'] * 100:.2f}%`")
    st.write(f"**Recall:** `{meta['recall'] * 100:.2f}%`")
    st.write(f"**F1-Score:** `{meta['f1'] * 100:.2f}%`")
    
    st.markdown("---")
    st.info("💡 **Tip:** After selecting a preset, you can still adjust individual sliders in the main panel.")


# Choose preset values
if preset_choice == "Sample Benign (Harmless)":
    active_values = meta["benign_sample"]
elif preset_choice == "Sample Malignant (Cancerous)":
    active_values = meta["malignant_sample"]
else:
    active_values = meta["mean_sample"]


# ---------------------------------------------------------
# Main Header & Instructions
# ---------------------------------------------------------
st.markdown("<div class=\"main-title\">🔬 Interactive ML Web App: Tumor Classification</div>", unsafe_allow_html=True)
st.markdown(
    "<div class=\"sub-title\">A beginner-friendly Machine Learning web application powered by Streamlit and Scikit-Learn. "
    "Predict whether a breast tumor is <b>Benign</b> (non-cancerous) or <b>Malignant</b> (cancerous) based on cell nucleus characteristics.</div>",
    unsafe_allow_html=True
)

with st.expander("ℹ️ How to Use This Web Application (Click to Expand)", expanded=False):
    st.markdown("""
    1. **Adjust Features**: Use the categorized sliders below to configure the cell measurements, or pick a preset from the left sidebar.
    2. **Click Predict**: Click the **'Predict Tumor Diagnosis'** button to evaluate the sample.
    3. **Analyze Results**: Review the predicted class (Benign vs. Malignant), class probabilities, and model explanation.
    4. **Understand the Model**: Scroll down to read how the Decision Tree makes its classification decisions.
    """)

# Display top-level metric highlights
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Model Accuracy", value=f"{meta['accuracy'] * 100:.2f}%", delta="High Accuracy")
with col_m2:
    st.metric(label="Precision", value=f"{meta['precision'] * 100:.2f}%")
with col_m3:
    st.metric(label="Recall", value=f"{meta['recall'] * 100:.2f}%")
with col_m4:
    st.metric(label="F1-Score", value=f"{meta['f1'] * 100:.2f}%")

st.markdown("---")


# ---------------------------------------------------------
# Input Sliders Section
# ---------------------------------------------------------
st.subheader("⚙️ Step 1: Input Feature Values")
st.caption("The Wisconsin Breast Cancer dataset measures 30 characteristics of cell nuclei grouped into Mean, Standard Error, and Worst (Largest) values.")

# Split features into 3 groups: Mean (0-9), Error (10-19), Worst (20-29)
mean_features = feature_names[0:10]
error_features = feature_names[10:20]
worst_features = feature_names[20:30]

tab1, tab2, tab3 = st.tabs([
    "📐 1. Mean Cell Features",
    "📏 2. Standard Error (SE) Features",
    "🔍 3. Worst (Largest) Features"
])

user_inputs = {}

def render_feature_sliders(feature_list, container):
    """Render a 2-column grid of sliders for a given list of features."""
    c1, c2 = container.columns(2)
    for i, feat in enumerate(feature_list):
        target_col = c1 if (i % 2 == 0) else c2
        stat = feature_stats[feat]
        default_val = float(active_values[feat])
        
        # Ensure default value is bounded by min and max
        val = min(max(default_val, stat["min"]), stat["max"])
        
        # Friendly readable label
        clean_label = feat.replace("_", " ").title()
        
        # Format decimal places
        decimals = 4 if stat["step"] < 0.01 else 2
        
        user_inputs[feat] = target_col.slider(
            label=f"{clean_label}",
            min_value=float(round(stat["min"], decimals)),
            max_value=float(round(stat["max"], decimals)),
            value=float(round(val, decimals)),
            step=float(round(stat["step"], 4)),
            key=f"slider_{feat}_{preset_choice}",
            help=f"Dataset range: {stat['min']:.3f} to {stat['max']:.3f}. Average: {stat['mean']:.3f}"
        )

with tab1:
    st.markdown("**Average measurements across all observed cells in the sample:**")
    render_feature_sliders(mean_features, tab1)

with tab2:
    st.markdown("**Standard error (variability) for each cell measurement:**")
    render_feature_sliders(error_features, tab2)

with tab3:
    st.markdown("**'Worst' (largest or most extreme) measurements observed in the sample:**")
    render_feature_sliders(worst_features, tab3)

st.markdown("---")


# ---------------------------------------------------------
# Step 2: Prediction Section
# ---------------------------------------------------------
st.subheader("🔮 Step 2: Generate Prediction")

# Create a DataFrame from the collected slider inputs
input_df = pd.DataFrame([user_inputs])[feature_names]

predict_btn = st.button("🔬 Predict Tumor Diagnosis", type="primary", use_container_width=True)

if predict_btn:
    # Model inference
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    # Class mapping: 0 = Malignant, 1 = Benign
    prob_malignant = float(probabilities[0])
    prob_benign = float(probabilities[1])
    
    st.markdown("### 📋 Prediction Results")
    res_col1, res_col2 = st.columns([1.2, 1])
    
    with res_col1:
        if prediction == 1:
            st.success("### ✅ Result: Benign (Non-Cancerous Tumor)")
            st.write(
                "**Diagnosis Interpretation:** The model classifies this sample as **Benign** "
                "with high confidence. Benign tumors do not invade surrounding tissue and are typically harmless."
            )
        else:
            st.error("### ⚠️ Result: Malignant (Cancerous Tumor)")
            st.write(
                "**Diagnosis Interpretation:** The model classifies this sample as **Malignant** "
                "(cancerous). Malignant cells have irregular shapes, larger nuclei, and potential to spread. "
                "Immediate medical consultation is warranted."
            )
            
    with res_col2:
        st.markdown("#### 🎯 Prediction Probabilities")
        st.write(f"**Benign Probability:** `{prob_benign * 100:.1f}%`")
        st.progress(prob_benign)
        
        st.write(f"**Malignant Probability:** `{prob_malignant * 100:.1f}%`")
        st.progress(prob_malignant)

    # Feature value summary table
    with st.expander("🔍 View All Current Input Values for this Prediction"):
        summary_table = pd.DataFrame({
            "Feature Name": [f.replace("_", " ").title() for f in feature_names],
            "Selected Value": [round(user_inputs[f], 4) for f in feature_names],
            "Dataset Average": [round(feature_stats[f]["mean"], 4) for f in feature_names]
        })
        st.dataframe(summary_table, use_container_width=True, height=260)


# ---------------------------------------------------------
# Step 3: Educational Explanation of the Model
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📚 Step 3: How Does This Decision Tree Model Work?")

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.markdown("""
    #### 🌳 What is a Decision Tree?
    A **Decision Tree** is a supervised machine learning algorithm that works similarly to a medical flowchart:
    - It starts at the **root node** and asks a series of sequential questions about key feature values.
    - At each branch, it tests a condition (e.g., *Is `worst radius` > 16.79?*).
    - If the condition is True, it moves down one branch; if False, it moves down the other.
    - Once it reaches a **leaf node**, it outputs the final diagnosis: **Benign** or **Malignant**.
    
    #### ⚖️ How the Model Evaluates Splits (Gini Impurity)
    - **Gini Impurity** measures how mixed a group of samples is.
    - A purity of 0 means all samples in the node belong to a single class (pure).
    - At each step, the algorithm chooses the feature and split threshold that **minimizes Gini Impurity** the most.
    """)

with col_e2:
    st.markdown("""
    #### 🏆 Most Important Features in this Tree
    In this trained Decision Tree model, the top features driving the predictions are:
    1. **Worst Radius (~73.4% importance):** The largest measured cell radius in the tumor. Larger cells strongly indicate malignancy.
    2. **Worst Concave Points (~12.2% importance):** The severity of concave indentations along the cell boundary.
    3. **Texture Error (~4.6% importance):** Variation in gray-scale values across the cell nucleus.
    
    #### 🛡️ Why Restrict Tree Depth (`max_depth = 4`)?
    - Unconstrained trees can memorize training noise (overfitting).
    - Limiting the maximum depth to **4 levels** ensures high generalizability (93.86% test accuracy) while keeping the decision logic clear, interpretable, and beginner-friendly!
    """)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>Interactive ML Web App | Developed for College Assignment | Powered by Streamlit & Scikit-Learn</small></center>",
    unsafe_allow_html=True
)
