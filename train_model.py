"""
train_model.py
==============
Script to train, evaluate, and save the Decision Tree Classifier
using the Breast Cancer Wisconsin dataset.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_and_save():
    os.makedirs("models", exist_ok=True)
    
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)  # ['malignant', 'benign']
    
    # Use DataFrame with feature names so the model stores feature_names_in_
    X = pd.DataFrame(data.data, columns=feature_names)
    y = data.target
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Train Decision Tree Classifier
    print("Training DecisionTreeClassifier(max_depth=4, random_state=42)...")
    clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    
    print(f"Model Accuracy:  {accuracy * 100:.2f}%")
    print(f"Model Precision: {precision * 100:.2f}%")
    print(f"Model Recall:    {recall * 100:.2f}%")
    print(f"Model F1-Score:  {f1 * 100:.2f}%")
    
    # Save model
    model_path = os.path.join("models", "decision_tree_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model to: {model_path}")
    
    # Compute feature ranges and sample instances for presets
    benign_samples = X[y == 1]
    malignant_samples = X[y == 0]
    
    metadata = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_names": feature_names,
        "target_names": target_names,
        "target_mapping": {"0": "Malignant", "1": "Benign"},
        "feature_stats": {
            feat: {
                "min": float(X[feat].min()),
                "max": float(X[feat].max()),
                "mean": float(X[feat].mean()),
                "median": float(X[feat].median()),
            }
            for feat in feature_names
        },
        "sample_benign": {feat: float(benign_samples[feat].iloc[0]) for feat in feature_names},
        "sample_malignant": {feat: float(malignant_samples[feat].iloc[0]) for feat in feature_names},
    }
    
    metadata_path = os.path.join("models", "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    train_and_save()
