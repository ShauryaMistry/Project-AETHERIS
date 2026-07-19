"""
==========================================================
AETHERIS
ML Classifier Training Script

Downloads NASA's labeled TESS Objects of Interest (TOI)
catalog and trains:

    1. A calibrated gradient-boosted classifier
       (real planet vs false positive)
    2. An isolation forest anomaly detector
       (flags candidates unlike anything in training)

Run this ONCE, offline, before the pipeline runs, to produce
`aetheris_classifier.pkl`, which modules/ml_classifier.py
loads at inference time.

Requirements:
    pip install pandas requests xgboost scikit-learn shap joblib

Author : Shaurya
==========================================================
"""

import io
import requests
import numpy as np
import pandas as pd
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# ==========================================================
# 1. Download the TOI catalog
# ==========================================================

# NASA Exoplanet Archive native column keys for the 'toi' table
TOI_TAP_QUERY = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_orbper,pl_trandurh,pl_trandep,"
    "st_rad,st_teff,st_logg,tfopwg_disp+"
    "from+toi&format=csv"
)


def download_toi_catalog():

    print("Downloading TOI catalog from NASA Exoplanet Archive...")

    response = requests.get(TOI_TAP_QUERY, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    print(f"Downloaded {len(df)} TOI rows.")

    return df


# ==========================================================
# 2. Label and engineer features
# ==========================================================

LABEL_MAP = {
    "CP": 1,   # Confirmed Planet
    "KP": 1,   # Known Planet
    "FP": 0,   # False Positive
    "FA": 0,   # False Alarm
    # "PC" (Planet Candidate) is unconfirmed -- dropped from training
}


def build_training_set(df):

    df = df.copy()

    df["label"] = df["tfopwg_disp"].map(LABEL_MAP)

    labeled = df.dropna(
        subset=["label", "pl_orbper", "pl_trandurh", "pl_trandep"]
    ).copy()

    print(f"Usable labeled rows: {len(labeled)}")
    print(labeled["label"].value_counts())

    labeled["log_period"] = np.log10(labeled["pl_orbper"].clip(lower=1e-3))
    labeled["log_depth"] = np.log10(labeled["pl_trandep"].clip(lower=1e-3))
    labeled["duration_period_ratio"] = (
        labeled["pl_trandurh"] / (labeled["pl_orbper"] * 24.0)
    )

    for col in ["st_rad", "st_teff", "st_logg"]:
        labeled[col] = labeled[col].fillna(labeled[col].median())

    feature_cols = [
        "pl_orbper", "pl_trandurh", "pl_trandep",
        "log_period", "log_depth", "duration_period_ratio",
        "st_rad", "st_teff", "st_logg",
    ]

    X = labeled[feature_cols]
    y = labeled["label"]

    return X, y, feature_cols


# ==========================================================
# 3. Train + calibrate the classifier
# ==========================================================

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    print("\nTraining base XGBoost model...")
    base_model.fit(X_train, y_train)

    print("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated_model.fit(X_train, y_train)

    probs = calibrated_model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n===== Classifier Evaluation =====")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds, target_names=["FalsePositive", "Planet"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return base_model, calibrated_model, X_train


# ==========================================================
# 4. Anomaly detector
# ==========================================================

def train_anomaly_model(X_train):

    print("\nTraining anomaly detector (Isolation Forest)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    anomaly_model = IsolationForest(contamination=0.05, random_state=42)
    anomaly_model.fit(X_scaled)

    print("Anomaly detector trained.")

    return anomaly_model, scaler


# ==========================================================
# 5. SHAP explainability
# ==========================================================

def build_explainer(base_model):

    print("\nBuilding SHAP explainer...")

    return shap.TreeExplainer(base_model)


# ==========================================================
# Main
# ==========================================================

def main():

    df = download_toi_catalog()

    X, y, feature_cols = build_training_set(df)

    base_model, calibrated_model, X_train = train_model(X, y)

    anomaly_model, anomaly_scaler = train_anomaly_model(X_train)

    explainer = build_explainer(base_model)

    bundle = {
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "anomaly_model": anomaly_model,
        "anomaly_scaler": anomaly_scaler,
    }

    joblib.dump(bundle, "aetheris_classifier.pkl")

    print("\nSaved model bundle -> aetheris_classifier.pkl")


if __name__ == "__main__":
    main()