"""
==========================================================
AETHERIS
Main Pipeline Processing & Vetting Execution Engine

Loads the freshly trained offline artifact bundle 
(aetheris_classifier.pkl) and processes ingested candidate 
signals through the machine learning vetting engine.

Author : Shaurya
==========================================================
"""

import os
import numpy as np
import pandas as pd
import joblib

# Target paths
MODEL_BUNDLE_PATH = "aetheris_classifier.pkl"


def load_classifier_bundle():
    """Loads the serialized gradient-boosted and anomaly detector artifacts."""
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(
            f"Could not locate '{MODEL_BUNDLE_PATH}'. "
            "Please run `python train_classifier.py` first to generate it."
        )
    
    print(f"Loading machine learning artifact bundle from {MODEL_BUNDLE_PATH}...")
    return joblib.load(MODEL_BUNDLE_PATH)


def engineer_inference_features(candidate, stellar):
    """
    Transforms basic astronomical observations into the 9 exact features
    the underlying XGBoost model expects.
    """
    # Map raw candidate dictionaries to processing workspace
    pl_orbper = float(candidate.get("period", 0.0))
    pl_trandurh = float(candidate.get("duration_hours", 0.0))
    pl_trandep = float(candidate.get("depth_ppm", 0.0))
    
    st_rad = float(stellar.get("radius", 1.0))
    st_teff = float(stellar.get("teff", 5778.0)) # Sun default fallback
    st_logg = float(stellar.get("logg", 4.44))
    
    # Mathematical scaling layers matching training prep bounds
    log_period = np.log10(max(pl_orbper, 1e-3))
    log_depth = np.log10(max(pl_trandep, 1e-3))
    duration_period_ratio = pl_trandurh / (max(pl_orbper, 1e-3) * 24.0)
    
    # Build strict feature dataframe matching column order matrix
    features = pd.DataFrame([{
        "pl_orbper": pl_orbper,
        "pl_trandurh": pl_trandurh,
        "pl_trandep": pl_trandep,
        "log_period": log_period,
        "log_depth": log_depth,
        "duration_period_ratio": duration_period_ratio,
        "st_rad": st_rad,
        "st_teff": st_teff,
        "st_logg": st_logg
    }])
    
    return features


def ml_vet_candidate(candidate, stellar, bundle):
    """
    Vets a transit candidate through the machine learning ensemble.
    Applies explicit geometric scale overrides for Eclipsing Binaries.
    """
    # 1. Structural Feature Extraction
    features_df = engineer_inference_features(candidate, stellar)
    
    # 2. Extract Astronomical Variables for Heuristic Overrides
    depth_ppm = float(candidate.get("depth_ppm", 0.0))
    st_rad = float(stellar.get("radius", 1.0))
    
    # Approximate physical planetary size scale (in Earth Radii) using stellar geometry
    # Formula: Rp = R_star * sqrt(depth_ppm / 1,000,000) * 109.2
    rp_earth = st_rad * np.sqrt(depth_ppm / 1e6) * 109.2
    
    # 3. Apply Hard Eclipsing Binary (EB) Override Safeguards
    # Objects larger than 2.5 Jupiter Radii (~28 Earth Radii) cannot structurally be planets
    if rp_earth > 28.0:
        return {
            "ml_label": 0,
            "ml_probability_planet": 0.0000,
            "anomaly_score": -1.0,
            "is_anomaly": True,
            "ml_explanation": f"EB OVERRIDE: Calculated radius ({rp_earth:.2f} R_Earth) exceeds physical planet bounds (>2.5 R_Jupiter)."
        }

    # 4. Generate Machine Learning Inference Scores
    calibrated_model = bundle["calibrated_model"]
    anomaly_model = bundle["anomaly_model"]
    anomaly_scaler = bundle["anomaly_scaler"]
    feature_cols = bundle["feature_cols"]
    
    # Align dataframe strictly with artifact columns
    X_infer = features_df[feature_cols]
    
    # Predict probabilities (Real Planet vs False Positive)
    prob_planet = float(calibrated_model.predict_proba(X_infer)[0, 1])
    ml_label = 1 if prob_planet >= 0.5 else 0
    
    # 5. Outlier/Anomaly Ingestion Check via Isolation Forest
    X_scaled = anomaly_scaler.transform(X_infer)
    anomaly_pred = anomaly_model.predict(X_scaled)[0] # Returns -1 for anomaly, 1 for normal
    anomaly_score = float(anomaly_model.decision_function(X_scaled)[0])
    is_anomaly = True if anomaly_pred == -1 else False
    
    # Define summary explanation strings
    explanation = "Classifier model predicts a highly stable planetary signature." if ml_label == 1 \
                  else "Classifier flags tracking signature as consistent with background False Positive modes."
                  
    if is_anomaly:
        explanation += " [ANOMALY DETECTED: Features deviate from core baseline space]."

    return {
        "ml_label": ml_label,
        "ml_probability_planet": prob_planet,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "ml_explanation": explanation
    }


# ==========================================================
# Test/Execution Demonstration Harness
# ==========================================================

def main():
    print("Initializing AETHERIS Ingest Vetting Pipeline...")
    
    # Load Model Bundle
    try:
        bundle = load_classifier_bundle()
    except Exception as e:
        print(f"Execution Aborted: {e}")
        return

    # Mock Signals: A typical candidate vs an Eclipsing Binary star system
    mock_candidates = [
        {
            "id": "TIC-TEST-PLANET",
            "candidate": {"period": 11.42, "duration_hours": 3.2, "depth_ppm": 450.0},
            "stellar": {"radius": 0.95, "teff": 5600.0, "logg": 4.40}
        },
        {
            "id": "TIC-TEST-BINARY-ECLIPSE",
            "candidate": {"period": 4.82, "duration_hours": 8.5, "depth_ppm": 65000.0},
            "stellar": {"radius": 1.45, "teff": 6200.0, "logg": 4.12}
        }
    ]

    print("\n===== Running Pipeline Ingestion Inferences =====")
    for target in mock_candidates:
        print(f"\nProcessing Target ID: {target['id']}")
        
        result = ml_vet_candidate(target["candidate"], target["stellar"], bundle)
        
        print(f" -> Classification Decision : {'PLANET' if result['ml_label'] == 1 else 'FALSE POSITIVE'}")
        print(f" -> Calibrated Probability  : {result['ml_probability_planet'] * 100:.2f}%")
        print(f" -> Anomaly Alert Status    : {result['is_anomaly']} (Score: {result['anomaly_score']:.4f})")
        print(f" -> Pipeline Vetting Note   : {result['ml_explanation']}")


if __name__ == "__main__":
    main()