"""
==========================================================
AETHERIS
ML Candidate Classifier

Loads the model bundle produced by train_classifier.py and
scores a detected BLS candidate as Planet vs False Positive,
with a calibrated probability, an anomaly flag, and a plain-
language explanation of the top contributing factors (SHAP).

This sits alongside vetting.py: vetting.py is the fast
rule-based sanity check, this is the learned classifier.
Showing both -- and any disagreement between them -- is a
good demo beat.
==========================================================
"""

import numpy as np
import joblib

from config import ML_MODEL_PATH, DEFAULT_ST_RADIUS, DEFAULT_ST_TEFF, DEFAULT_ST_LOGG

READABLE_FEATURES = {
    "pl_orbper": "orbital period",
    "pl_trandurh": "transit duration",
    "pl_trandep": "transit depth",
    "log_period": "orbital period",
    "log_depth": "transit depth",
    "duration_period_ratio": "duration-to-period ratio",
    "st_rad": "stellar radius",
    "st_teff": "stellar temperature",
    "st_logg": "stellar surface gravity",
}


class MLClassifier:

    def __init__(self, model_path=ML_MODEL_PATH):

        print(f"\nLoading ML classifier -> {model_path}")

        bundle = joblib.load(model_path)

        self.base_model = bundle["base_model"]
        self.calibrated_model = bundle["calibrated_model"]
        self.explainer = bundle["explainer"]
        self.feature_cols = bundle["feature_cols"]

        self.anomaly_model = bundle.get("anomaly_model")
        self.anomaly_scaler = bundle.get("anomaly_scaler")

        print("Classifier loaded.")

    # ======================================================
    # Feature construction
    # ======================================================

    def build_features(self, candidate, stellar=None):

        stellar = stellar or {}

        period = candidate["period"]
        duration_hours = candidate["duration_hours"]
        depth_ppm = candidate["depth_ppm"]

        st_rad = stellar.get("radius", DEFAULT_ST_RADIUS)
        st_teff = stellar.get("teff", DEFAULT_ST_TEFF)
        st_logg = stellar.get("logg", DEFAULT_ST_LOGG)

        row = {
            "pl_orbper": period,
            "pl_trandurh": duration_hours,
            "pl_trandep": depth_ppm,
            "log_period": np.log10(max(period, 1e-3)),
            "log_depth": np.log10(max(depth_ppm, 1e-3)),
            "duration_period_ratio": duration_hours / (period * 24.0),
            "st_rad": st_rad,
            "st_teff": st_teff,
            "st_logg": st_logg,
        }

        return np.array([[row[col] for col in self.feature_cols]]), row

    # ======================================================
    # Prediction
    # ======================================================

    def predict(self, candidate, stellar=None):

        X, row = self.build_features(candidate, stellar)
        stellar = stellar or {}

        prob_planet = float(self.calibrated_model.predict_proba(X)[0, 1])
        label = "Planet Candidate" if prob_planet >= 0.5 else "Likely False Positive"

        shap_values = self.explainer.shap_values(X)
        contributions = list(zip(self.feature_cols, shap_values[0]))
        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        top_features = contributions[:3]

        result = {
            "ml_probability_planet": round(prob_planet, 4),
            "ml_label": label,
            "ml_top_factors": [
                {"feature": name, "impact": round(float(val), 4)}
                for name, val in top_features
            ],
            "ml_explanation": self._explain(top_features),
        }

        # ------------------------------------------------------
        # Eclipsing Binary (EB) Protection Override
        # ------------------------------------------------------
        depth_ppm = candidate["depth_ppm"]
        spectral_type = str(stellar.get("spectral_type", "N/A"))
        
        # Check for deep stellar eclipses (>15,000 ppm) or known double-lined binaries
        if depth_ppm > 15000 or "SB" in spectral_type:
            result["ml_probability_planet"] = 0.0000
            result["ml_label"] = "Eclipsing Binary (EB)"
            result["ml_explanation"] = (
                f"Override: High probability of stellar eclipse due to massive "
                f"transit depth ({depth_ppm:.0f} ppm) or binary spectral class."
            )
            # Override baseline heuristic classification string inside main.py output
            candidate["classification"] = "Eclipsing Binary"
            candidate["confidence"] = 100

        if self.anomaly_model is not None and self.anomaly_scaler is not None:

            Xs = self.anomaly_scaler.transform(X)

            anomaly_score = float(self.anomaly_model.decision_function(Xs)[0])
            is_anomaly = bool(self.anomaly_model.predict(Xs)[0] == -1)

            result["anomaly_score"] = round(anomaly_score, 4)
            result["is_anomaly"] = is_anomaly

            # If overridden manually due to deep parameters, force an anomaly alert
            if result["ml_label"] == "Eclipsing Binary (EB)":
                result["is_anomaly"] = True

            if result["is_anomaly"]:
                result["ml_explanation"] += (
                    " This candidate's feature profile is unusual compared "
                    "to the training catalog -- worth manual follow-up "
                    "regardless of the classifier score."
                )

        return result

    @staticmethod
    def _explain(top_features):

        parts = []

        for name, val in top_features:
            direction = "supports a planet" if val > 0 else "raises false-positive risk"
            parts.append(f"{READABLE_FEATURES.get(name, name)} ({direction})")

        return "Driven mainly by: " + ", ".join(parts) + "."


_classifier_instance = None


def ml_vet_candidate(candidate, stellar=None, model_path=ML_MODEL_PATH):
    """
    Public API -- mirrors vetting.py's vet_candidate().
    """

    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = MLClassifier(model_path)

    result = _classifier_instance.predict(candidate, stellar)

    candidate.update(result)

    return candidate