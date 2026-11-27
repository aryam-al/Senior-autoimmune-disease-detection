"""XGBoost predictor with per-patient feature contributions (SHAP values).

XGBoost computes exact TreeSHAP values natively (pred_contribs=True), so the
separate `shap` package is not needed.
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from features import DISPLAY_NAMES, FEATURES, encode_features


class XGBoostPredictor:
    name = "XGBoost"

    def __init__(self, model_path: str = config.XGB_MODEL_PATH, meta_path: str = config.XGB_META_PATH):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["features"] != FEATURES:
            raise RuntimeError("Feature order in xgboost_meta.json does not match features.py")
        self.classes: list[str] = meta["classes"]
        self.test_metrics: dict = meta.get("test_metrics", {})
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)

        # Used only when some values were not given (see training/train_xgboost_missing.py).
        self.missing_booster = None
        self.missing_meta: dict = {}
        if os.path.exists(config.XGB_MISSING_MODEL_PATH):
            with open(config.XGB_MISSING_META_PATH, encoding="utf-8") as f:
                self.missing_meta = json.load(f)
            if self.missing_meta["features"] != FEATURES or self.missing_meta["classes"] != self.classes:
                raise RuntimeError("xgboost_missing_meta.json does not match the main model")
            self.missing_booster = xgb.Booster()
            self.missing_booster.load_model(config.XGB_MISSING_MODEL_PATH)

    def predict(self, patient: dict, top_k: int = 5) -> dict:
        X = encode_features(pd.DataFrame([patient]))
        dm = xgb.DMatrix(X, feature_names=FEATURES)

        # Complete values: the main model (the one in the reported results).
        # Some values missing: the version trained with missing values - nothing is filled in.
        incomplete = any(patient.get(f) is None for f in FEATURES)
        booster = self.missing_booster if incomplete and self.missing_booster is not None else self.booster

        probs = booster.predict(dm)[0]
        idx = int(np.argmax(probs))

        # shape: (1, n_classes, n_features + 1); last column is the bias term
        contribs = booster.predict(dm, pred_contribs=True)[0][idx][:-1]
        # Explain with the values that were actually given (a test that was not done is not a reason).
        given = [i for i in np.argsort(-np.abs(contribs)) if patient.get(FEATURES[i]) is not None]
        order = given[:top_k]

        return {
            "available": True,
            "model": self.name,
            "variant": "missing-aware" if booster is self.missing_booster else "main",
            "disease": self.classes[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
            "probabilities": sorted(
                ({"disease": c, "probability": round(float(p) * 100, 2)} for c, p in zip(self.classes, probs)),
                key=lambda r: -r["probability"],
            ),
            "top_features": [
                {
                    "feature": FEATURES[i],
                    "label": DISPLAY_NAMES[FEATURES[i]],
                    "value": patient[FEATURES[i]],
                    "impact": round(float(contribs[i]), 4),
                }
                for i in order
            ],
        }
