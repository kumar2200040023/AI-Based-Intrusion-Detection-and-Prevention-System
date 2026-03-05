"""
AI-Based IDS — Stage 2: Supervised Classification
======================================================
Random Forest + XGBoost ensemble with weighted voting
for multi-class attack classification.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────
# Random Forest Classifier
# ─────────────────────────────────────────────────────────
class RFClassifier:
    """Random Forest for multi-class attack detection."""

    def __init__(self):
        self.model = RandomForestClassifier(**config.RF_PARAMS)
        self.fitted = False

    def fit(self, X, y):
        print("  [RF] Training Random Forest...")
        self.model.fit(X, y)
        self.fitted = True
        # Feature importances
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[-5:][::-1]
        print(f"  [RF] ✓ Trained. Top 5 feature indices: {top_idx.tolist()}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n  [RF] Accuracy: {acc:.4f}")
        print(classification_report(
            y, y_pred,
            target_names=config.ATTACK_LABELS,
            zero_division=0
        ))
        return acc

    def save(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "random_forest.joblib")
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "random_forest.joblib")
        self.model = joblib.load(path)
        self.fitted = True


# ─────────────────────────────────────────────────────────
# XGBoost Classifier
# ─────────────────────────────────────────────────────────
class XGBClassifier:
    """XGBoost for high-accuracy attack classification."""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            **config.XGB_PARAMS,
            num_class=len(config.ATTACK_LABELS),
            objective='multi:softprob',
        )
        self.fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        print("  [XGB] Training XGBoost...")
        eval_set = [(X, y)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        self.fitted = True
        print(f"  [XGB] ✓ Trained. Best iteration: {self.model.best_iteration}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n  [XGB] Accuracy: {acc:.4f}")
        print(classification_report(
            y, y_pred,
            target_names=config.ATTACK_LABELS,
            zero_division=0
        ))
        return acc

    def save(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "xgboost.joblib")
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "xgboost.joblib")
        self.model = joblib.load(path)
        self.fitted = True


# ─────────────────────────────────────────────────────────
# Ensemble Voter (Stage 2 Output)
# ─────────────────────────────────────────────────────────
class EnsembleVoter:
    """
    Weighted soft-voting ensemble of RF + XGBoost.
    Output: predicted class + confidence score.
    """

    def __init__(self):
        self.rf = RFClassifier()
        self.xgb = XGBClassifier()
        self.weights = config.ENSEMBLE_WEIGHTS

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("\n━━━ Stage 2: Training Supervised Classifiers ━━━")
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train, X_val, y_val)
        print("━━━ Stage 2: Complete ━━━\n")

    def predict(self, X):
        """Weighted soft voting prediction."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """Weighted probability combination."""
        p_rf = self.rf.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)

        # Ensure same number of classes
        n_classes = len(config.ATTACK_LABELS)
        if p_rf.shape[1] < n_classes:
            pad = np.zeros((p_rf.shape[0], n_classes - p_rf.shape[1]))
            p_rf = np.hstack([p_rf, pad])
        if p_xgb.shape[1] < n_classes:
            pad = np.zeros((p_xgb.shape[0], n_classes - p_xgb.shape[1]))
            p_xgb = np.hstack([p_xgb, pad])

        combined = (
            self.weights['rf'] * p_rf +
            self.weights['xgb'] * p_xgb
        )
        return combined

    def confidence_score(self, X):
        """Return max probability as confidence for each sample."""
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

    def evaluate(self, X, y):
        print("\n── Ensemble Evaluation ──")
        self.rf.evaluate(X, y)
        self.xgb.evaluate(X, y)

        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n  [ENSEMBLE] Accuracy: {acc:.4f}")
        print(classification_report(
            y, y_pred,
            target_names=config.ATTACK_LABELS,
            zero_division=0
        ))
        return acc

    def save(self):
        self.rf.save()
        self.xgb.save()

    def load(self):
        self.rf.load()
        self.xgb.load()
