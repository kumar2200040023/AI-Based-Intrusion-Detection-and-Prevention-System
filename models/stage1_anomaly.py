"""
AI-Based IDS — Stage 1: Unsupervised Anomaly Detection
===========================================================
Isolation Forest, Autoencoder, and One-Class SVM ensemble
for zero-day attack detection.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────
# Isolation Forest Detector
# ─────────────────────────────────────────────────────────
class IsolationForestDetector:
    """Detect anomalies via Isolation Forest. Returns score in [0, 1]."""

    def __init__(self):
        self.model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
        self.fitted = False

    def fit(self, X_normal):
        """Train on normal traffic only."""
        print("  [IF] Training Isolation Forest...")
        self.model.fit(X_normal)
        self.fitted = True
        print(f"  [IF] ✓ Trained on {X_normal.shape[0]} samples")

    def score(self, X):
        """Return anomaly scores in [0, 1]. Higher = more anomalous."""
        raw_scores = self.model.decision_function(X)
        # Convert: IF returns negative for anomalies
        # Normalize to [0, 1] where 1 = highly anomalous
        scores = -raw_scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores

    def save(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "isolation_forest.joblib")
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "isolation_forest.joblib")
        self.model = joblib.load(path)
        self.fitted = True


# ─────────────────────────────────────────────────────────
# Autoencoder Detector (PyTorch)
# ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _AutoencoderNet(nn.Module):
    """PyTorch autoencoder network."""

    def __init__(self, input_dim, encoding_dim=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """
    Detect anomalies via reconstruction error (PyTorch).
    High error = anomalous sample.
    """

    def __init__(self, input_dim=None):
        self.input_dim = input_dim or config.TOP_K_FEATURES
        self.model = None
        self.threshold = None
        self.fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X_normal):
        """Train autoencoder on normal traffic."""
        print("  [AE] Training Autoencoder (PyTorch)...")
        self.input_dim = X_normal.shape[1]
        encoding_dim = config.AUTOENCODER_PARAMS['encoding_dim']

        self.model = _AutoencoderNet(self.input_dim, encoding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # DataLoader
        X_tensor = torch.FloatTensor(X_normal).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset,
                            batch_size=config.AUTOENCODER_PARAMS['batch_size'],
                            shuffle=True)

        # Train
        self.model.train()
        epochs = config.AUTOENCODER_PARAMS['epochs']
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, _ in loader:
                optimizer.zero_grad()
                recon = self.model(X_batch)
                loss = criterion(recon, X_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(loader)
                print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg:.6f}")

        # Set threshold as 95th percentile of reconstruction error
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor).cpu().numpy()
        errors = np.mean(np.square(X_normal - recon), axis=1)
        self.threshold = np.percentile(errors, 95)
        self.fitted = True
        print(f"  [AE] ✓ Trained. Threshold: {self.threshold:.4f}")

    def score(self, X):
        """Return anomaly scores in [0, 1]."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            recon = self.model(X_tensor).cpu().numpy()
        errors = np.mean(np.square(X - recon), axis=1)
        scores = errors / (self.threshold * 2 + 1e-10)
        scores = np.clip(scores, 0, 1)
        return scores

    def save(self, path=None):
        base = path or os.path.join(config.MODEL_DIR, "autoencoder")
        torch.save(self.model.state_dict(), base + ".pt")
        joblib.dump({'threshold': self.threshold, 'input_dim': self.input_dim},
                    base + "_meta.joblib")

    def load(self, path=None):
        base = path or os.path.join(config.MODEL_DIR, "autoencoder")
        meta = joblib.load(base + "_meta.joblib")
        self.threshold = meta['threshold']
        self.input_dim = meta['input_dim']
        encoding_dim = config.AUTOENCODER_PARAMS['encoding_dim']
        self.model = _AutoencoderNet(self.input_dim, encoding_dim).to(self.device)
        self.model.load_state_dict(torch.load(base + ".pt", map_location=self.device))
        self.model.eval()
        self.fitted = True


# ─────────────────────────────────────────────────────────
# One-Class SVM Detector
# ─────────────────────────────────────────────────────────
class OneClassSVMDetector:
    """Detect anomalies via One-Class SVM boundary."""

    def __init__(self):
        self.model = OneClassSVM(**config.OCSVM_PARAMS)
        self.fitted = False

    def fit(self, X_normal):
        """Train on normal traffic. Uses subsample if data is large."""
        print("  [SVM] Training One-Class SVM...")
        # Subsample for speed (OC-SVM is O(n²))
        max_samples = 10000
        if len(X_normal) > max_samples:
            idx = np.random.choice(len(X_normal), max_samples, replace=False)
            X_sub = X_normal[idx]
        else:
            X_sub = X_normal

        self.model.fit(X_sub)
        self.fitted = True
        print(f"  [SVM] ✓ Trained on {X_sub.shape[0]} samples")

    def score(self, X):
        """Return anomaly scores in [0, 1]."""
        raw_scores = self.model.decision_function(X)
        scores = -raw_scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores

    def save(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "ocsvm.joblib")
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "ocsvm.joblib")
        self.model = joblib.load(path)
        self.fitted = True


# ─────────────────────────────────────────────────────────
# Anomaly Ensemble (Stage 1 Output)
# ─────────────────────────────────────────────────────────
class AnomalyEnsemble:
    """
    Combines all three anomaly detectors into a unified
    anomaly scoring module. Output: score in [0, 1].
    """

    def __init__(self):
        self.isolation_forest = IsolationForestDetector()
        self.autoencoder = AutoencoderDetector()
        self.ocsvm = OneClassSVMDetector()
        self.weights = [0.4, 0.35, 0.25]  # IF, AE, SVM

    def fit(self, X_normal):
        """Train all detectors on normal traffic."""
        print("\n━━━ Stage 1: Training Anomaly Detectors ━━━")
        self.isolation_forest.fit(X_normal)
        self.autoencoder.fit(X_normal)
        self.ocsvm.fit(X_normal)
        print("━━━ Stage 1: Complete ━━━\n")

    def score(self, X):
        """Weighted ensemble anomaly score."""
        s1 = self.isolation_forest.score(X)
        s2 = self.autoencoder.score(X)
        s3 = self.ocsvm.score(X)

        combined = (
            self.weights[0] * s1 +
            self.weights[1] * s2 +
            self.weights[2] * s3
        )
        return combined

    def save(self):
        self.isolation_forest.save()
        self.autoencoder.save()
        self.ocsvm.save()

    def load(self):
        self.isolation_forest.load()
        self.autoencoder.load()
        self.ocsvm.load()
