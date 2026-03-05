"""
AI-Based IDS — Dataset Loader
==================================
Utilities to load raw and preprocessed datasets.
"""

import os
import pandas as pd
import numpy as np
import joblib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_nsl_kdd(split="train"):
    """
    Load NSL-KDD dataset. Downloads if not present.
    split: 'train' or 'test'
    """
    filename = "KDDTrain+.txt" if split == "train" else "KDDTest+.txt"
    filepath = os.path.join(config.DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[!] {filename} not found at {filepath}")
        print("[*] Downloading NSL-KDD dataset...")
        _download_nsl_kdd()

    df = pd.read_csv(filepath, names=config.KDD_COLUMNS, header=None)

    # Map attack names to categories
    df['attack_category'] = df['attack_type'].map(
        lambda x: config.ATTACK_MAP.get(x, 'Unknown')
    )

    # Drop unknown categories
    df = df[df['attack_category'] != 'Unknown']

    # Drop difficulty_level (last column in NSL-KDD)
    if 'difficulty_level' in df.columns:
        df = df.drop('difficulty_level', axis=1)

    return df


def _download_nsl_kdd():
    """Download NSL-KDD dataset from public mirror."""
    import urllib.request

    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    files = ["KDDTrain+.txt", "KDDTest+.txt"]

    os.makedirs(config.DATA_DIR, exist_ok=True)

    for fname in files:
        url = base_url + fname
        dest = os.path.join(config.DATA_DIR, fname)
        print(f"  Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  ✓ Saved to {dest}")
        except Exception as e:
            print(f"  ✗ Failed to download {fname}: {e}")
            print(f"  Please manually place {fname} in {config.DATA_DIR}")


def load_processed_data():
    """Load preprocessed train/test data."""
    data = {}
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        path = os.path.join(config.PROCESSED_DIR, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path, allow_pickle=True)
        else:
            raise FileNotFoundError(
                f"Processed data not found at {path}. "
                "Run `python data/preprocess.py` first."
            )

    # Load feature names & scaler
    feat_path = os.path.join(config.PROCESSED_DIR, "feature_names.joblib")
    scaler_path = os.path.join(config.PROCESSED_DIR, "scaler.joblib")

    data['feature_names'] = joblib.load(feat_path) if os.path.exists(feat_path) else None
    data['scaler'] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return data


def get_sample_features(n=5):
    """Get sample feature vectors for testing/demo."""
    try:
        data = load_processed_data()
        indices = np.random.choice(len(data['X_test']), size=n, replace=False)
        return data['X_test'][indices], data['y_test'][indices]
    except FileNotFoundError:
        # Return synthetic data
        n_features = config.TOP_K_FEATURES
        X = np.random.randn(n, n_features).astype(np.float32)
        y = np.random.randint(0, len(config.ATTACK_LABELS), size=n)
        return X, y
