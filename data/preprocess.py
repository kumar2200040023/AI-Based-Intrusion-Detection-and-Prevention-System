

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset_loader import load_nsl_kdd


def preprocess_pipeline():
    """Full preprocessing pipeline for NSL-KDD dataset."""
    print("=" * 60)
    print("  AI-Based IDS — Data Preprocessing Pipeline")
    print("=" * 60)


    print("\n[1/7] Loading NSL-KDD dataset...")
    df_train = load_nsl_kdd("train")
    df_test = load_nsl_kdd("test")
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}")

    print("\n[2/7] Label encoding categorical features...")
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}

    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    for col in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df_combined[col].astype(str))
        df_combined[col] = pd.Series(encoded, index=df_combined.index)
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")

    # Split back
    df_train = df_combined.iloc[:len(df_train)].copy()
    df_test = df_combined.iloc[len(df_train):].copy()

    # ── Step 3: Encode target labels ──────────────────────
    print("\n[3/7] Encoding attack categories...")
    y_train = df_train['attack_category'].map(config.LABEL_TO_INT).values
    y_test = df_test['attack_category'].map(config.LABEL_TO_INT).values

    # Drop non-feature columns
    drop_cols = ['attack_type', 'attack_category']
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns]).values
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns]).values

    print(f"  Classes: {config.ATTACK_LABELS}")
    for i, label in enumerate(config.ATTACK_LABELS):
        count = np.sum(y_train == i)
        print(f"    {label}: {count} ({count/len(y_train)*100:.1f}%)")

    # ── Step 4: Feature scaling ───────────────────────────
    print("\n[4/7] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Step 5: Feature selection (Top-K) ─────────────────
    print(f"\n[5/7] Selecting top {config.TOP_K_FEATURES} features (Mutual Information)...")
    mi_scores = mutual_info_classif(X_train, np.asarray(y_train), random_state=config.RANDOM_STATE)

    # Get column names (after encoding)
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    mi_ranking = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)

    top_features = [f[0] for f in mi_ranking[:config.TOP_K_FEATURES]]
    top_indices = [feature_cols.index(f) for f in top_features]

    print("  Top features:")
    for i, (name, score) in enumerate(mi_ranking[:config.TOP_K_FEATURES]):
        print(f"    {i+1:2d}. {name:30s} MI={score:.4f}")

    X_train = X_train[:, top_indices]
    X_test = X_test[:, top_indices]

    # ── Step 6: SMOTE oversampling ────────────────────────
    print(f"\n[6/7] Applying SMOTE for class balance...")
    y_train_arr = np.asarray(y_train)
    print(f"  Before: {dict(zip(*np.unique(y_train_arr, return_counts=True)))}")

    smote = SMOTE(random_state=config.RANDOM_STATE)
    result = smote.fit_resample(X_train, y_train_arr)
    X_train = result[0]
    y_train = np.asarray(result[1])

    print(f"  After:  {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ── Step 7: Save processed data ───────────────────────
    print(f"\n[7/7] Saving to {config.PROCESSED_DIR}...")
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(config.PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_DIR, "y_train.npy"), np.asarray(y_train))
    np.save(os.path.join(config.PROCESSED_DIR, "y_test.npy"), np.asarray(y_test))
    joblib.dump(top_features, os.path.join(config.PROCESSED_DIR, "feature_names.joblib"))
    joblib.dump(scaler, os.path.join(config.PROCESSED_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(config.PROCESSED_DIR, "encoders.joblib"))

    print(f"\n  ✓ X_train: {X_train.shape}")
    print(f"  ✓ X_test:  {X_test.shape}")
    print(f"  ✓ Features: {len(top_features)}")
    print("\n✅ Preprocessing complete!")

    return X_train, X_test, y_train, y_test, top_features


if __name__ == "__main__":
    preprocess_pipeline()
