"""
AI-Based IDS — End-to-End Training Pipeline
=================================================
Preprocesses data, trains all 3 ML stages, evaluates,
and saves models.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.preprocess import preprocess_pipeline
from data.dataset_loader import load_processed_data
from models.stage1_anomaly import AnomalyEnsemble
from models.stage2_classifier import EnsembleVoter
from models.stage3_temporal import TemporalEngine
from engine.feedback import IncrementalLearner


def train():
    """Full training pipeline."""
    start = time.time()

    print("╔══════════════════════════════════════════════════════╗")
    print("║       🛡 AI-Based IDS — Training Pipeline       ║")
    print("║       Multi-Stage Intelligent Detection System       ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: Preprocess ────────────────────────────────
    print("▶ PHASE 1: Data Preprocessing\n")
    try:
        data = load_processed_data()
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        print("  Using existing preprocessed data.")
    except FileNotFoundError:
        print("  Preprocessed data not found. Running pipeline...")
        X_train, X_test, y_train, y_test, _ = preprocess_pipeline()

    print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {np.unique(y_train).tolist()}")

    # ── Step 2: Stage 1 — Anomaly Detection ──────────────
    print("\n▶ PHASE 2: Stage 1 — Unsupervised Anomaly Detection\n")
    anomaly = AnomalyEnsemble()

    # Train on normal traffic only
    normal_mask = y_train == config.LABEL_TO_INT['Normal']
    X_normal = X_train[normal_mask]
    print(f"  Normal samples for training: {X_normal.shape[0]}")

    anomaly.fit(X_normal)

    # Evaluate
    print("  Evaluating on test set...")
    anomaly_scores = anomaly.score(X_test)
    normal_test_mask = y_test == config.LABEL_TO_INT['Normal']

    avg_normal = np.mean(anomaly_scores[normal_test_mask])
    avg_attack = np.mean(anomaly_scores[~normal_test_mask])
    print(f"  Avg anomaly score (normal):  {avg_normal:.4f}")
    print(f"  Avg anomaly score (attacks): {avg_attack:.4f}")
    print(f"  Separation gap: {avg_attack - avg_normal:.4f}")

    anomaly.save()
    print("  ✓ Stage 1 models saved")

    # ── Step 3: Stage 2 — Supervised Classification ──────
    print("\n▶ PHASE 3: Stage 2 — Supervised Classification\n")
    classifier = EnsembleVoter()
    classifier.fit(X_train, y_train, X_test, y_test)
    classifier.evaluate(X_test, y_test)
    classifier.save()
    print("  ✓ Stage 2 models saved")

    # ── Step 4: Stage 3 — Temporal Detection ─────────────
    print("\n▶ PHASE 4: Stage 3 — Temporal Detection (LSTM)\n")
    temporal = TemporalEngine(model_type='lstm')

    # Use a subset for temporal training (sequence-based)
    max_temporal = min(50000, len(X_train))
    idx = np.random.choice(len(X_train), max_temporal, replace=False)
    X_temporal = X_train[idx]
    y_temporal = y_train[idx]

    temporal.fit(X_temporal, y_temporal)

    # Evaluate
    print("  Evaluating temporal scores...")
    temporal_scores = temporal.temporal_score(X_test[:1000])
    avg_t_normal = np.mean(temporal_scores[normal_test_mask[:1000]])
    avg_t_attack = np.mean(temporal_scores[~normal_test_mask[:1000]])
    print(f"  Avg temporal score (normal):  {avg_t_normal:.4f}")
    print(f"  Avg temporal score (attacks): {avg_t_attack:.4f}")

    temporal.save()
    print("  ✓ Stage 3 model saved")

    # ── Step 5: Initialize Online Learner ─────────────────
    print("\n▶ PHASE 5: Initializing Online Learner\n")
    online = IncrementalLearner()
    online.initial_fit(X_train, y_train)
    online.save()
    print("  ✓ Online learner initialized and saved")

    # ── Summary ───────────────────────────────────────────
    elapsed = time.time() - start
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║              ✅ Training Complete!                   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Total time: {elapsed:.1f}s")
    print(f"║  Models saved to: {config.MODEL_DIR}")
    print("║  ")
    print("║  Next steps:")
    print("║    1. Start API:  python -m api.main")
    print("║    2. Dashboard:  streamlit run dashboard/app.py")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    train()
