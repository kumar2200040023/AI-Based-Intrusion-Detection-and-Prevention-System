

import os
import sys
import json
import numpy as np
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# ─────────────────────────────────────────────────────────
# Feedback Store
# ─────────────────────────────────────────────────────────
class FeedbackStore:
    """
    Store analyst feedback. Uses MongoDB if available,
    falls back to file-based JSON storage.
    """

    def __init__(self):
        self.use_mongo = MONGO_AVAILABLE
        self._file_path = os.path.join(config.LOG_DIR, "feedback.json")
        self._memory_store = deque(maxlen=10000)

        if self.use_mongo:
            try:
                from pymongo import MongoClient as MC
                self.client = MC(config.MONGO_URI, serverSelectionTimeoutMS=2000)
                self.client.server_info()  # Test connection
                self.db = self.client[config.MONGO_DB]
                self.collection = self.db[config.MONGO_FEEDBACK_COLLECTION]
                print("  [Feedback] Connected to MongoDB")
            except Exception:
                self.use_mongo = False
                print("  [Feedback] MongoDB unavailable, using file storage")

    def store_feedback(self, alert_id, feedback_type, features=None,
                       true_label=None, analyst_id="analyst_1"):
        """
        Store feedback from SOC analyst.

        feedback_type: 'true_positive', 'false_positive', 'false_negative'
        """
        record = {
            'alert_id': alert_id,
            'feedback_type': feedback_type,
            'true_label': true_label,
            'features': features.tolist() if features is not None else None,
            'analyst_id': analyst_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        if self.use_mongo:
            self.collection.insert_one(record)
        else:
            self._memory_store.append(record)
            self._save_to_file()

        return record

    def get_feedback(self, limit=100, feedback_type=None):
        """Retrieve recent feedback records."""
        if self.use_mongo:
            query = {}
            if feedback_type:
                query['feedback_type'] = feedback_type
            cursor = self.collection.find(query).sort(
                'timestamp', -1).limit(limit)
            return list(cursor)
        else:
            records = list(self._memory_store)
            if feedback_type:
                records = [r for r in records if r['feedback_type'] == feedback_type]
            return records[-limit:]

    def get_feedback_count(self):
        """Count total feedback entries."""
        if self.use_mongo:
            return self.collection.count_documents({})
        return len(self._memory_store)

    def get_feedback_stats(self):
        """Get feedback statistics."""
        if self.use_mongo:
            tp = self.collection.count_documents({'feedback_type': 'true_positive'})
            fp = self.collection.count_documents({'feedback_type': 'false_positive'})
            fn = self.collection.count_documents({'feedback_type': 'false_negative'})
        else:
            records = list(self._memory_store)
            tp = sum(1 for r in records if r['feedback_type'] == 'true_positive')
            fp = sum(1 for r in records if r['feedback_type'] == 'false_positive')
            fn = sum(1 for r in records if r['feedback_type'] == 'false_negative')

        total = tp + fp + fn
        return {
            'total': total,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'fpr': round(fp / (fp + tp), 4) if (fp + tp) > 0 else 0.0,
            'improvement_rate': round(tp / total, 4) if total > 0 else 0.0,
        }

    def _save_to_file(self):
        records = list(self._memory_store)
        # Clean for JSON serialization
        clean = []
        for r in records:
            c = {k: v for k, v in r.items() if k != '_id'}
            clean.append(c)
        with open(self._file_path, 'w') as f:
            json.dump(clean, f, indent=2)

    def _load_from_file(self):
        if os.path.exists(self._file_path):
            with open(self._file_path, 'r') as f:
                records = json.load(f)
                self._memory_store.extend(records)


# ─────────────────────────────────────────────────────────
# Incremental Learner (Online Learning)
# ─────────────────────────────────────────────────────────
class IncrementalLearner:
    """
    Online learning via SGDClassifier (partial_fit).
    Learns incrementally from analyst feedback without
    full retraining.
    """

    def __init__(self):
        self.model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.001,
            random_state=config.RANDOM_STATE,
            warm_start=True,
        )
        self.scaler = StandardScaler()
        self.classes = np.arange(len(config.ATTACK_LABELS))
        self.update_count = 0
        self.fitted = False

    def initial_fit(self, X, y):
        """Initial training with full dataset."""
        print("  [Online] Initial fit of incremental learner...")
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y, classes=self.classes)
        self.fitted = True
        self.update_count += 1
        print(f"  [Online] ✓ Initial fit complete ({len(X)} samples)")

    def update(self, X, y):
        """Incremental update with new feedback data."""
        if not self.fitted:
            self.initial_fit(X, y)
            return

        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y)
        self.update_count += 1

    def predict(self, X):
        if not self.fitted:
            return np.zeros(len(X), dtype=int)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        if not self.fitted:
            return np.ones((len(X), len(self.classes))) / len(self.classes)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "incremental_learner.joblib")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'update_count': self.update_count,
        }, path)

    def load(self, path=None):
        path = path or os.path.join(config.MODEL_DIR, "incremental_learner.joblib")
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.update_count = data['update_count']
        self.fitted = True


# ─────────────────────────────────────────────────────────
# Retraining Scheduler
# ─────────────────────────────────────────────────────────
class RetrainingScheduler:
    """
    Monitors feedback accumulation and triggers
    periodic model retraining.
    """

    def __init__(self, feedback_store, incremental_learner):
        self.feedback_store = feedback_store
        self.learner = incremental_learner
        self.last_retrain = datetime.now(timezone.utc)
        self.retrain_count = 0

    def check_and_retrain(self):
        """Check if retraining is needed and execute if so."""
        count = self.feedback_store.get_feedback_count()

        if count < config.MIN_FEEDBACK_FOR_RETRAIN:
            return False

        # Gather feedback with features
        records = self.feedback_store.get_feedback(limit=count)
        X_feedback = []
        y_feedback = []

        for r in records:
            if r.get('features') is not None and r.get('true_label') is not None:
                X_feedback.append(r['features'])
                y_feedback.append(r['true_label'])

        if len(X_feedback) < 10:
            return False

        X = np.array(X_feedback, dtype=np.float32)
        y = np.array(y_feedback, dtype=int)

        print(f"\n  [Retrain] Retraining with {len(X)} feedback samples...")
        self.learner.update(X, y)
        self.retrain_count += 1
        self.last_retrain = datetime.now(timezone.utc)
        print(f"  [Retrain] ✓ Retrain #{self.retrain_count} complete")

        return True

    def get_status(self):
        return {
            'retrain_count': self.retrain_count,
            'last_retrain': self.last_retrain.isoformat(),
            'pending_feedback': self.feedback_store.get_feedback_count(),
            'min_required': config.MIN_FEEDBACK_FOR_RETRAIN,
        }
