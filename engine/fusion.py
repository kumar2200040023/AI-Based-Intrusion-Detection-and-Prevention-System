"""
AI-Based IDS — Decision Fusion Engine
===========================================
Combines anomaly, classifier, and temporal scores
with dynamic threshold for final threat verdict.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ThreatVerdict:
    """Structured result from the Decision Fusion Engine."""

    def __init__(self, score, label, confidence, is_threat,
                 anomaly_score, classifier_score, temporal_score,
                 attack_type, threshold):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.score = float(score)
        self.label = label
        self.confidence = float(confidence)
        self.is_threat = bool(is_threat)
        self.anomaly_score = float(anomaly_score)
        self.classifier_score = float(classifier_score)
        self.temporal_score = float(temporal_score)
        self.attack_type = attack_type
        self.threshold = float(threshold)

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'threat_score': round(self.score, 4),
            'label': self.label,
            'attack_type': self.attack_type,
            'confidence_score': round(self.confidence, 4),
            'is_threat': self.is_threat,
            'threshold': round(self.threshold, 4),
            'components': {
                'anomaly_score': round(self.anomaly_score, 4),
                'classifier_score': round(self.classifier_score, 4),
                'temporal_score': round(self.temporal_score, 4),
            }
        }

    def to_siem_alert(self, src_ip="0.0.0.0", dst_ip="0.0.0.0"):
        """Format as SIEM-compatible JSON alert."""
        return {
            'timestamp': self.timestamp,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'attack_type': self.attack_type,
            'confidence_score': round(self.confidence, 4),
            'threat_score': round(self.score, 4),
            'model_version': config.MODEL_VERSION,
            'is_threat': self.is_threat,
        }


class DecisionFusionEngine:
    """
    Combines all three detection stages:
      Final Score = w1×anomaly + w2×classifier + w3×temporal

    Uses dynamic threshold that adjusts based on FPR feedback.
    """

    def __init__(self):
        self.weights = config.FUSION_WEIGHTS
        self.threshold = config.DYNAMIC_THRESHOLD_INIT
        self.threshold_history = [self.threshold]
        self.fp_count = 0
        self.tp_count = 0

    def fuse(self, anomaly_scores, classifier_proba, temporal_scores,
             classifier_labels):
        """
        Fuse all three stage outputs into final threat verdicts.

        Args:
            anomaly_scores: np.array of shape (n,), values in [0, 1]
            classifier_proba: np.array of shape (n, n_classes)
            temporal_scores: np.array of shape (n,), values in [0, 1]
            classifier_labels: np.array of shape (n,), predicted class indices

        Returns:
            List of ThreatVerdict objects
        """
        n = len(anomaly_scores)
        verdicts = []

        # Classifier confidence = max probability
        classifier_confidence = np.max(classifier_proba, axis=1)

        for i in range(n):
            # Weighted fusion
            final_score = (
                self.weights['anomaly'] * anomaly_scores[i] +
                self.weights['classifier'] * classifier_confidence[i] +
                self.weights['temporal'] * temporal_scores[i]
            )

            # Determine threat
            is_threat = final_score > self.threshold
            label_idx = classifier_labels[i]
            attack_type = config.INT_TO_LABEL.get(label_idx, 'Unknown')

            # If classified as Normal but anomaly is high → still flag
            if attack_type == 'Normal' and anomaly_scores[i] > 0.8:
                attack_type = 'Suspicious'
                is_threat = True

            label = 'THREAT' if is_threat else 'NORMAL'

            verdict = ThreatVerdict(
                score=final_score,
                label=label,
                confidence=classifier_confidence[i],
                is_threat=is_threat,
                anomaly_score=anomaly_scores[i],
                classifier_score=classifier_confidence[i],
                temporal_score=temporal_scores[i],
                attack_type=attack_type if is_threat else 'Normal',
                threshold=self.threshold,
            )
            verdicts.append(verdict)

        return verdicts

    def update_threshold(self, feedback_type):
        """
        Dynamically adjust threshold based on analyst feedback.
        - Too many False Positives → increase threshold
        - Too many False Negatives → decrease threshold
        """
        if feedback_type == 'false_positive':
            self.fp_count += 1
            self.threshold = min(0.95, self.threshold + config.THRESHOLD_DECAY)
        elif feedback_type == 'true_positive':
            self.tp_count += 1
        elif feedback_type == 'false_negative':
            self.threshold = max(0.1, self.threshold - config.THRESHOLD_DECAY)

        self.threshold_history.append(self.threshold)

    @property
    def false_positive_rate(self):
        total = self.fp_count + self.tp_count
        return self.fp_count / total if total > 0 else 0.0

    def get_stats(self):
        return {
            'current_threshold': round(self.threshold, 4),
            'fp_count': self.fp_count,
            'tp_count': self.tp_count,
            'fpr': round(self.false_positive_rate, 4),
            'threshold_adjustments': len(self.threshold_history),
        }
