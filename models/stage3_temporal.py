
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """LSTM for temporal attack sequence classification."""

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        out = self.classifier(context)
        return out


# ─────────────────────────────────────────────────────────
# GRU Model
# ─────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    """GRU — faster alternative to LSTM for temporal detection."""

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=5, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Use last timestep
        out = self.classifier(gru_out[:, -1, :])
        return out


# ─────────────────────────────────────────────────────────
# Temporal Engine (Stage 3 Manager)
# ─────────────────────────────────────────────────────────
class TemporalEngine:
    """
    Manages sequence windowing, training, and inference
    for LSTM/GRU temporal attack detection.
    """

    def __init__(self, model_type='lstm'):
        self.model_type = model_type
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = config.SEQUENCE_WINDOW
        self.fitted = False

    def _create_sequences(self, X, y=None):
        """Create sliding window sequences from flat feature data."""
        n_samples = len(X)
        n_features = X.shape[1]

        if n_samples < self.window_size:
            # Pad with zeros if not enough data
            padding = np.zeros((self.window_size - n_samples, n_features))
            X_padded = np.vstack([padding, X])
            sequences = X_padded[np.newaxis, :, :]
            labels = np.array([y[-1]]) if y is not None else None
            return sequences, labels

        n_sequences = n_samples - self.window_size + 1
        sequences = np.zeros((n_sequences, self.window_size, n_features))

        for i in range(n_sequences):
            sequences[i] = X[i:i + self.window_size]

        labels = y[self.window_size - 1:] if y is not None else None
        return sequences, labels

    def fit(self, X, y):
        """Train temporal model on sequence data."""
        print(f"\n━━━ Stage 3: Training {'LSTM' if self.model_type == 'lstm' else 'GRU'} ━━━")

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        print(f"  Sequences: {X_seq.shape}")

        n_features = X_seq.shape[2]
        n_classes = len(config.ATTACK_LABELS)
        params = config.LSTM_PARAMS

        # Create model
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_size=n_features,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                num_classes=n_classes,
                dropout=params['dropout'],
            ).to(self.device)
        else:
            self.model = GRUModel(
                input_size=n_features,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                num_classes=n_classes,
                dropout=params['dropout'],
            ).to(self.device)

        # Data loader
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

        self.model.train()
        for epoch in range(params['epochs']):
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            avg_loss = total_loss / len(loader)
            acc = correct / total
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{params['epochs']}: "
                      f"Loss={avg_loss:.4f}, Acc={acc:.4f}")

        self.fitted = True
        print(f"━━━ Stage 3: Complete ━━━\n")

    def predict_proba(self, X):
        """Return attack probabilities for each sample."""
        if not self.fitted or self.model is None:
            # Return uniform distribution if not trained
            return np.ones((len(X), len(config.ATTACK_LABELS))) / len(config.ATTACK_LABELS)

        self.model.eval()
        X_seq, _ = self._create_sequences(X)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # Pad to match original length
        if len(probs) < len(X):
            padding = np.ones((len(X) - len(probs), probs.shape[1])) / probs.shape[1]
            probs = np.vstack([padding, probs])

        return probs

    def temporal_score(self, X):
        """Return temporal threat score for each sample."""
        probs = self.predict_proba(X)
        # Score = 1 - P(Normal)
        normal_idx = config.LABEL_TO_INT['Normal']
        scores = 1.0 - probs[:, normal_idx]
        return scores

    def save(self, path=None):
        if self.model is None:
            print("  ⚠ No model to save (not trained yet)")
            return
        path = path or os.path.join(config.MODEL_DIR,
                                     f"{self.model_type}_temporal.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'window_size': self.window_size,
        }, path)

    def load(self, path=None):
        path = path or os.path.join(config.MODEL_DIR,
                                     f"{self.model_type}_temporal.pt")
        checkpoint = torch.load(path, map_location=self.device)
        self.model_type = checkpoint['model_type']
        self.window_size = checkpoint['window_size']

        # Reconstruct model architecture
        n_features = config.TOP_K_FEATURES
        params = config.LSTM_PARAMS
        n_classes = len(config.ATTACK_LABELS)

        if self.model_type == 'lstm':
            self.model = LSTMModel(n_features, params['hidden_size'],
                                    params['num_layers'], n_classes,
                                    params['dropout']).to(self.device)
        else:
            self.model = GRUModel(n_features, params['hidden_size'],
                                   params['num_layers'], n_classes,
                                   params['dropout']).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        self.fitted = True
