"""
AI-Based IDS — Central Configuration
=========================================
All system parameters, model paths, thresholds, and integration settings.
"""

import os

# ─── Base Paths ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create dirs
for d in [PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset Configuration ───────────────────────────────
DATASET_NAME = "NSL-KDD"  # Options: NSL-KDD, CICIDS2017, UNSW-NB15
RANDOM_STATE = 42
TEST_SIZE = 0.25
TOP_K_FEATURES = 20

# Attack type mapping (NSL-KDD)
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'snmpgetattack': 'R2L',
    'named': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'sendmail': 'R2L', 'httptunnel': 'R2L', 'worm': 'R2L',
    'snmpguess': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
    'httptunnel': 'U2R',
}

ATTACK_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
LABEL_TO_INT = {label: i for i, label in enumerate(ATTACK_LABELS)}
INT_TO_LABEL = {i: label for i, label in enumerate(ATTACK_LABELS)}

# NSL-KDD column names
KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# ─── Stage 1: Anomaly Detection ──────────────────────────
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 150,
    'contamination': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

AUTOENCODER_PARAMS = {
    'encoding_dim': 14,
    'epochs': 50,
    'batch_size': 256,
    'validation_split': 0.1,
}

OCSVM_PARAMS = {
    'kernel': 'rbf',
    'gamma': 'scale',
    'nu': 0.1,
}

ANOMALY_THRESHOLD = 0.65  # Anomaly score threshold

# ─── Stage 2: Supervised Classification ───────────────────
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced',
}

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
}

ENSEMBLE_WEIGHTS = {'rf': 0.4, 'xgb': 0.6}

# ─── Stage 3: Temporal Detection (LSTM/GRU) ──────────────
SEQUENCE_WINDOW = 50  # Packets per sequence
LSTM_PARAMS = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 30,
    'batch_size': 64,
}

# ─── Decision Fusion Engine ──────────────────────────────
FUSION_WEIGHTS = {
    'anomaly': 0.3,
    'classifier': 0.4,
    'temporal': 0.3,
}
DYNAMIC_THRESHOLD_INIT = 0.55
THRESHOLD_DECAY = 0.01  # Adjustment step

# ─── Online Learning ─────────────────────────────────────
RETRAIN_INTERVAL_HOURS = 24
MIN_FEEDBACK_FOR_RETRAIN = 50

# ─── MongoDB ──────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = "antigravity_ids"
MONGO_ALERTS_COLLECTION = "alerts"
MONGO_FEEDBACK_COLLECTION = "feedback"
MONGO_METRICS_COLLECTION = "metrics"

# ─── Kafka ────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_INGEST = "network_traffic"
KAFKA_TOPIC_ALERTS = "threat_alerts"

# ─── API ──────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_VERSION = "1.0.0"

# ─── SIEM Integration ────────────────────────────────────
SYSLOG_HOST = os.environ.get("SYSLOG_HOST", "localhost")
SYSLOG_PORT = int(os.environ.get("SYSLOG_PORT", "514"))
WEBHOOK_URL = os.environ.get("SIEM_WEBHOOK_URL", "")

# ─── Key Challenges & Mitigation ─────────────────────────
CHALLENGES = [
    {
        "title": "Encrypted Traffic",
        "problem": "ML cannot inspect payload.",
        "mitigation": "Focus on Flow-based features (packet length, timing).",
    },
    {
        "title": "Zero-Day Attacks",
        "problem": "Supervised models fail on new threats.",
        "mitigation": "Implementing Unsupervised Autoencoders.",
    },
    {
        "title": "Hardware Latency",
        "problem": "Python bottlenecks.",
        "mitigation": "Offloading preprocessing to C++ extensions or specialized GPUs.",
    },
]
