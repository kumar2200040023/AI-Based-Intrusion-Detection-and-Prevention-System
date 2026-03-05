# 🛡 AI-Based IDS

**A Lightweight, Distributed, Self-Learning Intrusion Detection System**

AI-Based IDS is a multi-stage intelligent detection system that detects threats in milliseconds, self-learns from feedback, integrates with SIEM/SOAR, reduces False Positive Rate, and uses a hybrid ML pipeline.

---

## 🏗 Architecture

```
[Network Traffic]
        ↓
[Feature Extraction Engine]
        ↓
[Stage 1: Unsupervised Outlier Detection]   ← Isolation Forest + Autoencoder + One-Class SVM
        ↓
[Stage 2: Supervised ML Classification]     ← Random Forest + XGBoost (Ensemble Voting)
        ↓
[Stage 3: Temporal Pattern Detection]       ← LSTM / GRU (Bidirectional + Attention)
        ↓
[Decision Fusion Engine]                    ← Weighted: 0.3×Anomaly + 0.4×Classifier + 0.3×Temporal
        ↓
[Human-in-the-Loop Feedback]               ← Analyst review → Online learning
        ↓
[SIEM / SOAR Integration]                  ← Splunk / QRadar / Sentinel / XSOAR
```

---

## 📁 Project Structure

```
ids in AG/
├── README.md
├── requirements.txt
├── config.py                    # Central configuration
├── train.py                     # End-to-end training pipeline
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── dataset_loader.py        # NSL-KDD auto-download & loading
│   ├── preprocess.py            # Feature engineering pipeline
│   └── processed/               # Preprocessed data (generated)
├── models/
│   ├── stage1_anomaly.py        # Unsupervised anomaly ensemble
│   ├── stage2_classifier.py     # RF + XGBoost ensemble voter
│   └── stage3_temporal.py       # LSTM/GRU temporal detector
├── engine/
│   ├── fusion.py                # Decision fusion + dynamic threshold
│   └── feedback.py              # Online learning + retraining
├── api/
│   ├── main.py                  # FastAPI REST endpoints
│   ├── schemas.py               # Pydantic request/response models
│   └── siem.py                  # Syslog + Webhook integration
├── dashboard/
│   └── app.py                   # Streamlit dashboard (6 pages)
└── saved_models/                # Trained models (generated)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train.py
```

This will:
- Download & preprocess the NSL-KDD dataset
- Train Stage 1 (Anomaly detectors)
- Train Stage 2 (Random Forest + XGBoost)
- Train Stage 3 (LSTM temporal detector)
- Save all models to `saved_models/`

### 3. Start the API Server

```bash
python -m api.main
# or
uvicorn api.main:app --reload --port 8000
```

API available at: http://localhost:8000/docs

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard at: http://localhost:8501

---

## 🐳 Docker Deployment

```bash
# Core services (API + Dashboard + MongoDB)
docker-compose up -d

# With Kafka streaming
docker-compose --profile streaming up -d
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Detect threats in network features |
| `/detect/batch` | POST | Batch detection for multiple samples |
| `/feedback` | POST | Submit analyst feedback |
| `/alerts` | GET | Retrieve recent alerts (SIEM polling) |
| `/health` | GET | System health check |
| `/model/status` | GET | Model version & performance stats |

### Example: Detect

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.5, 0.3, 0.8, 0.2, 0.1, 0.4, 0.6, 0.9, 0.1,
                 0.3, 0.5, 0.7, 0.2, 0.4, 0.1, 0.8, 0.3, 0.5, 0.6],
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1"
  }'
```

### Response (SIEM-compatible)

```json
{
  "alert_id": "a1b2c3d4e5f6",
  "timestamp": "2026-02-26T08:00:00Z",
  "threat_score": 0.7823,
  "label": "THREAT",
  "attack_type": "DoS",
  "confidence_score": 0.9134,
  "is_threat": true,
  "components": {
    "anomaly_score": 0.7456,
    "classifier_score": 0.9134,
    "temporal_score": 0.6521
  }
}
```

---

## 🧪 Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Capture | Scapy / Wireshark |
| Stream Processing | Apache Kafka |
| ML Pipeline | Scikit-learn + XGBoost + PyTorch |
| API | FastAPI + Uvicorn |
| Database | MongoDB |
| Dashboard | Streamlit + Plotly |
| Explainability | SHAP |
| Integration | REST API + Syslog + Webhooks |
| Deployment | Docker + Docker Compose |

---

## 📊 Dataset

- **NSL-KDD** (default, auto-downloaded)
- CICIDS2017 (supported)
- UNSW-NB15 (supported)

---

## ⚠️ Key Challenges & Mitigation

| Challenge | Problem | Mitigation |
|-----------|---------|-----------|
| **Encrypted Traffic** | ML cannot inspect payload | Flow-based features (packet length, timing) |
| **Zero-Day Attacks** | Supervised models fail on new threats | Unsupervised Autoencoders + Online Learning |
| **Hardware Latency** | Python bottlenecks | C++ extensions, ONNX Runtime, GPU acceleration |

---

## 📈 Why This Project Excels

✔ Hybrid ML + Deep Learning pipeline  
✔ Real-time streaming architecture  
✔ Online learning with human-in-the-loop  
✔ SIEM/SOAR integration  
✔ Dynamic threshold for minimal FPR  
✔ SHAP explainability  
✔ Enterprise-level deployment (Docker)  
✔ Beautiful dashboard visualization  

---

## 📄 License

MIT License — Built for academic and research purposes.
