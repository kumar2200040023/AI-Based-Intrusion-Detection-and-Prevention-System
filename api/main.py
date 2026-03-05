"""
AI-Based IDS — FastAPI REST API
=====================================
Real-time detection, feedback, alerts, and SIEM endpoints.
"""

import os
import sys
import uuid
import time
import logging
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from api.schemas import (
    DetectionRequest, BatchDetectionRequest, FeedbackRequest,
    DetectionResponse, FeedbackResponse, HealthResponse,
    ModelStatusResponse, ComponentScores,
)
from engine.fusion import DecisionFusionEngine
from engine.feedback import FeedbackStore, IncrementalLearner, RetrainingScheduler
from api.siem import SIEMIntegration

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s',
)
logger = logging.getLogger("antigravity.api")

# ─── App Init ─────────────────────────────────────────────
app = FastAPI(
    title="AI-Based IDS",
    description=(
        "🛡 Lightweight, distributed, self-learning Intrusion Detection System "
        "with multi-stage ML pipeline, real-time detection, and SIEM integration."
    ),
    version=config.MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────
START_TIME = time.time()
MODELS_LOADED = False

# Core engines
fusion_engine = DecisionFusionEngine()
feedback_store = FeedbackStore()
incremental_learner = IncrementalLearner()
retraining_scheduler = RetrainingScheduler(feedback_store, incremental_learner)
siem_integration = SIEMIntegration()

# ML Models (lazy loaded)
anomaly_ensemble = None
classifier_ensemble = None
temporal_engine = None


def _load_models():
    """Lazy load trained models."""
    global anomaly_ensemble, classifier_ensemble, temporal_engine, MODELS_LOADED

    if MODELS_LOADED:
        return True

    try:
        from models.stage1_anomaly import AnomalyEnsemble
        from models.stage2_classifier import EnsembleVoter
        from models.stage3_temporal import TemporalEngine

        anomaly_ensemble = AnomalyEnsemble()
        classifier_ensemble = EnsembleVoter()
        temporal_engine = TemporalEngine(model_type='lstm')

        # Try loading saved models
        try:
            anomaly_ensemble.load()
            classifier_ensemble.load()
            temporal_engine.load()
            MODELS_LOADED = True
            logger.info("✓ All models loaded successfully")
        except FileNotFoundError:
            logger.warning("⚠ Saved models not found. Run train.py first.")
            logger.info("  API will use demo mode with synthetic scores.")

        return MODELS_LOADED
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("━" * 50)
    logger.info("  🚀 AI-Based IDS API Starting...")
    logger.info("━" * 50)
    _load_models()
    siem_integration.initialize()
    logger.info(f"  Version: {config.MODEL_VERSION}")
    logger.info(f"  Docs: http://localhost:{config.API_PORT}/docs")
    logger.info("━" * 50)


# ─── Detection Endpoint ──────────────────────────────────
@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """
    🔍 Detect threats in network traffic features.

    Runs the full 3-stage pipeline:
    1. Unsupervised Anomaly Detection
    2. Supervised Classification (RF + XGBoost)
    3. Temporal Pattern Detection (LSTM)
    4. Decision Fusion → Threat Verdict
    """
    alert_id = str(uuid.uuid4())[:12]
    features = np.array([request.features], dtype=np.float32)

    if MODELS_LOADED and anomaly_ensemble is not None and classifier_ensemble is not None and temporal_engine is not None:
        # Real detection pipeline
        anomaly_scores = anomaly_ensemble.score(features)
        classifier_proba = classifier_ensemble.predict_proba(features)
        classifier_labels = classifier_ensemble.predict(features)
        temporal_scores = temporal_engine.temporal_score(features)
    else:
        # Demo mode — synthetic scores
        anomaly_scores = np.random.uniform(0.1, 0.9, size=1)
        n_classes = len(config.ATTACK_LABELS)
        classifier_proba = np.random.dirichlet(np.ones(n_classes), size=1)
        classifier_labels = np.argmax(classifier_proba, axis=1)
        temporal_scores = np.random.uniform(0.1, 0.8, size=1)

    # Fuse
    verdicts = fusion_engine.fuse(
        anomaly_scores, classifier_proba,
        temporal_scores, classifier_labels
    )
    v = verdicts[0]

    # SIEM alert
    siem_integration.send_alert(v)

    return DetectionResponse(
        alert_id=alert_id,
        timestamp=v.timestamp,
        threat_score=v.score,
        label=v.label,
        attack_type=v.attack_type,
        confidence_score=v.confidence,
        is_threat=v.is_threat,
        threshold=v.threshold,
        components=ComponentScores(
            anomaly_score=v.anomaly_score,
            classifier_score=v.classifier_score,
            temporal_score=v.temporal_score,
        ),
        src_ip=request.src_ip,
        dst_ip=request.dst_ip,
        model_version=config.MODEL_VERSION,
    )


# ─── Batch Detection ─────────────────────────────────────
@app.post("/detect/batch")
async def detect_batch(request: BatchDetectionRequest):
    """🔍 Batch detection for multiple samples."""
    results = []
    for sample in request.samples:
        result = await detect(sample)
        results.append(result)
    return {"results": results, "count": len(results)}


# ─── Feedback Endpoint ────────────────────────────────────
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    💬 Submit analyst feedback on a detection alert.

    This feeds the Human-in-the-Loop learning system:
    - Updates dynamic threshold
    - Stores feedback for retraining
    - Triggers incremental learning
    """
    # Store feedback
    feedback_store.store_feedback(
        alert_id=request.alert_id,
        feedback_type=request.feedback_type,
        true_label=request.true_label,
        analyst_id=request.analyst_id,
    )

    # Update fusion threshold
    fusion_engine.update_threshold(request.feedback_type)
    new_threshold = fusion_engine.threshold

    # Check if retraining needed
    retraining_scheduler.check_and_retrain()

    logger.info(
        f"Feedback: {request.feedback_type} for alert {request.alert_id} "
        f"→ threshold={new_threshold:.4f}"
    )

    return FeedbackResponse(
        status="accepted",
        alert_id=request.alert_id,
        feedback_type=request.feedback_type,
        threshold_updated=True,
        new_threshold=new_threshold,
    )


# ─── Alerts Endpoint (SIEM Polling) ──────────────────────
@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """📋 Retrieve recent alerts (for SIEM polling)."""
    alerts = siem_integration.get_recent_alerts(limit=limit)
    return {
        "alerts": alerts,
        "count": len(alerts),
        "total": siem_integration.get_alert_count(),
    }


# ─── Health Check ────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """💚 System health check."""
    return HealthResponse(
        status="healthy",
        model_version=config.MODEL_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 2),
        models_loaded=MODELS_LOADED,
    )


# ─── Model Status ────────────────────────────────────────
@app.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """📊 Detailed model and system status."""
    return ModelStatusResponse(
        model_version=config.MODEL_VERSION,
        models_loaded=MODELS_LOADED,
        fusion_stats=fusion_engine.get_stats(),
        feedback_stats=feedback_store.get_feedback_stats(),
        retraining_status=retraining_scheduler.get_status(),
        challenges=config.CHALLENGES,
    )


# ─── Run Server ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
