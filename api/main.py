"""
AI-Based IDS — FastAPI REST API
=====================================
Real-time detection, feedback, alerts, and SIEM endpoints.
"""

import os
import sys

# ─── STREAMLIT COMMUNITY CLOUD FIX ───────────────────────
# If the user mapped Streamlit Cloud to run api/main.py by accident,
# we intercept it here to run the dashboard instead so it doesn't crash.
is_streamlit = False
if "streamlit" in sys.modules:
    is_streamlit = True
elif os.environ.get("USER") in ["appuser", "adminuser"]:
    is_streamlit = True

if is_streamlit:
    import streamlit as st
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard', 'app.py')
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), dashboard_path, 'exec'), globals())
    st.stop()
# ─────────────────────────────────────────────────────────

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
from engine.ips import IPSEngine
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
ips_engine = IPSEngine(mode=config.IPS_MODE, block_duration=config.IPS_BLOCK_DURATION)
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
    
    # ── IPS Fast Path: Drop if blocked ──
    if config.IPS_ENABLED and ips_engine.is_blocked(request.src_ip):
        raise HTTPException(
            status_code=403, 
            detail=f"IPS BLOCK: Traffic from {request.src_ip} is currently dropped."
        )

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

    # ── IPS Active Defense ──
    if config.IPS_ENABLED and v.is_threat and v.score >= config.IPS_AUTO_BLOCK_THRESHOLD:
        ips_engine.block_ip(
            request.src_ip, 
            reason=f"Auto-blocked: {v.attack_type} attack with threat score {v.score:.4f}"
        )

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


# ─── IPS Endpoints ────────────────────────────────────
@app.get("/ips/blocklist")
async def get_ips_blocklist():
    """🛡 Retrieve current IPS blocklist."""
    return {
        "blocklist": ips_engine.get_blocklist(),
        "mode": ips_engine.mode,
        "enabled": config.IPS_ENABLED
    }

from pydantic import BaseModel
class UnblockRequest(BaseModel):
    ip_address: str

@app.post("/ips/unblock")
async def ips_unblock(request: UnblockRequest):
    """🔓 Formally unblock an IP address."""
    success = ips_engine.unblock_ip(request.ip_address)
    if success:
        return {"status": "success", "message": f"IP {request.ip_address} unblocked."}
    raise HTTPException(status_code=404, detail=f"IP {request.ip_address} not found in blocklist.")


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
