"""
AI-Based IDS — API Schemas
================================
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ─── Request Models ──────────────────────────────────────

class DetectionRequest(BaseModel):
    """Request body for /detect endpoint."""
    features: List[float] = Field(
        ...,
        description="Feature vector (top-20 features, scaled)",
        min_length=1,
    )
    src_ip: str = Field(default="0.0.0.0", description="Source IP address")
    dst_ip: str = Field(default="0.0.0.0", description="Destination IP address")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class BatchDetectionRequest(BaseModel):
    """Batch detection request."""
    samples: List[DetectionRequest]


class FeedbackRequest(BaseModel):
    """Analyst feedback on an alert."""
    alert_id: str = Field(..., description="ID of the alert being reviewed")
    feedback_type: str = Field(
        ...,
        description="Type: true_positive, false_positive, false_negative",
        pattern="^(true_positive|false_positive|false_negative)$",
    )
    true_label: Optional[int] = Field(
        default=None,
        description="Correct label index (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)"
    )
    analyst_id: str = Field(default="analyst_1", description="Analyst identifier")
    notes: Optional[str] = Field(default=None, description="Additional notes")


# ─── Response Models ─────────────────────────────────────

class ComponentScores(BaseModel):
    anomaly_score: float
    classifier_score: float
    temporal_score: float


class DetectionResponse(BaseModel):
    """Response from /detect endpoint."""
    alert_id: str
    timestamp: str
    threat_score: float
    label: str
    attack_type: str
    confidence_score: float
    is_threat: bool
    threshold: float
    components: ComponentScores
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    model_version: str


class SIEMAlert(BaseModel):
    """SIEM-compatible alert format."""
    timestamp: str
    src_ip: str
    dst_ip: str
    attack_type: str
    confidence_score: float
    threat_score: float
    model_version: str
    is_threat: bool


class FeedbackResponse(BaseModel):
    """Response from /feedback endpoint."""
    status: str
    alert_id: str
    feedback_type: str
    threshold_updated: bool
    new_threshold: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    models_loaded: bool


class ModelStatusResponse(BaseModel):
    model_version: str
    models_loaded: bool
    fusion_stats: dict
    feedback_stats: dict
    retraining_status: dict
    challenges: list
