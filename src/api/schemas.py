"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class DamageInfo(BaseModel):
    """Information about a detected damage."""
    damage_id: int
    bounding_box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    damage_type: str
    classification_confidence: float = Field(..., ge=0.0, le=1.0)
    vehicle_part: Optional[str] = None
    part_confidence: float = Field(0.0, ge=0.0, le=1.0)
    severity: str = Field(..., description="low, medium, or high")
    severity_score: float = Field(..., ge=0.0, le=1.0)
    damage_area_pixels: int = Field(..., ge=0)
    is_cosmetic: bool
    is_functional: bool
    cosmetic_confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class OverallAssessment(BaseModel):
    """Overall damage assessment summary."""
    max_severity: str
    has_functional_damage: bool
    total_damages: int = Field(..., ge=0)
    damage_types_count: Dict[str, int]
    requires_immediate_attention: bool


class DetectionResponse(BaseModel):
    """Response for damage detection."""
    damage_detected: bool
    num_damages: int = Field(0, ge=0)
    damages: List[DamageInfo] = []
    overall_assessment: Optional[OverallAssessment] = None
    image_shape: List[int]
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
