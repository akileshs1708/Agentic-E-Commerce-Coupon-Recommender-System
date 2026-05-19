"""
Pydantic schemas for request/response validation across all services.
These are the shared data contracts between microservices.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# ─── Request Schemas ────────────────────────────────────────────────────────

class CouponRequest(BaseModel):
    """Request body for POST /get-coupon"""
    user_id: str = Field(..., description="Unique user identifier")
    age: int = Field(..., ge=18, le=90, description="User age (18-90)")
    category: str = Field(..., description="Shopping category of interest")
    session_time: float = Field(..., ge=0, description="Current session duration in seconds")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_001",
                "age": 32,
                "category": "electronics",
                "session_time": 245.5
            }
        }


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback — records user reaction to shown coupon"""
    user_id: str = Field(..., description="User identifier")
    coupon_code: str = Field(..., description="Coupon that was shown")
    clicked: int = Field(..., ge=0, le=1, description="1 if user clicked, 0 otherwise")
    purchased: int = Field(..., ge=0, le=1, description="1 if user purchased, 0 otherwise")
    age: int = Field(..., ge=18, le=90)
    category: str
    session_time: float = Field(..., ge=0)

    @validator("purchased")
    def purchase_requires_click(cls, v, values):
        """A purchase logically requires a click first."""
        if v == 1 and values.get("clicked", 0) == 0:
            raise ValueError("purchased=1 requires clicked=1")
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_001",
                "coupon_code": "SAVE10",
                "clicked": 1,
                "purchased": 1,
                "age": 32,
                "category": "electronics",
                "session_time": 245.5
            }
        }


# ─── Response Schemas ────────────────────────────────────────────────────────

class CouponResponse(BaseModel):
    """Response for POST /get-coupon"""
    user_id: str
    coupon_code: str
    discount_percent: float
    category: str
    description: str
    was_exploration: bool   # True = random explore, False = exploit best known
    agent_confidence: float  # Q-value normalized 0-1
    message: str = "Coupon recommendation generated successfully"


class FeedbackResponse(BaseModel):
    """Response for POST /feedback"""
    user_id: str
    coupon_code: str
    reward: float
    message: str
    updated_q_value: Optional[float] = None


class MetricsResponse(BaseModel):
    """Response for GET /metrics"""
    total_impressions: int
    total_clicks: int
    total_purchases: int
    ctr: float                    # Click-Through Rate = clicks / impressions
    conversion_rate: float        # Conversion Rate = purchases / impressions
    avg_reward: float             # Average reward per interaction
    exploration_count: int
    exploitation_count: int
    exploration_ratio: float      # exploration / total decisions
    current_epsilon: float        # Current epsilon value
    top_coupons: List[Dict[str, Any]]  # Best performing coupons
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CouponInfo(BaseModel):
    """Internal coupon data transfer object"""
    coupon_code: str
    discount_percent: float
    category: str
    min_purchase_amount: float
    description: str
    is_active: bool


class AgentDecision(BaseModel):
    """Internal DTO from agent service to API service"""
    coupon_code: str
    q_value: float
    was_exploration: bool
    epsilon: float
    context_key: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
