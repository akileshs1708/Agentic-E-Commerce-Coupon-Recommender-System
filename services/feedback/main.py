"""
Feedback Service — Handles all user interaction feedback:
1. Records interaction results (click/purchase/ignore) to PostgreSQL
2. Computes rewards and forwards to agent for Q-table updates
3. Maintains running metrics (CTR, conversion rate, avg reward)
4. Provides metrics snapshots for monitoring
"""
import sys
sys.path.insert(0, "/app")

import httpx
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any
import uvicorn

from shared.database import get_db, init_db
from shared.models import Interaction, MetricsSnapshot
from shared.schemas import FeedbackRequest, FeedbackResponse, MetricsResponse, HealthResponse
from shared.logger import get_logger, LogContext
from shared.cache import cache_get, cache_set, cache_increment, cache_hset, cache_hgetall

logger = get_logger(__name__, "feedback-service")

AGENT_SERVICE_URL = "http://agent:8001"

app = FastAPI(
    title="Feedback Service",
    description="Reward processing and metrics aggregation",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    logger.info("Feedback service starting")
    init_db()
    logger.info("Feedback service ready")


async def _notify_agent(feedback: FeedbackRequest):
    """
    Background task: notify agent service to update Q-values.
    Fire-and-forget — feedback response doesn't wait for this.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/update",
                json={
                    "user_id": feedback.user_id,
                    "age": feedback.age,
                    "category": feedback.category,
                    "coupon_code": feedback.coupon_code,
                    "clicked": feedback.clicked,
                    "purchased": feedback.purchased
                }
            )
            if response.status_code != 200:
                logger.error(
                    "Agent update failed",
                    extra=LogContext(status_code=response.status_code).as_extra()
                )
    except Exception as e:
        logger.error(f"Failed to notify agent: {e}")


def _compute_reward(clicked: int, purchased: int) -> float:
    if purchased == 1:
        return 5.0
    elif clicked == 1:
        return 1.0
    return 0.0


def _update_redis_metrics(clicked: int, purchased: int, reward: float):
    """Atomically update Redis metric counters."""
    cache_increment("metrics:impressions")
    if clicked:
        cache_increment("metrics:clicks")
    if purchased:
        cache_increment("metrics:purchases")
    # Store reward as integer cents to avoid float precision issues
    cache_increment("metrics:total_reward_cents", int(reward * 100))
    cache_increment("metrics:feedback_count")


@app.post("/record", response_model=FeedbackResponse)
async def record_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Main feedback endpoint. Records interaction to DB and triggers agent learning.
    """
    reward = _compute_reward(feedback.clicked, feedback.purchased)

    logger.info(
        "Feedback received",
        extra=LogContext(
            user_id=feedback.user_id,
            coupon_code=feedback.coupon_code,
            clicked=feedback.clicked,
            purchased=feedback.purchased,
            reward=reward,
            category=feedback.category,
            age=feedback.age
        ).as_extra()
    )

    # Persist interaction to PostgreSQL
    interaction = Interaction(
        user_id=feedback.user_id,
        coupon_code=feedback.coupon_code,
        age=feedback.age,
        category=feedback.category,
        session_time=feedback.session_time,
        coupon_shown=True,
        clicked=feedback.clicked,
        purchased=feedback.purchased,
        reward=reward
    )
    db.add(interaction)
    db.commit()

    # Update Redis counters (non-blocking)
    _update_redis_metrics(feedback.clicked, feedback.purchased, reward)

    # Notify agent to update Q-table (background)
    background_tasks.add_task(_notify_agent, feedback)

    logger.info(
        "Interaction recorded",
        extra=LogContext(
            interaction_id=interaction.id,
            reward=reward
        ).as_extra()
    )

    return FeedbackResponse(
        user_id=feedback.user_id,
        coupon_code=feedback.coupon_code,
        reward=reward,
        message=f"Feedback recorded. Reward: {reward}"
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(db: Session = Depends(get_db)):
    """
    Returns comprehensive system metrics aggregated from PostgreSQL + Redis.
    Combines fast Redis counters with accurate DB aggregations.
    """
    # Fast path: read from Redis counters
    impressions = int(cache_get("metrics:impressions") or 0)
    clicks = int(cache_get("metrics:clicks") or 0)
    purchases = int(cache_get("metrics:purchases") or 0)
    total_reward_cents = int(cache_get("metrics:total_reward_cents") or 0)
    exploration = int(cache_get("metrics:exploration_count") or 0)
    exploitation = int(cache_get("metrics:exploitation_count") or 0)

    # Fallback to DB if Redis is cold
    if impressions == 0:
        result = db.query(
            func.count(Interaction.id).label("total"),
            func.sum(Interaction.clicked).label("clicks"),
            func.sum(Interaction.purchased).label("purchases"),
            func.sum(Interaction.reward).label("total_reward")
        ).first()
        impressions = result.total or 0
        clicks = int(result.clicks or 0)
        purchases = int(result.purchases or 0)
        total_reward_cents = int((result.total_reward or 0) * 100)

    total_reward = total_reward_cents / 100.0
    ctr = round(clicks / impressions, 4) if impressions > 0 else 0.0
    conversion_rate = round(purchases / impressions, 4) if impressions > 0 else 0.0
    avg_reward = round(total_reward / impressions, 4) if impressions > 0 else 0.0
    total_decisions = exploration + exploitation
    exploration_ratio = round(exploration / total_decisions, 4) if total_decisions > 0 else 0.0

    # Top performing coupons from DB
    top_coupons_raw = (
        db.query(
            Interaction.coupon_code,
            func.count(Interaction.id).label("shown"),
            func.sum(Interaction.clicked).label("clicks"),
            func.sum(Interaction.purchased).label("purchases"),
            func.avg(Interaction.reward).label("avg_reward")
        )
        .group_by(Interaction.coupon_code)
        .order_by(func.avg(Interaction.reward).desc())
        .limit(10)
        .all()
    )

    top_coupons = [
        {
            "coupon_code": r.coupon_code,
            "shown": r.shown,
            "clicks": int(r.clicks or 0),
            "purchases": int(r.purchases or 0),
            "avg_reward": round(float(r.avg_reward or 0), 3),
            "ctr": round(int(r.clicks or 0) / r.shown, 3) if r.shown > 0 else 0.0
        }
        for r in top_coupons_raw
    ]

    # Get current epsilon from agent service
    current_epsilon = 0.3  # default
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{AGENT_SERVICE_URL}/epsilon")
            if resp.status_code == 200:
                current_epsilon = resp.json().get("epsilon", 0.3)
    except Exception:
        pass

    return MetricsResponse(
        total_impressions=impressions,
        total_clicks=clicks,
        total_purchases=purchases,
        ctr=ctr,
        conversion_rate=conversion_rate,
        avg_reward=avg_reward,
        exploration_count=exploration,
        exploitation_count=exploitation,
        exploration_ratio=exploration_ratio,
        current_epsilon=current_epsilon,
        top_coupons=top_coupons
    )


@app.post("/snapshot")
async def save_metrics_snapshot(db: Session = Depends(get_db)):
    """Save current metrics as a time-series snapshot for historical tracking."""
    metrics = await get_metrics(db)
    snapshot = MetricsSnapshot(
        total_impressions=metrics.total_impressions,
        total_clicks=metrics.total_clicks,
        total_purchases=metrics.total_purchases,
        ctr=metrics.ctr,
        conversion_rate=metrics.conversion_rate,
        avg_reward=metrics.avg_reward,
        exploration_count=metrics.exploration_count,
        exploitation_count=metrics.exploitation_count,
        epsilon=metrics.current_epsilon
    )
    db.add(snapshot)
    db.commit()
    return {"status": "snapshot saved", "snapshot_id": snapshot.id}


@app.get("/history")
async def metrics_history(limit: int = 50, db: Session = Depends(get_db)):
    """Returns recent metrics snapshots for trend analysis."""
    snapshots = (
        db.query(MetricsSnapshot)
        .order_by(MetricsSnapshot.snapshot_time.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "time": s.snapshot_time,
            "impressions": s.total_impressions,
            "ctr": s.ctr,
            "conversion_rate": s.conversion_rate,
            "avg_reward": s.avg_reward,
            "epsilon": s.epsilon
        }
        for s in snapshots
    ]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", service="feedback-service")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False)
