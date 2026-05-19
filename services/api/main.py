"""
API Gateway Service — The single public-facing FastAPI service.

This is the entry point for all external requests. It orchestrates:
1. POST /get-coupon  → calls Recommender (filter) → Agent (decide) → returns coupon
2. POST /feedback    → calls Feedback service (record + trigger agent update)
3. GET  /metrics     → calls Feedback service for aggregated metrics
4. GET  /health      → health check across all services

Architecture flow:
  Client → API → Recommender (filter coupons) → Agent (select best) → Client
  Client → API → Feedback (record + reward) → Agent (update Q-table)
"""
import sys
sys.path.insert(0, "/app")

import os
import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import uvicorn

from shared.database import get_db, init_db
from shared.schemas import (
    CouponRequest, CouponResponse,
    FeedbackRequest, FeedbackResponse,
    MetricsResponse, HealthResponse, CouponInfo
)
from shared.logger import get_logger, LogContext

logger = get_logger(__name__, "api-gateway")

# Internal service URLs
RECOMMENDER_URL = os.getenv("RECOMMENDER_URL", "http://recommender:8002")
AGENT_URL = os.getenv("AGENT_URL", "http://agent:8001")
FEEDBACK_URL = os.getenv("FEEDBACK_URL", "http://feedback:8003")
GENERATOR_URL = os.getenv("GENERATOR_URL", "http://generator:8004")

app = FastAPI(
    title="Agentic Coupon Recommender API",
    description="""
    ## Real-Time AI Coupon Recommendation System

    This API exposes a self-learning coupon recommendation engine powered by a
    Contextual Multi-Armed Bandit (Epsilon-Greedy) agent that continuously improves
    recommendations based on user interactions.

    ### How it works
    1. **POST /get-coupon** — System observes user context, agent selects optimal coupon
    2. **POST /feedback** — User interaction is recorded, agent updates its policy
    3. **GET /metrics** — Monitor CTR, conversion rate, and agent learning progress
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("API Gateway starting up")
    init_db()
    logger.info("API Gateway ready — all services should be available")


# ─── Core Endpoints ──────────────────────────────────────────────────────────

@app.post("/get-coupon", response_model=CouponResponse, tags=["Core"])
async def get_coupon(request: CouponRequest):
    """
    ## Get Personalized Coupon Recommendation

    The system:
    1. Filters coupons by user category (Recommender Service)
    2. Agent selects optimal coupon using epsilon-greedy policy (Agent Service)
    3. Returns coupon with metadata including whether it was an exploration decision

    **Reward Structure:**
    - Ignore: 0 points
    - Click: +1 point
    - Purchase: +5 points
    """
    logger.info(
        "Coupon request received",
        extra=LogContext(
            user_id=request.user_id,
            age=request.age,
            category=request.category,
            session_time=request.session_time
        ).as_extra()
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Step 1: Get candidate coupons from Recommender
        try:
            rec_response = await client.post(
                f"{RECOMMENDER_URL}/filter-coupons",
                json={
                    "user_id": request.user_id,
                    "category": request.category,
                    "age": request.age,
                    "session_time": request.session_time,
                    "max_results": 8
                }
            )
            rec_response.raise_for_status()
            candidates: list = rec_response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Recommender error: {e}")
            raise HTTPException(status_code=502, detail="Recommender service error")
        except Exception as e:
            logger.error(f"Recommender unreachable: {e}")
            raise HTTPException(status_code=503, detail="Recommender service unavailable")

        if not candidates:
            raise HTTPException(status_code=404, detail="No coupons available for this category")

        coupon_codes = [c["coupon_code"] for c in candidates]
        coupon_map = {c["coupon_code"]: c for c in candidates}

        # Step 2: Agent selects best coupon
        try:
            agent_response = await client.post(
                f"{AGENT_URL}/decide",
                json={
                    "user_id": request.user_id,
                    "age": request.age,
                    "category": request.category,
                    "session_time": request.session_time,
                    "available_coupons": coupon_codes
                }
            )
            agent_response.raise_for_status()
            decision = agent_response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Agent error: {e}")
            raise HTTPException(status_code=502, detail="Agent service error")
        except Exception as e:
            logger.error(f"Agent unreachable: {e}")
            raise HTTPException(status_code=503, detail="Agent service unavailable")

    selected_coupon_code = decision["coupon_code"]
    selected_coupon = coupon_map.get(selected_coupon_code, candidates[0])

    # Normalize confidence score (0-1)
    q_value = decision.get("q_value", 0.0)
    agent_confidence = min(1.0, max(0.0, q_value / 5.0))  # 5.0 is max reward

    logger.info(
        "Coupon recommendation served",
        extra=LogContext(
            user_id=request.user_id,
            recommended_coupon=selected_coupon_code,
            was_exploration=decision.get("was_exploration", False),
            agent_confidence=round(agent_confidence, 3),
            epsilon=decision.get("epsilon", 0.0)
        ).as_extra()
    )

    return CouponResponse(
        user_id=request.user_id,
        coupon_code=selected_coupon_code,
        discount_percent=selected_coupon["discount_percent"],
        category=selected_coupon["category"],
        description=selected_coupon.get("description", ""),
        was_exploration=decision.get("was_exploration", False),
        agent_confidence=round(agent_confidence, 3)
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Core"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    ## Submit User Interaction Feedback

    Records the outcome of a coupon recommendation. The agent will immediately
    update its policy based on the reward signal.

    **Reward Mapping:**
    - `clicked=0, purchased=0` → reward = 0 (no engagement)
    - `clicked=1, purchased=0` → reward = +1 (engagement)
    - `clicked=1, purchased=1` → reward = +5 (conversion)
    """
    logger.info(
        "Feedback submitted",
        extra=LogContext(
            user_id=feedback.user_id,
            coupon_code=feedback.coupon_code,
            clicked=feedback.clicked,
            purchased=feedback.purchased
        ).as_extra()
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                f"{FEEDBACK_URL}/record",
                json=feedback.dict()
            )
            response.raise_for_status()
            return FeedbackResponse(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"Feedback service error: {e}")
            raise HTTPException(status_code=502, detail="Feedback service error")
        except Exception as e:
            logger.error(f"Feedback service unreachable: {e}")
            raise HTTPException(status_code=503, detail="Feedback service unavailable")


@app.get("/metrics", response_model=MetricsResponse, tags=["Analytics"])
async def get_metrics():
    """
    ## System Performance Metrics

    Returns real-time metrics including:
    - **CTR** (Click-Through Rate): clicks / impressions
    - **Conversion Rate**: purchases / impressions
    - **Average Reward**: mean reward per interaction
    - **Exploration Ratio**: % of decisions that were random exploration
    - **Top Coupons**: best performing coupons by average reward
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{FEEDBACK_URL}/metrics")
            response.raise_for_status()
            return MetricsResponse(**response.json())
        except Exception as e:
            logger.error(f"Metrics fetch failed: {e}")
            raise HTTPException(status_code=503, detail="Metrics service unavailable")


@app.get("/agent/stats", tags=["Analytics"])
async def get_agent_stats():
    """Returns the agent's current Q-table statistics."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{AGENT_URL}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Agent stats unavailable: {e}")


@app.get("/agent/epsilon", tags=["Analytics"])
async def get_epsilon():
    """Returns current epsilon (exploration rate)."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{AGENT_URL}/epsilon")
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.post("/generate-data", tags=["Dev Tools"])
async def generate_data(n_users: int = 50):
    """
    ## Generate Synthetic Training Data

    Trigger the data generator to create synthetic user interactions.
    Useful for bootstrapping the agent or testing.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{GENERATOR_URL}/generate",
                params={"n_users": n_users}
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Generator unavailable: {e}")


@app.post("/seed-database", tags=["Dev Tools"])
async def seed_database(n_interactions: int = 200):
    """Seed the database with historical interactions to bootstrap agent learning."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{GENERATOR_URL}/seed-history",
                params={"n_interactions": n_interactions}
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Generator unavailable: {e}")


@app.get("/metrics/history", tags=["Analytics"])
async def metrics_history(limit: int = 50):
    """Returns historical metrics snapshots for trend analysis."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{FEEDBACK_URL}/history", params={"limit": limit})
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.get("/coupons", tags=["Catalog"])
async def list_coupons(category: str = None):
    """List all available coupons."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {}
        if category:
            params["category"] = category
        try:
            response = await client.get(f"{RECOMMENDER_URL}/coupons", params=params)
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))


@app.get("/health", response_model=dict, tags=["System"])
async def health_check():
    """
    ## Health Check

    Checks connectivity to all downstream services.
    """
    services = {
        "api": "healthy",
        "agent": AGENT_URL,
        "recommender": RECOMMENDER_URL,
        "feedback": FEEDBACK_URL,
        "generator": GENERATOR_URL
    }

    statuses = {"api": "healthy"}

    async with httpx.AsyncClient(timeout=3.0) as client:
        for service_name, url in list(services.items())[1:]:
            try:
                resp = await client.get(f"{url}/health")
                statuses[service_name] = "healthy" if resp.status_code == 200 else "degraded"
            except Exception:
                statuses[service_name] = "unreachable"

    overall = "healthy" if all(v == "healthy" for v in statuses.values()) else "degraded"
    return {
        "status": overall,
        "services": statuses,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
