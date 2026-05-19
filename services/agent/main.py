"""
Agent Service — FastAPI microservice responsible for:
1. Receiving user context and returning the best coupon decision
2. Accepting feedback (rewards) and updating the Q-table
3. Exposing agent stats for monitoring
"""
import sys
import os
sys.path.insert(0, "/app")

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from shared.database import get_db, init_db
from shared.schemas import AgentDecision, HealthResponse
from shared.logger import get_logger, LogContext
from shared.cache import cache_get, cache_set, cache_increment
from services.agent.bandit import (
    get_context_key,
    get_current_epsilon,
    select_coupon,
    update_q_value,
    compute_reward,
    get_agent_stats
)
from pydantic import BaseModel
from typing import List
from datetime import datetime

logger = get_logger(__name__, "agent-service")

app = FastAPI(
    title="Agent Service",
    description="Epsilon-Greedy Contextual Bandit Decision Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup():
    logger.info("Agent service starting up — initializing database tables")
    init_db()
    logger.info("Agent service ready")


# ─── Request/Response schemas specific to this service ─────────────────────

class DecisionRequest(BaseModel):
    user_id: str
    age: int
    category: str
    session_time: float
    available_coupons: List[str]  # filtered list from recommender


class UpdateRequest(BaseModel):
    user_id: str
    age: int
    category: str
    coupon_code: str
    clicked: int
    purchased: int


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.post("/decide", response_model=AgentDecision)
async def decide(request: DecisionRequest, db: Session = Depends(get_db)):
    """
    Core decision endpoint. Given user context and candidate coupons,
    returns the agent's chosen coupon using epsilon-greedy policy.
    """
    logger.info(
        "Decision requested",
        extra=LogContext(
            user_id=request.user_id,
            age=request.age,
            category=request.category,
            n_candidates=len(request.available_coupons)
        ).as_extra()
    )

    if not request.available_coupons:
        raise HTTPException(status_code=400, detail="No available coupons provided")

    # Build context key from user features
    context_key = get_context_key(request.age, request.category)

    # Get current exploration rate
    epsilon = get_current_epsilon(db)

    # Agent selects action (coupon)
    coupon_code, q_value, was_exploration = select_coupon(
        db=db,
        context_key=context_key,
        available_coupons=request.available_coupons,
        epsilon=epsilon
    )

    # Track exploration/exploitation counters in Redis
    if was_exploration:
        cache_increment("metrics:exploration_count")
    else:
        cache_increment("metrics:exploitation_count")

    logger.info(
        "Agent decision made",
        extra=LogContext(
            user_id=request.user_id,
            context_key=context_key,
            chosen_coupon=coupon_code,
            q_value=round(q_value, 4),
            was_exploration=was_exploration,
            epsilon=round(epsilon, 4)
        ).as_extra()
    )

    return AgentDecision(
        coupon_code=coupon_code,
        q_value=q_value,
        was_exploration=was_exploration,
        epsilon=epsilon,
        context_key=context_key
    )


@app.post("/update")
async def update(request: UpdateRequest, db: Session = Depends(get_db)):
    """
    Update the agent's Q-table with observed reward from user feedback.
    This is the learning step — called after the user interacts (or ignores).
    """
    reward = compute_reward(request.clicked, request.purchased)
    context_key = get_context_key(request.age, request.category)

    logger.info(
        "Agent update received",
        extra=LogContext(
            user_id=request.user_id,
            coupon_code=request.coupon_code,
            clicked=request.clicked,
            purchased=request.purchased,
            reward=reward,
            context_key=context_key
        ).as_extra()
    )

    new_q = update_q_value(
        db=db,
        context_key=context_key,
        coupon_code=request.coupon_code,
        reward=reward
    )

    # Update rolling reward sum in Redis
    cache_increment("metrics:total_reward_x100", int(reward * 100))
    cache_increment("metrics:total_feedback_count")

    return {
        "status": "updated",
        "context_key": context_key,
        "coupon_code": request.coupon_code,
        "reward": reward,
        "new_q_value": round(new_q, 4)
    }


@app.get("/stats")
async def stats(db: Session = Depends(get_db)):
    """Returns agent learning statistics for monitoring."""
    return get_agent_stats(db)


@app.get("/epsilon")
async def current_epsilon(db: Session = Depends(get_db)):
    """Returns current exploration rate."""
    epsilon = get_current_epsilon(db)
    return {"epsilon": epsilon, "epsilon_min": 0.05, "epsilon_initial": 0.3}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", service="agent-service")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
