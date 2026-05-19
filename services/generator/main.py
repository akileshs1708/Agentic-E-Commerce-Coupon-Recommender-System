"""
Data Generator Service — Creates synthetic e-commerce interaction data.

Simulates realistic user shopping behavior:
- Users have age-correlated category preferences
- Session time influences purchase probability
- Higher discounts increase click probability
- The generated data pre-trains the agent with historical interactions
"""
import sys
sys.path.insert(0, "/app")

import random
import asyncio
import httpx
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from shared.database import get_db, init_db
from shared.models import UserProfile, Interaction
from shared.logger import get_logger, LogContext
from shared.cache import cache_set, cache_get

logger = get_logger(__name__, "generator-service")

FEEDBACK_SERVICE_URL = "http://feedback:8003"
RECOMMENDER_SERVICE_URL = "http://recommender:8002"
AGENT_SERVICE_URL = "http://agent:8001"

app = FastAPI(
    title="Data Generator Service",
    description="Synthetic e-commerce interaction data generation",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Synthetic user population config ────────────────────────────────────────

CATEGORIES = ["electronics", "fashion", "groceries", "home", "books", "sports"]

# Age-category affinity matrix (probability weights)
AGE_CATEGORY_AFFINITY = {
    "18_25": {"electronics": 0.35, "fashion": 0.30, "books": 0.10, "sports": 0.15, "groceries": 0.05, "home": 0.05},
    "26_35": {"electronics": 0.25, "fashion": 0.20, "home": 0.20, "sports": 0.15, "groceries": 0.10, "books": 0.10},
    "36_45": {"electronics": 0.20, "home": 0.25, "groceries": 0.20, "fashion": 0.15, "sports": 0.10, "books": 0.10},
    "46_55": {"home": 0.25, "groceries": 0.25, "books": 0.20, "electronics": 0.15, "fashion": 0.10, "sports": 0.05},
    "56_plus": {"groceries": 0.30, "home": 0.25, "books": 0.25, "electronics": 0.10, "fashion": 0.05, "sports": 0.05},
}

# Coupon catalog (mirrors recommender seed data)
COUPONS_BY_CATEGORY = {
    "electronics": ["TECH10", "TECH20", "GADGET15"],
    "fashion":     ["STYLE10", "FASHION25", "TRENDY5"],
    "groceries":   ["FRESH10", "GROCERY15"],
    "home":        ["HOME20", "DECOR10"],
    "books":       ["READ10", "BOOKWORM20"],
    "sports":      ["FIT15", "ACTIVE20"],
    "all":         ["SAVE5", "WELCOME10"]
}

DISCOUNT_MAP = {
    "TECH10": 10, "TECH20": 20, "GADGET15": 15,
    "STYLE10": 10, "FASHION25": 25, "TRENDY5": 5,
    "FRESH10": 10, "GROCERY15": 15,
    "HOME20": 20, "DECOR10": 10,
    "READ10": 10, "BOOKWORM20": 20,
    "FIT15": 15, "ACTIVE20": 20,
    "SAVE5": 5, "WELCOME10": 10
}


def _get_age_bucket(age: int) -> str:
    if age <= 25: return "18_25"
    elif age <= 35: return "26_35"
    elif age <= 45: return "36_45"
    elif age <= 55: return "46_55"
    else: return "56_plus"


def generate_user(user_id: str) -> dict:
    """Generate a synthetic user profile with realistic demographics."""
    age = random.randint(18, 70)
    age_bucket = _get_age_bucket(age)
    affinity = AGE_CATEGORY_AFFINITY[age_bucket]
    categories = list(affinity.keys())
    weights = list(affinity.values())
    category = random.choices(categories, weights=weights, k=1)[0]
    return {
        "user_id": user_id,
        "age": age,
        "preferred_category": category,
        "total_sessions": random.randint(1, 50)
    }


def simulate_interaction(user: dict, coupon_code: str) -> dict:
    """
    Simulate realistic user behavior using a probabilistic model:
    - Click probability increases with coupon discount and session time
    - Purchase probability is conditional on click

    Base click rates:
    - Category match: 25-40% base rate
    - Discount bonus: +1% per 2% discount
    - Session time bonus: longer sessions = more engagement

    Purchase rate:
    - ~30% of clicks convert to purchases
    """
    discount = DISCOUNT_MAP.get(coupon_code, 5)
    session_time = random.uniform(30, 600)  # 30s to 10min

    # Click probability model
    base_click_prob = 0.2
    discount_bonus = discount * 0.008         # +0.8% per discount point
    session_bonus = min(0.15, session_time / 4000)  # max 15% bonus

    # Category match bonus
    coupon_cats = [cat for cat, codes in COUPONS_BY_CATEGORY.items() if coupon_code in codes]
    category_match = user["preferred_category"] in coupon_cats or "all" in coupon_cats
    category_bonus = 0.10 if category_match else -0.05

    click_prob = min(0.75, base_click_prob + discount_bonus + session_bonus + category_bonus)
    clicked = 1 if random.random() < click_prob else 0

    # Purchase conditional on click
    purchase_prob = 0.28 if clicked else 0.0
    purchase_prob += discount * 0.003  # higher discount = more likely to buy
    purchased = 1 if (clicked == 1 and random.random() < min(0.55, purchase_prob)) else 0

    reward = 5.0 if purchased else (1.0 if clicked else 0.0)

    return {
        "user_id": user["user_id"],
        "coupon_code": coupon_code,
        "age": user["age"],
        "category": user["preferred_category"],
        "session_time": round(session_time, 2),
        "clicked": clicked,
        "purchased": purchased,
        "reward": reward
    }


async def generate_batch(n_users: int = 50, db: Session = None) -> dict:
    """
    Generate a batch of synthetic interactions and submit them to the feedback service.
    Also creates user profiles in the DB if they don't exist.
    """
    generated = []
    errors = []

    # Determine starting user ID
    existing_count = db.query(UserProfile).count() if db else 0

    # Get available coupons from recommender
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{RECOMMENDER_SERVICE_URL}/coupons")
            all_coupons = [c["coupon_code"] for c in resp.json()] if resp.status_code == 200 else list(DISCOUNT_MAP.keys())
    except Exception:
        all_coupons = list(DISCOUNT_MAP.keys())

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(n_users):
            user_id = f"user_{existing_count + i + 1:05d}"
            user = generate_user(user_id)

            # Save user profile
            if db:
                existing = db.query(UserProfile).filter(
                    UserProfile.user_id == user_id
                ).first()
                if not existing:
                    profile = UserProfile(
                        user_id=user["user_id"],
                        age=user["age"],
                        preferred_category=user["preferred_category"],
                        total_sessions=user["total_sessions"]
                    )
                    db.add(profile)

            # Pick a random coupon to show this user
            coupon = random.choice(all_coupons)
            interaction = simulate_interaction(user, coupon)
            generated.append(interaction)

            # Submit to feedback service
            try:
                resp = await client.post(
                    f"{FEEDBACK_SERVICE_URL}/record",
                    json={**interaction, "session_time": interaction["session_time"]}
                )
                if resp.status_code != 200:
                    errors.append(f"Feedback failed for {user_id}")
            except Exception as e:
                errors.append(str(e))

        if db:
            db.commit()

    logger.info(
        "Batch generation complete",
        extra=LogContext(
            n_generated=len(generated),
            n_errors=len(errors)
        ).as_extra()
    )

    return {
        "generated": len(generated),
        "errors": len(errors),
        "error_details": errors[:5] if errors else []
    }


@app.on_event("startup")
async def startup():
    logger.info("Data generator service starting")
    init_db()
    logger.info("Data generator service ready")


@app.post("/generate")
async def trigger_generation(
    n_users: int = 50,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Trigger synthetic data generation.
    n_users: number of user interactions to generate
    """
    logger.info(f"Generating {n_users} synthetic interactions")
    result = await generate_batch(n_users=n_users, db=db)
    return result


@app.post("/seed-history")
async def seed_historical_data(
    n_interactions: int = 500,
    db: Session = Depends(get_db)
):
    """
    Seed the system with historical interaction data to bootstrap agent learning.
    Useful for cold-start: agent performs better with some prior data.
    """
    logger.info(f"Seeding {n_interactions} historical interactions")
    batches = n_interactions // 50
    remainder = n_interactions % 50
    total_generated = 0

    for _ in range(batches):
        result = await generate_batch(n_users=50, db=db)
        total_generated += result["generated"]
        await asyncio.sleep(0.5)  # Rate limiting

    if remainder > 0:
        result = await generate_batch(n_users=remainder, db=db)
        total_generated += result["generated"]

    return {
        "status": "historical data seeded",
        "total_interactions": total_generated
    }


@app.get("/stats")
async def generation_stats(db: Session = Depends(get_db)):
    """Returns data generation statistics."""
    user_count = db.query(UserProfile).count()
    interaction_count = db.query(Interaction).count()
    return {
        "total_users": user_count,
        "total_interactions": interaction_count,
        "timestamp": datetime.utcnow()
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "generator-service"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=False)
