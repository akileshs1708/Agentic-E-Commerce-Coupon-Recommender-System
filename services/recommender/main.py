"""
Recommendation Engine Service — FastAPI microservice that:
1. Filters coupons based on user context (category, age group)
2. Returns a ranked candidate list to the agent for final selection
3. Manages the coupon catalog (CRUD + seeding)

This service sits between the API and the agent, pre-filtering
irrelevant coupons so the agent has a focused action space.
"""
import sys
sys.path.insert(0, "/app")

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert

from typing import List, Optional
import uvicorn

from shared.database import get_db, init_db
from shared.models import Coupon
from shared.schemas import CouponInfo, HealthResponse
from shared.logger import get_logger, LogContext
from shared.cache import cache_get, cache_set
from pydantic import BaseModel
from datetime import datetime

logger = get_logger(__name__, "recommender-service")

app = FastAPI(
    title="Recommender Service",
    description="Coupon catalog management and context-based filtering",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Seed Coupons ─────────────────────────────────────────────────────────────

SEED_COUPONS = [
    # Electronics
    {"coupon_code": "TECH10", "discount_percent": 10.0, "category": "electronics", "min_purchase_amount": 50.0, "description": "10% off electronics over $50"},
    {"coupon_code": "TECH20", "discount_percent": 20.0, "category": "electronics", "min_purchase_amount": 100.0, "description": "20% off electronics over $100"},
    {"coupon_code": "GADGET15", "discount_percent": 15.0, "category": "electronics", "min_purchase_amount": 75.0, "description": "15% off all gadgets"},
    # Fashion
    {"coupon_code": "STYLE10", "discount_percent": 10.0, "category": "fashion", "min_purchase_amount": 30.0, "description": "10% off fashion items"},
    {"coupon_code": "FASHION25", "discount_percent": 25.0, "category": "fashion", "min_purchase_amount": 80.0, "description": "25% off fashion over $80"},
    {"coupon_code": "TRENDY5", "discount_percent": 5.0, "category": "fashion", "min_purchase_amount": 0.0, "description": "5% off any fashion item"},
    # Groceries
    {"coupon_code": "FRESH10", "discount_percent": 10.0, "category": "groceries", "min_purchase_amount": 20.0, "description": "10% off fresh groceries"},
    {"coupon_code": "GROCERY15", "discount_percent": 15.0, "category": "groceries", "min_purchase_amount": 40.0, "description": "15% off grocery haul"},
    # Home & Living
    {"coupon_code": "HOME20", "discount_percent": 20.0, "category": "home", "min_purchase_amount": 60.0, "description": "20% off home items"},
    {"coupon_code": "DECOR10", "discount_percent": 10.0, "category": "home", "min_purchase_amount": 25.0, "description": "10% off home decor"},
    # Books
    {"coupon_code": "READ10", "discount_percent": 10.0, "category": "books", "min_purchase_amount": 15.0, "description": "10% off any book order"},
    {"coupon_code": "BOOKWORM20", "discount_percent": 20.0, "category": "books", "min_purchase_amount": 50.0, "description": "20% off books over $50"},
    # Sports
    {"coupon_code": "FIT15", "discount_percent": 15.0, "category": "sports", "min_purchase_amount": 40.0, "description": "15% off sports gear"},
    {"coupon_code": "ACTIVE20", "discount_percent": 20.0, "category": "sports", "min_purchase_amount": 80.0, "description": "20% off active wear"},
    # Universal coupons (any category)
    {"coupon_code": "SAVE5", "discount_percent": 5.0, "category": "all", "min_purchase_amount": 0.0, "description": "5% off anything"},
    {"coupon_code": "WELCOME10", "discount_percent": 10.0, "category": "all", "min_purchase_amount": 0.0, "description": "10% welcome offer"},
]


@app.on_event("startup")
async def startup():
    logger.info("Recommender service starting — seeding coupon catalog")
    init_db()
    db = next(get_db())
    try:
        _seed_coupons(db)
    finally:
        db.close()
    logger.info(f"Recommender service ready with {len(SEED_COUPONS)} coupons")


def _seed_coupons(db: Session):
    """Insert seed coupons safely (no duplicates even with concurrency)."""
    
    for coupon_data in SEED_COUPONS:
        stmt = insert(Coupon).values(**coupon_data)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["coupon_code"]
        )
        db.execute(stmt)

    db.commit()
    logger.info("Coupon catalog seeded (idempotent)")


# ─── Request schemas ──────────────────────────────────────────────────────────

class FilterRequest(BaseModel):
    user_id: str
    category: str
    age: int
    session_time: float
    max_results: int = 8


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/filter-coupons", response_model=List[CouponInfo])
async def filter_coupons(request: FilterRequest, db: Session = Depends(get_db)):
    """
    Filter the coupon catalog to coupons relevant for this user context.
    Returns category-matched + universal coupons.
    """
    cache_key = f"filtered:{request.category}:{request.age}"
    cached = cache_get(cache_key)
    if cached:
        logger.info("Cache hit for coupon filter", extra={"extra_fields": {"cache_key": cache_key}})
        return [CouponInfo(**c) for c in cached]

    # Get category-specific + universal coupons
    coupons = db.query(Coupon).filter(
        Coupon.is_active == True,
        Coupon.category.in_([request.category.lower(), "all"])
    ).limit(request.max_results).all()

    if not coupons:
        # Fallback: return universal coupons only
        coupons = db.query(Coupon).filter(
            Coupon.is_active == True,
            Coupon.category == "all"
        ).all()

    result = [
        CouponInfo(
            coupon_code=c.coupon_code,
            discount_percent=c.discount_percent,
            category=c.category,
            min_purchase_amount=c.min_purchase_amount,
            description=c.description or "",
            is_active=c.is_active
        )
        for c in coupons
    ]

    cache_set(cache_key, [r.dict() for r in result], ttl=300)

    logger.info(
        "Coupons filtered",
        extra=LogContext(
            user_id=request.user_id,
            category=request.category,
            n_results=len(result)
        ).as_extra()
    )
    return result


@app.get("/coupons", response_model=List[CouponInfo])
async def list_coupons(
    category: Optional[str] = Query(None),
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """List all coupons, optionally filtered by category."""
    query = db.query(Coupon)
    if active_only:
        query = query.filter(Coupon.is_active == True)
    if category:
        query = query.filter(Coupon.category == category.lower())
    coupons = query.all()
    return [
        CouponInfo(
            coupon_code=c.coupon_code,
            discount_percent=c.discount_percent,
            category=c.category,
            min_purchase_amount=c.min_purchase_amount,
            description=c.description or "",
            is_active=c.is_active
        )
        for c in coupons
    ]


@app.get("/coupon/{coupon_code}", response_model=CouponInfo)
async def get_coupon(coupon_code: str, db: Session = Depends(get_db)):
    """Get details of a specific coupon by code."""
    coupon = db.query(Coupon).filter(Coupon.coupon_code == coupon_code).first()
    if not coupon:
        raise HTTPException(status_code=404, detail=f"Coupon {coupon_code} not found")
    return CouponInfo(
        coupon_code=coupon.coupon_code,
        discount_percent=coupon.discount_percent,
        category=coupon.category,
        min_purchase_amount=coupon.min_purchase_amount,
        description=coupon.description or "",
        is_active=coupon.is_active
    )


@app.get("/categories")
async def list_categories(db: Session = Depends(get_db)):
    """List all available shopping categories."""
    from sqlalchemy import distinct
    categories = db.query(distinct(Coupon.category)).all()
    return {"categories": [c[0] for c in categories]}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", service="recommender-service")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
