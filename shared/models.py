"""
Shared SQLAlchemy models used across all services.
Defines the database schema for users, coupons, interactions, and agent state.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class UserProfile(Base):
    """Stores synthetic user profile data."""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=False)
    preferred_category = Column(String(50), nullable=False)
    total_sessions = Column(Integer, default=0)
    total_clicks = Column(Integer, default=0)
    total_purchases = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Coupon(Base):
    """Available coupons in the system."""
    __tablename__ = "coupons"

    id = Column(Integer, primary_key=True, index=True)
    coupon_code = Column(String(50), unique=True, index=True, nullable=False)
    discount_percent = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)
    min_purchase_amount = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())


class Interaction(Base):
    """Logs every user-coupon interaction for training the agent."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True, nullable=False)
    coupon_code = Column(String(50), index=True, nullable=False)
    age = Column(Integer, nullable=False)
    category = Column(String(50), nullable=False)
    session_time = Column(Float, nullable=False)  # seconds
    coupon_shown = Column(Boolean, default=True)
    clicked = Column(Integer, default=0)     # 0 or 1
    purchased = Column(Integer, default=0)   # 0 or 1
    reward = Column(Float, default=0.0)      # click=+1, purchase=+5, ignore=0
    epsilon_used = Column(Float, nullable=True)  # epsilon at decision time
    was_exploration = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())


class AgentState(Base):
    """
    Persists the Multi-Armed Bandit agent's learned Q-values.
    Each row represents the agent's knowledge about a (context_bucket, coupon) pair.
    """
    __tablename__ = "agent_states"

    id = Column(Integer, primary_key=True, index=True)
    context_key = Column(String(200), index=True, nullable=False)  # e.g. "age_30_45|category_electronics"
    coupon_code = Column(String(50), index=True, nullable=False)
    q_value = Column(Float, default=0.0)        # estimated reward
    n_selections = Column(Integer, default=0)    # how often chosen
    n_rewards = Column(Integer, default=0)       # how often rewarded
    total_reward = Column(Float, default=0.0)    # cumulative reward
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class MetricsSnapshot(Base):
    """Periodic snapshots of system metrics for tracking over time."""
    __tablename__ = "metrics_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_time = Column(DateTime, default=func.now())
    total_impressions = Column(Integer, default=0)
    total_clicks = Column(Integer, default=0)
    total_purchases = Column(Integer, default=0)
    ctr = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)
    avg_reward = Column(Float, default=0.0)
    exploration_count = Column(Integer, default=0)
    exploitation_count = Column(Integer, default=0)
    epsilon = Column(Float, default=0.0)
