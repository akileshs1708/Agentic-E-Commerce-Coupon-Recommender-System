"""
Contextual Multi-Armed Bandit Agent using Epsilon-Greedy strategy.

The agent maintains Q-values (estimated expected reward) for each
(context, coupon) pair and uses epsilon-greedy exploration:
  - With probability epsilon: explore (pick random coupon)
  - With probability 1-epsilon: exploit (pick best known coupon)

Epsilon decays over time as the agent learns more.

Reward structure:
  - User ignores coupon: reward = 0
  - User clicks coupon:  reward = +1
  - User purchases:      reward = +5 (click + purchase bonus)
"""
import os
import math
import random
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from shared.models import AgentState, Interaction
from shared.logger import get_logger, LogContext
from shared.cache import cache_get, cache_set

logger = get_logger(__name__, "agent")

# ─── Agent Hyperparameters ────────────────────────────────────────────────────
EPSILON_INITIAL = float(os.getenv("EPSILON_INITIAL", "0.3"))    # start: 30% exploration
EPSILON_MIN = float(os.getenv("EPSILON_MIN", "0.05"))           # floor: 5% always explore
EPSILON_DECAY = float(os.getenv("EPSILON_DECAY", "0.9995"))     # multiplicative decay per decision
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.1"))        # alpha for Q-value update
DISCOUNT_FACTOR = float(os.getenv("DISCOUNT_FACTOR", "0.9"))    # gamma (for future rewards)

# Reward values — can be tuned via env vars
REWARD_CLICK = float(os.getenv("REWARD_CLICK", "1.0"))
REWARD_PURCHASE = float(os.getenv("REWARD_PURCHASE", "5.0"))
REWARD_IGNORE = float(os.getenv("REWARD_IGNORE", "0.0"))


def get_context_key(age: int, category: str) -> str:
    """
    Discretize continuous age into buckets and combine with category
    to form a context key for the Q-table.

    Age buckets: 18-25, 26-35, 36-45, 46-55, 56-65, 65+
    This reduces state space while preserving meaningful variation.
    """
    if age <= 25:
        age_bucket = "18_25"
    elif age <= 35:
        age_bucket = "26_35"
    elif age <= 45:
        age_bucket = "36_45"
    elif age <= 55:
        age_bucket = "46_55"
    elif age <= 65:
        age_bucket = "56_65"
    else:
        age_bucket = "65_plus"

    return f"age_{age_bucket}|cat_{category.lower()}"


def compute_reward(clicked: int, purchased: int) -> float:
    """
    Convert user behavior to a scalar reward signal.
    Purchase gives the highest reward to align agent incentives
    with business goals (revenue over clicks).
    """
    if purchased == 1:
        return REWARD_PURCHASE   # +5: highest value action
    elif clicked == 1:
        return REWARD_CLICK      # +1: engagement signal
    else:
        return REWARD_IGNORE     # 0: no engagement


def get_current_epsilon(db: Session) -> float:
    """
    Compute current epsilon based on total number of decisions made.
    Uses exponential decay: epsilon = max(epsilon_min, epsilon_initial * decay^n)
    """
    # Count total interactions from DB (or use cached counter)
    cached = cache_get("agent:total_decisions")
    if cached is not None:
        n_decisions = int(cached)
    else:
        n_decisions = db.query(Interaction).count()
        cache_set("agent:total_decisions", n_decisions, ttl=60)

    epsilon = max(EPSILON_MIN, EPSILON_INITIAL * (EPSILON_DECAY ** n_decisions))
    return round(epsilon, 6)


def get_q_value(db: Session, context_key: str, coupon_code: str) -> float:
    """
    Look up Q-value for a (context, coupon) pair.
    Returns 0.0 if never seen (optimistic initialization).
    Checks Redis cache first for speed.
    """
    cache_key = f"qval:{context_key}:{coupon_code}"
    cached = cache_get(cache_key)
    if cached is not None:
        return float(cached)

    state = (
        db.query(AgentState)
        .filter(
            AgentState.context_key == context_key,
            AgentState.coupon_code == coupon_code
        )
        .first()
    )

    q = state.q_value if state else 0.0
    cache_set(cache_key, q, ttl=30)  # short TTL so updates propagate fast
    return q


def select_coupon(
    db: Session,
    context_key: str,
    available_coupons: List[str],
    epsilon: float
) -> Tuple[str, float, bool]:
    """
    Epsilon-greedy action selection.

    Args:
        db: Database session
        context_key: Discretized context string
        available_coupons: List of coupon codes to choose from
        epsilon: Current exploration probability

    Returns:
        (selected_coupon_code, q_value, was_exploration)
    """
    if not available_coupons:
        raise ValueError("No coupons available for selection")

    # ── Exploration: random choice ──────────────────────────────────────────
    if random.random() < epsilon:
        chosen = random.choice(available_coupons)
        q_val = get_q_value(db, context_key, chosen)
        logger.info(
            "Agent EXPLORING",
            extra=LogContext(
                context_key=context_key,
                chosen_coupon=chosen,
                epsilon=epsilon,
                action="explore"
            ).as_extra()
        )
        return chosen, q_val, True

    # ── Exploitation: pick coupon with highest Q-value ───────────────────────
    q_values = {
        coupon: get_q_value(db, context_key, coupon)
        for coupon in available_coupons
    }

    # Break ties randomly (important for fair initial exploration)
    max_q = max(q_values.values())
    best_coupons = [c for c, q in q_values.items() if q == max_q]
    chosen = random.choice(best_coupons)

    logger.info(
        "Agent EXPLOITING",
        extra=LogContext(
            context_key=context_key,
            chosen_coupon=chosen,
            q_value=max_q,
            epsilon=epsilon,
            action="exploit",
            q_table_snapshot={k: round(v, 3) for k, v in q_values.items()}
        ).as_extra()
    )
    return chosen, max_q, False


def update_q_value(
    db: Session,
    context_key: str,
    coupon_code: str,
    reward: float
) -> float:
    """
    Update Q-value using incremental mean (online learning).

    Formula (incremental mean):
        Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))

    This is equivalent to a weighted moving average, giving more
    weight to recent observations via the learning_rate alpha.

    Args:
        db: Database session
        context_key: State context
        coupon_code: Action taken
        reward: Observed reward

    Returns:
        Updated Q-value
    """
    state = (
        db.query(AgentState)
        .filter(
            AgentState.context_key == context_key,
            AgentState.coupon_code == coupon_code
        )
        .first()
    )

    if state is None:
        # First time seeing this (context, coupon) pair — initialize
        state = AgentState(
            context_key=context_key,
            coupon_code=coupon_code,
            q_value=reward,      # Initialize with first reward
            n_selections=1,
            n_rewards=1 if reward > 0 else 0,
            total_reward=reward
        )
        db.add(state)
    else:
        # Incremental Q-value update: Q += alpha * (reward - Q)
        old_q = state.q_value
        state.q_value = old_q + LEARNING_RATE * (reward - old_q)
        state.n_selections += 1
        state.total_reward += reward
        if reward > 0:
            state.n_rewards += 1

    db.commit()
    db.refresh(state)

    # Invalidate cache so next lookup gets fresh value
    cache_key = f"qval:{context_key}:{coupon_code}"
    cache_set(cache_key, state.q_value, ttl=30)

    logger.info(
        "Q-value updated",
        extra=LogContext(
            context_key=context_key,
            coupon_code=coupon_code,
            reward=reward,
            new_q_value=round(state.q_value, 4),
            n_selections=state.n_selections
        ).as_extra()
    )

    return state.q_value


def get_agent_stats(db: Session) -> Dict:
    """
    Retrieve summary statistics about the agent's current state.
    Used by the metrics endpoint.
    """
    states = db.query(AgentState).all()
    if not states:
        return {"total_states": 0, "avg_q_value": 0.0, "top_pairs": []}

    avg_q = sum(s.q_value for s in states) / len(states)
    top_pairs = sorted(states, key=lambda s: s.q_value, reverse=True)[:10]

    return {
        "total_states": len(states),
        "avg_q_value": round(avg_q, 4),
        "top_pairs": [
            {
                "context_key": s.context_key,
                "coupon_code": s.coupon_code,
                "q_value": round(s.q_value, 4),
                "n_selections": s.n_selections,
                "total_reward": round(s.total_reward, 2)
            }
            for s in top_pairs
        ]
    }
