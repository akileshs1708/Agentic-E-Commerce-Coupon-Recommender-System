"""
Unit tests for the bandit agent core logic.
These tests run WITHOUT Docker — they mock the database.

Usage:
  pip install pytest
  pytest tests/test_bandit.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import random
from unittest.mock import MagicMock, patch

# Patch the cache to use a simple dict in tests
import shared.cache as cache_module
_fake_cache = {}

def fake_get(key):
    return _fake_cache.get(key)

def fake_set(key, value, ttl=300):
    _fake_cache[key] = value
    return True

cache_module.cache_get = fake_get
cache_module.cache_set = fake_set

from services.agent.bandit import (
    get_context_key,
    compute_reward,
    select_coupon,
    update_q_value,
    get_q_value,
    REWARD_CLICK,
    REWARD_PURCHASE,
    REWARD_IGNORE,
)


# ─── Context Key Tests ────────────────────────────────────────────────────────

class TestContextKey:
    def test_young_adult(self):
        assert get_context_key(22, "electronics") == "age_18_25|cat_electronics"

    def test_middle_age(self):
        assert get_context_key(40, "fashion") == "age_36_45|cat_fashion"

    def test_senior(self):
        assert get_context_key(70, "home") == "age_65_plus|cat_home"

    def test_age_boundary_26(self):
        assert get_context_key(26, "books") == "age_26_35|cat_books"

    def test_age_boundary_25(self):
        assert get_context_key(25, "books") == "age_18_25|cat_books"

    def test_category_lowercased(self):
        key1 = get_context_key(30, "Electronics")
        key2 = get_context_key(30, "electronics")
        assert key1 == key2


# ─── Reward Tests ─────────────────────────────────────────────────────────────

class TestRewards:
    def test_purchase_reward(self):
        assert compute_reward(1, 1) == REWARD_PURCHASE  # +5

    def test_click_reward(self):
        assert compute_reward(1, 0) == REWARD_CLICK     # +1

    def test_ignore_reward(self):
        assert compute_reward(0, 0) == REWARD_IGNORE    # 0

    def test_purchase_is_highest(self):
        assert compute_reward(1, 1) > compute_reward(1, 0) > compute_reward(0, 0)


# ─── Agent Decision Tests ─────────────────────────────────────────────────────

class TestAgentDecision:
    def setup_method(self):
        """Fresh DB mock for each test."""
        _fake_cache.clear()
        self.db = MagicMock()
        # Default: no existing AgentState rows
        self.db.query.return_value.filter.return_value.first.return_value = None

    def test_selects_from_candidates(self):
        coupons = ["TECH10", "TECH20", "GADGET15"]
        chosen, q, was_explore = select_coupon(
            db=self.db,
            context_key="age_26_35|cat_electronics",
            available_coupons=coupons,
            epsilon=0.0   # pure exploitation
        )
        assert chosen in coupons

    def test_exploration_uses_random(self):
        """With epsilon=1.0, every choice is random but still valid."""
        coupons = ["TECH10", "TECH20", "GADGET15"]
        chosen, q, was_explore = select_coupon(
            db=self.db,
            context_key="age_26_35|cat_electronics",
            available_coupons=coupons,
            epsilon=1.0
        )
        assert chosen in coupons
        assert was_explore is True

    def test_exploitation_picks_highest_q(self):
        """With epsilon=0.0, agent should pick the coupon with highest Q-value."""
        # Inject Q-values via cache
        _fake_cache["qval:age_26_35|cat_electronics:TECH10"] = 0.5
        _fake_cache["qval:age_26_35|cat_electronics:TECH20"] = 3.8  # highest
        _fake_cache["qval:age_26_35|cat_electronics:GADGET15"] = 1.2

        coupons = ["TECH10", "TECH20", "GADGET15"]
        chosen, q, was_explore = select_coupon(
            db=self.db,
            context_key="age_26_35|cat_electronics",
            available_coupons=coupons,
            epsilon=0.0   # no exploration
        )
        assert chosen == "TECH20", f"Expected TECH20 (q=3.8), got {chosen}"
        assert was_explore is False

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError, match="No coupons available"):
            select_coupon(self.db, "some_key", [], epsilon=0.3)


# ─── Q-Value Update Tests ─────────────────────────────────────────────────────

class TestQValueUpdate:
    def setup_method(self):
        _fake_cache.clear()
        self.db = MagicMock()

    def test_first_update_initializes(self):
        """First reward for unseen (context, coupon) initializes Q = reward."""
        self.db.query.return_value.filter.return_value.first.return_value = None

        mock_state = MagicMock()
        mock_state.q_value = 5.0
        mock_state.n_selections = 1
        mock_state.total_reward = 5.0

        # Simulate: after db.add + commit, refresh sets values
        def fake_refresh(obj):
            obj.q_value = 5.0
            obj.n_selections = 1
            obj.total_reward = 5.0

        self.db.refresh.side_effect = fake_refresh

        with patch("services.agent.bandit.AgentState") as MockState:
            instance = MagicMock()
            instance.q_value = 5.0
            MockState.return_value = instance

            new_q = update_q_value(self.db, "age_26_35|cat_electronics", "TECH20", 5.0)
            # Q-value should be initialized to the first reward
            assert new_q == 5.0

    def test_incremental_update_formula(self):
        """Q(s,a) += alpha * (reward - Q(s,a)) with alpha=0.1"""
        existing_state = MagicMock()
        existing_state.q_value = 2.0
        existing_state.n_selections = 5
        existing_state.total_reward = 10.0
        existing_state.n_rewards = 3
        self.db.query.return_value.filter.return_value.first.return_value = existing_state

        def fake_refresh(obj):
            pass  # already updated in-place

        self.db.refresh.side_effect = fake_refresh

        # alpha=0.1, old_q=2.0, reward=5.0
        # expected: 2.0 + 0.1 * (5.0 - 2.0) = 2.0 + 0.3 = 2.3
        new_q = update_q_value(self.db, "age_26_35|cat_electronics", "TECH20", 5.0)
        assert abs(new_q - 2.3) < 0.001, f"Expected 2.3, got {new_q}"

    def test_q_value_increases_with_positive_reward(self):
        """Positive reward should push Q-value upward."""
        existing_state = MagicMock()
        existing_state.q_value = 1.0
        existing_state.n_selections = 3
        existing_state.total_reward = 3.0
        existing_state.n_rewards = 3
        self.db.query.return_value.filter.return_value.first.return_value = existing_state
        self.db.refresh.side_effect = lambda obj: None

        old_q = existing_state.q_value
        new_q = update_q_value(self.db, "test_ctx", "SAVE5", 5.0)
        assert new_q > old_q, "Positive reward should increase Q-value"

    def test_q_value_decreases_with_zero_reward(self):
        """Zero reward on high Q should decrease it toward 0."""
        existing_state = MagicMock()
        existing_state.q_value = 3.0
        existing_state.n_selections = 10
        existing_state.total_reward = 30.0
        existing_state.n_rewards = 10
        self.db.query.return_value.filter.return_value.first.return_value = existing_state
        self.db.refresh.side_effect = lambda obj: None

        new_q = update_q_value(self.db, "test_ctx", "TECH10", 0.0)
        assert new_q < 3.0, "Zero reward should decrease Q-value"


# ─── Simulation Sanity Test ───────────────────────────────────────────────────

class TestSimulation:
    """
    Sanity-check the agent's learning trajectory over many steps.
    Simulates a simple bandit problem where coupon A always gives reward 5
    and coupon B always gives reward 0. Agent should learn to prefer A.
    """
    def test_agent_converges_to_best_arm(self):
        _fake_cache.clear()
        db = MagicMock()

        # Simulate Q-values evolving over 100 steps
        q_values = {"GOOD": 0.0, "BAD": 0.0}
        alpha = 0.1
        epsilon = 0.3
        good_selections = 0
        total = 100

        for step in range(total):
            # Decay epsilon
            epsilon = max(0.05, epsilon * 0.99)

            # Inject current Q-values into cache
            _fake_cache["qval:ctx:GOOD"] = q_values["GOOD"]
            _fake_cache["qval:ctx:BAD"] = q_values["BAD"]

            # Mock DB (Q-values via cache)
            db.query.return_value.filter.return_value.first.return_value = None

            chosen, q, was_explore = select_coupon(db, "ctx", ["GOOD", "BAD"], epsilon)

            # Environment: GOOD always rewards 5, BAD always 0
            reward = 5.0 if chosen == "GOOD" else 0.0

            # Update Q manually (simulating update_q_value)
            old_q = q_values[chosen]
            q_values[chosen] = old_q + alpha * (reward - old_q)

            if chosen == "GOOD":
                good_selections += 1

        good_rate = good_selections / total
        # After 100 steps with decay epsilon, agent should pick GOOD > 60% of the time
        assert good_rate > 0.6, (
            f"Agent failed to converge: only picked the good arm {good_rate:.0%} "
            f"of the time (expected > 60%)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
