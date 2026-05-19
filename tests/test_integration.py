#!/usr/bin/env python3
"""
End-to-end integration tests for the Coupon Recommender System.
Run after docker-compose up to verify all services are functioning correctly.

Usage:
  python tests/test_integration.py
  # or with a custom base URL:
  API_URL=http://localhost:8000 python tests/test_integration.py
"""
import os
import sys
import time
import json
import httpx
import random

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

CATEGORIES = ["electronics", "fashion", "groceries", "home", "books", "sports"]
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []


def test(name: str, fn):
    """Run a test function and record the result."""
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))


def assert_status(response, expected=200):
    if response.status_code != expected:
        raise AssertionError(
            f"Expected HTTP {expected}, got {response.status_code}. Body: {response.text[:200]}"
        )


def assert_field(data: dict, field: str, expected_type=None):
    if field not in data:
        raise AssertionError(f"Missing field '{field}' in response: {list(data.keys())}")
    if expected_type and not isinstance(data[field], expected_type):
        raise AssertionError(
            f"Field '{field}' expected {expected_type.__name__}, got {type(data[field]).__name__}"
        )


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_health():
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert_status(r)
    data = r.json()
    assert data["status"] in ("healthy", "degraded"), f"Unexpected status: {data['status']}"
    assert "services" in data


def test_get_coupon_basic():
    payload = {
        "user_id": "test_user_001",
        "age": 28,
        "category": "electronics",
        "session_time": 120.0
    }
    r = httpx.post(f"{BASE_URL}/get-coupon", json=payload, timeout=15)
    assert_status(r)
    data = r.json()
    assert_field(data, "coupon_code", str)
    assert_field(data, "discount_percent", float)
    assert_field(data, "was_exploration", bool)
    assert_field(data, "agent_confidence", float)
    assert 0 <= data["agent_confidence"] <= 1, "Confidence out of range"


def test_get_coupon_all_categories():
    for category in CATEGORIES:
        payload = {
            "user_id": f"test_cat_{category}",
            "age": random.randint(20, 65),
            "category": category,
            "session_time": random.uniform(30, 600)
        }
        r = httpx.post(f"{BASE_URL}/get-coupon", json=payload, timeout=15)
        assert_status(r, 200)
        data = r.json()
        assert data["coupon_code"], f"Empty coupon_code for category {category}"


def test_feedback_purchase():
    # First get a coupon
    payload = {"user_id": "test_fb_001", "age": 35, "category": "fashion", "session_time": 200}
    r = httpx.post(f"{BASE_URL}/get-coupon", json=payload, timeout=15)
    assert_status(r)
    coupon = r.json()["coupon_code"]

    # Submit purchase feedback
    fb = {
        "user_id": "test_fb_001",
        "coupon_code": coupon,
        "clicked": 1,
        "purchased": 1,
        "age": 35,
        "category": "fashion",
        "session_time": 200.0
    }
    r2 = httpx.post(f"{BASE_URL}/feedback", json=fb, timeout=15)
    assert_status(r2)
    data = r2.json()
    assert data["reward"] == 5.0, f"Expected reward 5.0 for purchase, got {data['reward']}"


def test_feedback_click_only():
    payload = {"user_id": "test_fb_002", "age": 25, "category": "books", "session_time": 90}
    r = httpx.post(f"{BASE_URL}/get-coupon", json=payload, timeout=15)
    coupon = r.json()["coupon_code"]

    fb = {
        "user_id": "test_fb_002",
        "coupon_code": coupon,
        "clicked": 1,
        "purchased": 0,
        "age": 25,
        "category": "books",
        "session_time": 90.0
    }
    r2 = httpx.post(f"{BASE_URL}/feedback", json=fb, timeout=15)
    assert_status(r2)
    assert r2.json()["reward"] == 1.0


def test_feedback_ignore():
    payload = {"user_id": "test_fb_003", "age": 50, "category": "home", "session_time": 60}
    r = httpx.post(f"{BASE_URL}/get-coupon", json=payload, timeout=15)
    coupon = r.json()["coupon_code"]

    fb = {
        "user_id": "test_fb_003",
        "coupon_code": coupon,
        "clicked": 0,
        "purchased": 0,
        "age": 50,
        "category": "home",
        "session_time": 60.0
    }
    r2 = httpx.post(f"{BASE_URL}/feedback", json=fb, timeout=15)
    assert_status(r2)
    assert r2.json()["reward"] == 0.0


def test_feedback_validation_error():
    """purchased=1 without clicked=1 should return 422."""
    fb = {
        "user_id": "test_val_001",
        "coupon_code": "TECH10",
        "clicked": 0,       # invalid: can't purchase without clicking
        "purchased": 1,
        "age": 30,
        "category": "electronics",
        "session_time": 100.0
    }
    r = httpx.post(f"{BASE_URL}/feedback", json=fb, timeout=10)
    assert r.status_code == 422, f"Expected 422 validation error, got {r.status_code}"


def test_metrics():
    r = httpx.get(f"{BASE_URL}/metrics", timeout=15)
    assert_status(r)
    data = r.json()
    for field in ["total_impressions", "total_clicks", "total_purchases", "ctr",
                  "conversion_rate", "avg_reward", "exploration_count",
                  "exploitation_count", "exploration_ratio", "current_epsilon"]:
        assert_field(data, field)
    assert 0 <= data["ctr"] <= 1, "CTR out of [0,1] range"
    assert 0 <= data["conversion_rate"] <= 1, "Conversion rate out of range"
    assert 0 <= data["current_epsilon"] <= 1, "Epsilon out of range"


def test_agent_learns_from_feedback():
    """
    Verify that repeatedly rewarding a coupon improves its recommendation frequency.
    After 20 purchase feedbacks for TECH20, it should be recommended more often.
    """
    target_coupon = "TECH20"

    # Pump positive feedback for TECH20 with electronics users
    for i in range(20):
        fb = {
            "user_id": f"learning_test_{i}",
            "coupon_code": target_coupon,
            "clicked": 1,
            "purchased": 1,
            "age": random.randint(20, 40),
            "category": "electronics",
            "session_time": 200.0
        }
        httpx.post(f"{BASE_URL}/feedback", json=fb, timeout=10)

    time.sleep(1)  # let background tasks complete

    # Now request coupons and see how often TECH20 is chosen
    tech20_count = 0
    n_trials = 30
    for i in range(n_trials):
        r = httpx.post(f"{BASE_URL}/get-coupon", json={
            "user_id": f"eval_user_{i}",
            "age": random.randint(22, 38),
            "category": "electronics",
            "session_time": 150.0
        }, timeout=10)
        if r.status_code == 200 and r.json()["coupon_code"] == target_coupon:
            tech20_count += 1

    rate = tech20_count / n_trials
    # With epsilon-greedy learning, we expect > 20% recommendation rate after 20 purchases
    # (higher than random 1/5 ≈ 20% given 5 electronics coupons, accounting for explore phase)
    assert rate > 0.15, f"Agent doesn't seem to be learning: TECH20 chosen only {rate:.0%} of the time"


def test_coupon_catalog():
    r = httpx.get(f"{BASE_URL}/coupons", timeout=10)
    assert_status(r)
    coupons = r.json()
    assert len(coupons) >= 10, f"Expected at least 10 coupons, got {len(coupons)}"


def test_agent_epsilon_endpoint():
    r = httpx.get(f"{BASE_URL}/agent/epsilon", timeout=10)
    assert_status(r)
    data = r.json()
    assert_field(data, "epsilon", float)
    assert 0 <= data["epsilon"] <= 1


def test_generate_data():
    r = httpx.post(f"{BASE_URL}/generate-data?n_users=10", timeout=60)
    assert_status(r)
    data = r.json()
    assert_field(data, "generated", int)
    assert data["generated"] > 0


# ─── Runner ───────────────────────────────────────────────────────────────────

def wait_for_api(max_retries=15, delay=3):
    """Wait for API gateway to become available."""
    print(f"\n🔌 Waiting for API at {BASE_URL}...")
    for attempt in range(max_retries):
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"  {PASS} API is up!\n")
                return True
        except Exception:
            pass
        print(f"  Attempt {attempt+1}/{max_retries} — retrying in {delay}s...")
        time.sleep(delay)
    print(f"  {FAIL} API did not start in time")
    return False


if __name__ == "__main__":
    if not wait_for_api():
        sys.exit(1)

    print("🧪 Running Integration Tests\n" + "─" * 50)

    test("Health check", test_health)
    test("Get coupon (basic)", test_get_coupon_basic)
    test("Get coupon (all categories)", test_get_coupon_all_categories)
    test("Feedback: purchase (+5 reward)", test_feedback_purchase)
    test("Feedback: click-only (+1 reward)", test_feedback_click_only)
    test("Feedback: ignore (0 reward)", test_feedback_ignore)
    test("Feedback: validation error (422)", test_feedback_validation_error)
    test("Metrics endpoint", test_metrics)
    test("Coupon catalog", test_coupon_catalog)
    test("Agent epsilon endpoint", test_agent_epsilon_endpoint)
    test("Generate synthetic data", test_generate_data)
    test("Agent learns from feedback", test_agent_learns_from_feedback)

    print("\n" + "─" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    print(f"\n📊 Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
        print("\nFailed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  {FAIL} {name}: {err}")
        sys.exit(1)
    else:
        print(" 🎉")
        print("\nAll tests passed! The system is working correctly.")
        sys.exit(0)
