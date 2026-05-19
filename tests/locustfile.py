"""
Load testing for the Coupon Recommender API using Locust.

Usage:
  pip install locust
  locust -f tests/locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 for the Locust web UI.
"""
import random
from locust import HttpUser, task, between, events

CATEGORIES = ["electronics", "fashion", "groceries", "home", "books", "sports"]
USER_COUNT = 0


class CouponUser(HttpUser):
    """
    Simulates a real user flow:
    1. Request a coupon recommendation
    2. With probability p_click: click the coupon
    3. With probability p_purchase (given click): purchase
    4. Submit feedback
    """
    wait_time = between(0.5, 2.0)  # think time between requests

    def on_start(self):
        global USER_COUNT
        USER_COUNT += 1
        self.user_id = f"load_test_user_{USER_COUNT:06d}"
        self.age = random.randint(18, 70)
        self.category = random.choice(CATEGORIES)
        self.last_coupon = None

    @task(3)
    def get_and_feedback(self):
        """Main task: get coupon + immediately submit feedback."""
        # ── Get coupon ────────────────────────────────────────────────────────
        with self.client.post(
            "/get-coupon",
            json={
                "user_id": self.user_id,
                "age": self.age,
                "category": self.category,
                "session_time": random.uniform(30, 600)
            },
            catch_response=True,
            name="POST /get-coupon"
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            data = response.json()
            coupon_code = data.get("coupon_code")
            if not coupon_code:
                response.failure("Missing coupon_code in response")
                return
            response.success()

        # ── Simulate user behavior ────────────────────────────────────────────
        p_click = 0.25 + (data.get("discount_percent", 10) * 0.008)
        clicked = 1 if random.random() < p_click else 0
        purchased = 1 if (clicked and random.random() < 0.30) else 0

        # ── Submit feedback ───────────────────────────────────────────────────
        with self.client.post(
            "/feedback",
            json={
                "user_id": self.user_id,
                "coupon_code": coupon_code,
                "clicked": clicked,
                "purchased": purchased,
                "age": self.age,
                "category": self.category,
                "session_time": random.uniform(30, 600)
            },
            catch_response=True,
            name="POST /feedback"
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                response.success()

    @task(1)
    def check_metrics(self):
        """Periodically check metrics (background monitoring)."""
        self.client.get("/metrics", name="GET /metrics")

    @task(1)
    def check_health(self):
        self.client.get("/health", name="GET /health")

    def on_stop(self):
        """Occasionally change category to simulate browsing."""
        self.category = random.choice(CATEGORIES)
        self.age = random.randint(18, 70)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n🚀 Load test starting — Coupon Recommender System")
    print(f"   Target: {environment.host}")
    print("   Simulating real user behavior: get-coupon → feedback loop\n")
