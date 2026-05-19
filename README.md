#  Agentic E-Commerce Coupon Recommender System

A production-style, self-learning coupon recommendation engine powered by a **Contextual Multi-Armed Bandit (Epsilon-Greedy)** AI agent. The system continuously improves its recommendations based on real-time user feedback.

---

##  Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           docker-compose network         │
                         │                                          │
  Client ──────────────► │  API Gateway :8000                       │
                         │      │                                   │
                         │      ├──► Recommender :8002              │
                         │      │       (filter coupons by context) │
                         │      │                                   │
                         │      ├──► Agent :8001                    │
                         │      │       (epsilon-greedy decision)   │
                         │      │                                   │
                         │      └──► Feedback :8003                 │
                         │              (record + reward agent)     │
                         │                                          │
                         │  Generator :8004 ──► Feedback (batch)    │
                         │                                          │
                         │  PostgreSQL :5432   Redis :6379          │
                         │                                          │
                         │  Dashboard :3000 (nginx + HTML)          │
                         └─────────────────────────────────────────┘
```

### Services

| Service | Port | Responsibility |
|---------|------|----------------|
| **API Gateway** | 8000 | Public-facing entry point, request routing |
| **Agent Service** | 8001 | Epsilon-Greedy bandit, Q-table management |
| **Recommender** | 8002 | Coupon catalog, context-based filtering |
| **Feedback** | 8003 | Reward processing, metrics aggregation |
| **Generator** | 8004 | Synthetic data generation |
| **Dashboard** | 3000 | Live monitoring UI |
| **PostgreSQL** | 5432 | Persistent storage |
| **Redis** | 6379 | Session cache, fast counters |

---

## 🚀 Quick Start

### Prerequisites
- Docker 24+ and Docker Compose v2
- 4GB RAM available for containers

### 1. Clone and Start

```bash
git clone <repo>
cd coupon-recommender

# Full setup: build → start → seed database
make setup

# Or manually:
docker-compose up --build -d
```

### 2. Wait for Services (~30 seconds)

```bash
make ps          # check all services are "healthy"
make logs        # tail logs from all services
```

### 3. Seed Historical Data

The agent learns better with prior data. Seed 200 interactions:

```bash
make seed
# Or: curl -X POST "http://localhost:8000/seed-database?n_interactions=200"
```

### 4. Access the System

| URL | Description |
|-----|-------------|
| http://localhost:3000 | 📊 Live Dashboard |
| http://localhost:8000/docs | 📖 Swagger API Docs |
| http://localhost:8000/redoc | 📖 ReDoc API Docs |

---

##  API Reference

### `POST /get-coupon`
Get a personalized coupon recommendation for a user.

```bash
curl -X POST http://localhost:8000/get-coupon \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "age": 28,
    "category": "electronics",
    "session_time": 245.5
  }'
```

Response:
```json
{
  "user_id": "user_001",
  "coupon_code": "TECH20",
  "discount_percent": 20.0,
  "category": "electronics",
  "description": "20% off electronics over $100",
  "was_exploration": false,
  "agent_confidence": 0.76,
  "message": "Coupon recommendation generated successfully"
}
```

### `POST /feedback`
Submit user interaction feedback to train the agent.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "coupon_code": "TECH20",
    "clicked": 1,
    "purchased": 1,
    "age": 28,
    "category": "electronics",
    "session_time": 245.5
  }'
```

**Reward mapping:**
- `clicked=0, purchased=0` → reward = **0** (ignored)
- `clicked=1, purchased=0` → reward = **+1** (engaged)
- `clicked=1, purchased=1` → reward = **+5** (converted)

### `GET /metrics`
Get real-time performance metrics.

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "total_impressions": 1523,
  "total_clicks": 287,
  "total_purchases": 64,
  "ctr": 0.1884,
  "conversion_rate": 0.042,
  "avg_reward": 0.26,
  "exploration_count": 412,
  "exploitation_count": 1111,
  "exploration_ratio": 0.27,
  "current_epsilon": 0.183,
  "top_coupons": [...]
}
```

---

## 🧠 How the Agent Works

### Epsilon-Greedy Contextual Bandit

The agent maintains a **Q-table**: estimated expected reward for each `(context, coupon)` pair.

**Context** is formed by discretizing user features:
```
context_key = age_bucket (6 levels) × category (6 types) = ~36 states
```

**Decision at each step:**
```
with probability ε:   EXPLORE → pick random coupon
with probability 1-ε: EXPLOIT → pick coupon with highest Q-value
```

**Q-value update (online learning):**
```
Q(s,a) ← Q(s,a) + α × (reward − Q(s,a))
```
Where:
- `α = 0.1` (learning rate)
- `reward ∈ {0, 1, 5}` depending on user action

**Epsilon decay:**
```
ε = max(ε_min, ε_initial × decay^n_decisions)
ε_initial = 0.30,  ε_min = 0.05,  decay = 0.9995
```

As the agent sees more interactions, it explores less and exploits its learned knowledge more.

---

##  Database Schema

```sql
user_profiles     -- synthetic user demographics
coupons           -- coupon catalog (16 coupons seeded)
interactions      -- every user-coupon event (for audit + retraining)
agent_states      -- Q-table: (context_key, coupon_code) → q_value
metrics_snapshots -- periodic metric snapshots for trend analysis
```

---

##  Testing

### Integration Tests (requires running system)
```bash
make test
# or: python tests/test_integration.py
```

### Unit Tests (no Docker needed)
```bash
make test-unit
# or: pytest tests/test_bandit.py -v
```

### Load Tests (requires Locust)
```bash
pip install locust
make test-load
# Open http://localhost:8089 for Locust UI
```

---

##  Configuration

All agent hyperparameters are configurable via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EPSILON_INITIAL` | `0.3` | Starting exploration rate |
| `EPSILON_MIN` | `0.05` | Minimum exploration rate |
| `EPSILON_DECAY` | `0.9995` | Epsilon decay per decision |
| `LEARNING_RATE` | `0.1` | Q-value update step size (α) |
| `REWARD_CLICK` | `1.0` | Reward for user click |
| `REWARD_PURCHASE` | `5.0` | Reward for purchase |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

##  Monitoring

### Live Dashboard
Open http://localhost:3000 for real-time visualization of:
- CTR, Conversion Rate, Average Reward
- Epsilon (exploration rate) gauge
- Explore vs Exploit split
- Top coupons by reward (bar chart + table)
- Live API demo panel

### JSON Logs
All services emit structured JSON logs:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "service": "agent-service",
  "level": "INFO",
  "message": "Agent EXPLOITING",
  "context_key": "age_26_35|cat_electronics",
  "chosen_coupon": "TECH20",
  "q_value": 3.82,
  "epsilon": 0.183
}
```

Pipe to jq for easy reading:
```bash
docker-compose logs -f agent | grep -v "^$" | jq .
```

---

##  Development

### Hot-Reload Mode
```bash
make up-dev
# Services will auto-reload on code changes
```

### Database Shell
```bash
make shell-db    # psql
make shell-redis # redis-cli

# Example queries:
# SELECT context_key, coupon_code, q_value, n_selections FROM agent_states ORDER BY q_value DESC LIMIT 10;
# SELECT coupon_code, COUNT(*), AVG(reward) FROM interactions GROUP BY coupon_code;
```

### Generate More Data
```bash
# 100 more synthetic users:
curl -X POST "http://localhost:8000/generate-data?n_users=100"
```

---

##  Project Structure

```
coupon-recommender/
├── docker-compose.yml         # Service orchestration
├── docker-compose.dev.yml     # Dev overrides (hot-reload)
├── Makefile                   # Convenience commands
├── README.md
│
├── shared/                    # Code shared across all services
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy ORM models
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── database.py            # PostgreSQL connection pool
│   ├── cache.py               # Redis utilities
│   └── logger.py              # Structured JSON logging
│
├── services/
│   ├── api/
│   │   ├── main.py            # API Gateway (port 8000)
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── agent/
│   │   ├── bandit.py          # Epsilon-Greedy core algorithm
│   │   ├── main.py            # Agent Service (port 8001)
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── recommender/
│   │   ├── main.py            # Recommender Service (port 8002)
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── feedback/
│   │   ├── main.py            # Feedback Service (port 8003)
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── generator/
│       ├── main.py            # Generator Service (port 8004)
│       ├── requirements.txt
│       └── Dockerfile
│
├── dashboard/
│   ├── index.html             # Single-page monitoring dashboard
│   ├── nginx.conf             # Nginx reverse proxy
│   └── Dockerfile
│
├── tests/
│   ├── test_integration.py    # End-to-end API tests
│   ├── test_bandit.py         # Unit tests for agent logic
│   └── locustfile.py          # Load testing script
│
└── postgres/
    └── init.sql               # DB initialization script
```

set AMI_ID=ami-0446b021dec428a7b

set INSTANCE_ID=i-0f1e52fb98116de74


set EC2_IP=100.53.169.46

ssh -i %USERPROFILE%\.ssh\streampulse-key-n.pem -o StrictHostKeyChecking=no ec2-user@%EC2_IP%


