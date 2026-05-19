# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Agentic Coupon Recommender System
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: up down build logs clean test seed shell-api shell-db ps help

# Colors
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RESET  := \033[0m

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ─── Core Operations ─────────────────────────────────────────────────────────

build:  ## Build all Docker images
	@echo "$(YELLOW)Building all service images...$(RESET)"
	docker-compose build --parallel

up:  ## Start all services (detached)
	@echo "$(YELLOW)Starting Coupon Recommender System...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(RESET)"
	@echo "  API:       http://localhost:8000"
	@echo "  API Docs:  http://localhost:8000/docs"
	@echo "  Dashboard: http://localhost:3000"
	@echo ""
	@echo "Run 'make seed' to populate with sample data."

up-dev:  ## Start with hot-reload (development mode)
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

down:  ## Stop all services
	docker-compose down

down-v:  ## Stop all services and remove volumes (WIPES DATA)
	@echo "$(YELLOW)WARNING: This will delete all data!$(RESET)"
	docker-compose down -v

restart: down up  ## Restart all services

ps:  ## Show running service status
	docker-compose ps

logs:  ## Tail logs from all services
	docker-compose logs -f --tail=50

logs-api:  ## Tail API gateway logs
	docker-compose logs -f api

logs-agent:  ## Tail agent service logs
	docker-compose logs -f agent

logs-feedback:  ## Tail feedback service logs
	docker-compose logs -f feedback

# ─── Data Operations ─────────────────────────────────────────────────────────

seed:  ## Seed 200 historical interactions to bootstrap agent learning
	@echo "$(YELLOW)Seeding database with 200 historical interactions...$(RESET)"
	@sleep 5  # brief wait for services to fully start
	curl -s -X POST "http://localhost:8000/seed-database?n_interactions=200" | python3 -m json.tool
	@echo "$(GREEN)Database seeded!$(RESET)"

generate:  ## Generate 50 synthetic user interactions
	@echo "$(YELLOW)Generating 50 synthetic interactions...$(RESET)"
	curl -s -X POST "http://localhost:8000/generate-data?n_users=50" | python3 -m json.tool

metrics:  ## Display current system metrics
	curl -s "http://localhost:8000/metrics" | python3 -m json.tool

coupon:  ## Test: get a coupon recommendation (category=electronics, age=30)
	curl -s -X POST "http://localhost:8000/get-coupon" \
	  -H "Content-Type: application/json" \
	  -d '{"user_id":"demo_user","age":30,"category":"electronics","session_time":120}' \
	  | python3 -m json.tool

# ─── Testing ─────────────────────────────────────────────────────────────────

test:  ## Run integration tests (requires running system)
	@echo "$(YELLOW)Running integration tests...$(RESET)"
	python3 tests/test_integration.py

test-unit:  ## Run unit tests (no Docker required)
	@echo "$(YELLOW)Running unit tests...$(RESET)"
	pip install pytest --quiet
	pytest tests/test_bandit.py -v

test-load:  ## Start Locust load test (open http://localhost:8089)
	@echo "$(YELLOW)Starting Locust UI at http://localhost:8089$(RESET)"
	locust -f tests/locustfile.py --host=http://localhost:8000

# ─── Database ────────────────────────────────────────────────────────────────

shell-db:  ## Open psql shell in the postgres container
	docker-compose exec postgres psql -U coupon_user -d coupon_db

shell-redis:  ## Open redis-cli
	docker-compose exec redis redis-cli

shell-api:  ## Open bash shell in api container
	docker-compose exec api bash

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean: down-v  ## Remove containers, volumes, and dangling images
	docker system prune -f --filter "label=com.docker.compose.project=coupon-recommender"
	@echo "$(GREEN)Cleanup complete$(RESET)"

# ─── Full Setup ──────────────────────────────────────────────────────────────

setup: build up  ## Build + start + seed (full first-run setup)
	@echo "$(YELLOW)Waiting 20 seconds for services to initialize...$(RESET)"
	@sleep 20
	@$(MAKE) seed
	@echo ""
	@echo "$(GREEN)✅ System ready!$(RESET)"
	@echo "  Open the dashboard: http://localhost:3000"
	@echo "  Browse the API:     http://localhost:8000/docs"
