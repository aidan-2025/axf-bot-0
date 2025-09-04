# AXF Bot 0 - Makefile
# Common development commands and workflows

.PHONY: help setup install test test-app1 test-app2 lint format start-app1 start-app2 start-docker stop-docker logs db-setup db-reset tasks next-task clean

# Default target
help:
	@echo "AXF Bot 0 - Available Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup           - Initial project setup"
	@echo "  install         - Install all dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test            - Run all tests with coverage"
	@echo "  test-app1       - Run App1 tests only"
	@echo "  test-app2       - Run App2 tests only"
	@echo "  lint            - Run linting checks"
	@echo "  format          - Format code with black"
	@echo ""
	@echo "Development:"
	@echo "  start-app1      - Start App1 (Strategy Generator)"
	@echo "  start-app2      - Start App2 (MT4 EA Generator)"
	@echo "  start-docker    - Start all services with Docker"
	@echo "  stop-docker     - Stop all Docker services"
	@echo "  logs            - View Docker logs"
	@echo ""
	@echo "Database:"
	@echo "  db-setup        - Set up database"
	@echo "  db-reset         - Reset database (WARNING: deletes data)"
	@echo ""
	@echo "Task Management:"
	@echo "  tasks           - Show Taskmaster tasks"
	@echo "  next-task       - Show next task to work on"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean           - Clean temporary files"
	@echo ""

# Setup and installation
setup:
	@./scripts/setup.sh

install:
	@./scripts/dev-workflow.sh install

# Testing
test:
	@./scripts/dev-workflow.sh test

test-app1:
	@./scripts/dev-workflow.sh test-app1

test-app2:
	@./scripts/dev-workflow.sh test-app2

# Code quality
lint:
	@./scripts/dev-workflow.sh lint

format:
	@./scripts/dev-workflow.sh format

# Development servers
start-app1:
	@./scripts/dev-workflow.sh start-app1

start-app2:
	@./scripts/dev-workflow.sh start-app2

# Docker services
start-docker:
	@./scripts/dev-workflow.sh start-docker

stop-docker:
	@./scripts/dev-workflow.sh stop-docker

logs:
	@./scripts/dev-workflow.sh logs

# Database operations
db-setup:
	@./scripts/dev-workflow.sh db-setup

db-reset:
	@./scripts/dev-workflow.sh db-reset

# Task management
tasks:
	@./scripts/dev-workflow.sh tasks

next-task:
	@./scripts/dev-workflow.sh next-task

# Maintenance
clean:
	@./scripts/dev-workflow.sh clean

# Quick development workflow
dev: install test lint format
	@echo "Development environment ready!"

# Full setup from scratch
full-setup: setup install db-setup
	@echo "Full setup completed!"

# Production deployment
deploy:
	@echo "Deploying to production..."
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Production deployment completed!"

# Backup database
backup-db:
	@echo "Creating database backup..."
	@docker-compose exec postgres pg_dump -U postgres axf_bot_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backup created!"

# Restore database
restore-db:
	@echo "Restoring database from backup..."
	@read -p "Enter backup filename: " backup_file; \
	docker-compose exec -T postgres psql -U postgres axf_bot_db < $$backup_file
	@echo "Database restored!"

# Monitor system resources
monitor:
	@echo "System resource monitoring..."
	@docker stats

# Check system health
health:
	@echo "Checking system health..."
	@curl -s http://localhost:8000/health | jq .
	@curl -s http://localhost:8001/health | jq .
