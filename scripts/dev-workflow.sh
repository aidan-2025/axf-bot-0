#!/bin/bash

# AXF Bot 0 - Development Workflow Script
# This script provides common development commands and workflows

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if virtual environment is active
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not active. Activating..."
        source venv/bin/activate
    fi
}

# Function to show help
show_help() {
    echo "AXF Bot 0 - Development Workflow Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup           - Initial project setup"
    echo "  install         - Install dependencies"
    echo "  test            - Run all tests"
    echo "  test-app1       - Run App1 tests only"
    echo "  test-app2       - Run App2 tests only"
    echo "  lint            - Run linting checks"
    echo "  format          - Format code with black"
    echo "  start-app1      - Start App1 (Strategy Generator)"
    echo "  start-app2      - Start App2 (MT4 EA Generator)"
    echo "  start-docker    - Start all services with Docker"
    echo "  stop-docker     - Stop all Docker services"
    echo "  logs            - View Docker logs"
    echo "  db-setup        - Set up database"
    echo "  db-reset        - Reset database"
    echo "  tasks           - Show Taskmaster tasks"
    echo "  next-task       - Show next task to work on"
    echo "  clean           - Clean temporary files"
    echo "  help            - Show this help message"
}

# Function to install dependencies
install_deps() {
    print_status "Installing dependencies..."
    check_venv
    pip install -r requirements.txt
    pip install -r app1/requirements.txt
    pip install -r app2/requirements.txt
    print_success "Dependencies installed"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    check_venv
    pytest --cov=app1 --cov=app2 --cov-report=html --cov-report=term
    print_success "Tests completed"
}

run_tests_app1() {
    print_status "Running App1 tests..."
    check_venv
    pytest app1/tests/ --cov=app1 --cov-report=html --cov-report=term
    print_success "App1 tests completed"
}

run_tests_app2() {
    print_status "Running App2 tests..."
    check_venv
    pytest app2/tests/ --cov=app2 --cov-report=html --cov-report=term
    print_success "App2 tests completed"
}

# Function to run linting
run_lint() {
    print_status "Running linting checks..."
    check_venv
    flake8 app1/ app2/ --max-line-length=100 --ignore=E203,W503
    mypy app1/ app2/ --ignore-missing-imports
    print_success "Linting completed"
}

# Function to format code
format_code() {
    print_status "Formatting code..."
    check_venv
    black app1/ app2/ --line-length=100
    print_success "Code formatted"
}

# Function to start App1
start_app1() {
    print_status "Starting App1 (Strategy Generator)..."
    check_venv
    cd app1
    python main.py
}

# Function to start App2
start_app2() {
    print_status "Starting App2 (MT4 EA Generator)..."
    check_venv
    cd app2
    python main.py
}

# Function to start Docker services
start_docker() {
    print_status "Starting Docker services..."
    docker-compose up -d
    print_success "Docker services started"
    print_status "Services available at:"
    echo "  - App1 (Strategy Generator): http://localhost:8000"
    echo "  - App2 (MT4 EA Generator): http://localhost:8001"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
}

# Function to stop Docker services
stop_docker() {
    print_status "Stopping Docker services..."
    docker-compose down
    print_success "Docker services stopped"
}

# Function to view logs
view_logs() {
    print_status "Viewing Docker logs..."
    docker-compose logs -f
}

# Function to set up database
db_setup() {
    print_status "Setting up database..."
    docker-compose up -d postgres
    sleep 10
    docker-compose exec postgres psql -U postgres -d axf_bot_db -f /docker-entrypoint-initdb.d/init.sql
    print_success "Database setup completed"
}

# Function to reset database
db_reset() {
    print_warning "Resetting database (this will delete all data)..."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v
        docker-compose up -d postgres
        sleep 10
        docker-compose exec postgres psql -U postgres -d axf_bot_db -f /docker-entrypoint-initdb.d/init.sql
        print_success "Database reset completed"
    else
        print_status "Database reset cancelled"
    fi
}

# Function to show tasks
show_tasks() {
    print_status "Showing Taskmaster tasks..."
    check_venv
    task-master list --with-subtasks
}

# Function to show next task
show_next_task() {
    print_status "Showing next task to work on..."
    check_venv
    task-master next
}

# Function to clean temporary files
clean_files() {
    print_status "Cleaning temporary files..."
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name ".pytest_cache" -delete
    find . -type f -name ".coverage" -delete
    rm -rf htmlcov/
    rm -rf .mypy_cache/
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-help}" in
    setup)
        ./scripts/setup.sh
        ;;
    install)
        install_deps
        ;;
    test)
        run_tests
        ;;
    test-app1)
        run_tests_app1
        ;;
    test-app2)
        run_tests_app2
        ;;
    lint)
        run_lint
        ;;
    format)
        format_code
        ;;
    start-app1)
        start_app1
        ;;
    start-app2)
        start_app2
        ;;
    start-docker)
        start_docker
        ;;
    stop-docker)
        stop_docker
        ;;
    logs)
        view_logs
        ;;
    db-setup)
        db_setup
        ;;
    db-reset)
        db_reset
        ;;
    tasks)
        show_tasks
        ;;
    next-task)
        show_next_task
        ;;
    clean)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
