#!/bin/bash

# AXF Bot 0 - Environment Testing Script
# Comprehensive testing of the local development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_test_header() {
    echo ""
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TOTAL_TESTS++))
    
    print_status "Running: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        if [ $? -eq $expected_exit_code ]; then
            print_success "$test_name"
            return 0
        else
            print_error "$test_name (unexpected exit code: $?)"
            return 1
        fi
    else
        print_error "$test_name"
        return 1
    fi
}

# Function to test HTTP endpoint
test_http_endpoint() {
    local test_name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing HTTP endpoint: $test_name"
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [ "$response_code" = "$expected_status" ]; then
        print_success "$test_name (HTTP $response_code)"
        return 0
    else
        print_error "$test_name (HTTP $response_code, expected $expected_status)"
        return 1
    fi
}

# Function to test Docker container
test_docker_container() {
    local container_name="$1"
    local expected_status="${2:-Up}"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing Docker container: $container_name"
    
    local container_status
    container_status=$(docker-compose -f docker-compose.dev.yml ps --format "table {{.Name}}\t{{.Status}}" | grep "$container_name" | awk '{print $2}' || echo "Not Found")
    
    if [[ $container_status == *"$expected_status"* ]]; then
        print_success "$container_name container is $container_status"
        return 0
    else
        print_error "$container_name container status: $container_status (expected: $expected_status)"
        return 1
    fi
}

# Function to test database connection
test_database_connection() {
    local test_name="$1"
    local db_name="$2"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing database connection: $test_name"
    
    if docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U postgres -d "$db_name" > /dev/null 2>&1; then
        print_success "$test_name database connection"
        return 0
    else
        print_error "$test_name database connection failed"
        return 1
    fi
}

# Function to test Redis connection
test_redis_connection() {
    local test_name="$1"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing Redis connection: $test_name"
    
    if docker-compose -f docker-compose.dev.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_success "$test_name Redis connection"
        return 0
    else
        print_error "$test_name Redis connection failed"
        return 1
    fi
}

# Function to test file existence
test_file_exists() {
    local test_name="$1"
    local file_path="$2"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing file existence: $test_name"
    
    if [ -f "$file_path" ]; then
        print_success "$test_name file exists"
        return 0
    else
        print_error "$test_name file not found: $file_path"
        return 1
    fi
}

# Function to test directory existence
test_directory_exists() {
    local test_name="$1"
    local dir_path="$2"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing directory existence: $test_name"
    
    if [ -d "$dir_path" ]; then
        print_success "$test_name directory exists"
        return 0
    else
        print_error "$test_name directory not found: $dir_path"
        return 1
    fi
}

# Function to test command availability
test_command_available() {
    local test_name="$1"
    local command="$2"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing command availability: $test_name"
    
    if command -v "$command" > /dev/null 2>&1; then
        print_success "$test_name command available"
        return 0
    else
        print_error "$test_name command not found: $command"
        return 1
    fi
}

# Function to test environment variables
test_env_variable() {
    local test_name="$1"
    local var_name="$2"
    
    ((TOTAL_TESTS++))
    
    print_status "Testing environment variable: $test_name"
    
    if [ -n "${!var_name}" ]; then
        print_success "$test_name environment variable set"
        return 0
    else
        print_error "$test_name environment variable not set: $var_name"
        return 1
    fi
}

# Main testing function
run_all_tests() {
    echo "🧪 AXF Bot 0 - Environment Testing"
    echo "=================================="
    echo ""
    
    # Test 1: Prerequisites
    print_test_header "Testing Prerequisites"
    
    test_command_available "Docker" "docker"
    test_command_available "Docker Compose" "docker-compose"
    test_command_available "Python" "python3"
    test_command_available "Node.js" "node"
    test_command_available "npm" "npm"
    test_command_available "Git" "git"
    test_command_available "Make" "make"
    test_command_available "curl" "curl"
    
    # Test 2: Project Structure
    print_test_header "Testing Project Structure"
    
    test_directory_exists "Project root" "."
    test_directory_exists "App1" "app1"
    test_directory_exists "App2" "app2"
    test_directory_exists "Web UI" "web-ui"
    test_directory_exists "Scripts" "scripts"
    test_directory_exists "Documentation" "docs"
    test_directory_exists "Configuration" "config"
    
    test_file_exists "Docker Compose" "docker-compose.yml"
    test_file_exists "Docker Compose Dev" "docker-compose.dev.yml"
    test_file_exists "Docker Compose Secrets" "docker-compose.secrets.yml"
    test_file_exists "Makefile" "Makefile"
    test_file_exists "Environment Template" "env.development"
    test_file_exists "Environment Template" "env.production"
    
    # Test 3: Environment Configuration
    print_test_header "Testing Environment Configuration"
    
    test_file_exists "Environment file" ".env"
    test_file_exists "Environment manager" "config/environment.py"
    test_file_exists "Secrets manager" "scripts/manage-secrets.sh"
    
    # Test 4: Docker Services
    print_test_header "Testing Docker Services"
    
    test_docker_container "app1"
    test_docker_container "app2"
    test_docker_container "web-ui"
    test_docker_container "postgres"
    test_docker_container "redis"
    test_docker_container "influxdb"
    
    # Test 5: Database Connections
    print_test_header "Testing Database Connections"
    
    test_database_connection "PostgreSQL" "axf_bot_db"
    test_redis_connection "Redis"
    
    # Test 6: HTTP Endpoints
    print_test_header "Testing HTTP Endpoints"
    
    test_http_endpoint "Web UI Health" "http://localhost:3000/api/health"
    test_http_endpoint "App1 Health" "http://localhost:8000/health"
    test_http_endpoint "App2 Health" "http://localhost:8001/health"
    test_http_endpoint "App1 API Docs" "http://localhost:8000/docs"
    test_http_endpoint "App2 API Docs" "http://localhost:8001/docs"
    
    # Test 7: API Functionality
    print_test_header "Testing API Functionality"
    
    # Test App1 API endpoints
    test_http_endpoint "App1 Strategies" "http://localhost:8000/api/v1/strategies"
    test_http_endpoint "App1 Performance" "http://localhost:8000/api/v1/performance"
    test_http_endpoint "App1 Data" "http://localhost:8000/api/v1/data"
    test_http_endpoint "App1 Sentiment" "http://localhost:8000/api/v1/sentiment"
    
    # Test App2 API endpoints
    test_http_endpoint "App2 EA Generation" "http://localhost:8001/api/v1/ea/generate"
    test_http_endpoint "App2 Backtesting" "http://localhost:8001/api/v1/ea/backtest"
    
    # Test 8: Monitoring
    print_test_header "Testing Monitoring"
    
    test_http_endpoint "Grafana" "http://localhost:3001"
    test_file_exists "Prometheus config" "deployment/prometheus.yml"
    test_file_exists "Alert rules" "deployment/alert_rules.yml"
    test_file_exists "Grafana datasource" "deployment/grafana/datasources/prometheus.yml"
    test_file_exists "Grafana dashboard" "deployment/grafana/dashboards/axf-bot-overview.json"
    
    # Test 9: Scripts and Tools
    print_test_header "Testing Scripts and Tools"
    
    test_file_exists "Monitor script" "scripts/monitor.sh"
    test_file_exists "Secrets manager" "scripts/manage-secrets.sh"
    test_file_exists "Dev workflow" "scripts/dev-workflow.sh"
    test_file_exists "Migration script" "scripts/migrate.py"
    
    # Test 10: Documentation
    print_test_header "Testing Documentation"
    
    test_file_exists "Quick Start Guide" "docs/QUICK_START.md"
    test_file_exists "Development Guide" "docs/DEVELOPMENT_GUIDE.md"
    test_file_exists "Database Schema" "docs/DATABASE_SCHEMA.md"
    test_file_exists "Deployment Guide" "docs/DEPLOYMENT.md"
    test_file_exists "Environment Setup" "docs/ENVIRONMENT_SETUP.md"
    test_file_exists "Contributing Guide" "CONTRIBUTING.md"
    
    # Test 11: Make Commands
    print_test_header "Testing Make Commands"
    
    run_test "Make help" "make help"
    run_test "Make dev-start" "make dev-start"
    run_test "Make health-check" "make health-check"
    run_test "Make monitor" "make monitor"
    run_test "Make secrets-status" "make secrets-status"
    
    # Test 12: Performance
    print_test_header "Testing Performance"
    
    # Test response times
    local start_time
    local end_time
    local response_time
    
    start_time=$(date +%s%N)
    curl -s http://localhost:8000/health > /dev/null
    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 ))
    
    if [ $response_time -lt 1000 ]; then
        print_success "App1 response time: ${response_time}ms"
    else
        print_warning "App1 response time: ${response_time}ms (slow)"
    fi
    
    start_time=$(date +%s%N)
    curl -s http://localhost:8001/health > /dev/null
    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 ))
    
    if [ $response_time -lt 1000 ]; then
        print_success "App2 response time: ${response_time}ms"
    else
        print_warning "App2 response time: ${response_time}ms (slow)"
    fi
    
    start_time=$(date +%s%N)
    curl -s http://localhost:3000/api/health > /dev/null
    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 ))
    
    if [ $response_time -lt 1000 ]; then
        print_success "Web UI response time: ${response_time}ms"
    else
        print_warning "Web UI response time: ${response_time}ms (slow)"
    fi
}

# Function to show test summary
show_test_summary() {
    echo ""
    echo "=========================================="
    echo "🧪 Test Summary"
    echo "=========================================="
    echo ""
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Environment is ready for development.${NC}"
        return 0
    else
        echo -e "${RED}❌ Some tests failed. Please check the errors above.${NC}"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "AXF Bot 0 - Environment Testing Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick      Run quick tests only"
    echo "  --full       Run full test suite (default)"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0            # Run full test suite"
    echo "  $0 --quick    # Run quick tests only"
}

# Main script logic
case "${1:-}" in
    --quick)
        print_status "Running quick tests..."
        # Run only essential tests
        run_test "Docker" "docker --version"
        run_test "Docker Compose" "docker-compose --version"
        test_http_endpoint "Web UI" "http://localhost:3000/api/health"
        test_http_endpoint "App1" "http://localhost:8000/health"
        test_http_endpoint "App2" "http://localhost:8001/health"
        show_test_summary
        ;;
    --help|-h)
        show_help
        ;;
    *)
        print_status "Running full test suite..."
        run_all_tests
        show_test_summary
        ;;
esac
