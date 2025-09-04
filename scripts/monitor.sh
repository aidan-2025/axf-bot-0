#!/bin/bash

# AXF Bot 0 - Monitoring Script
# Provides comprehensive system monitoring and health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check service health
check_service_health() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    print_status "Checking $service_name health..."
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        print_success "$service_name is healthy"
        return 0
    else
        print_error "$service_name is unhealthy"
        return 1
    fi
}

# Function to get detailed health information
get_detailed_health() {
    local service_name=$1
    local url=$2
    
    print_status "Getting detailed health for $service_name..."
    
    if response=$(curl -s "$url"); then
        echo "$response" | jq . 2>/dev/null || echo "$response"
    else
        print_error "Failed to get health information for $service_name"
    fi
}

# Function to check Docker container status
check_container_status() {
    local container_name=$1
    
    print_status "Checking Docker container: $container_name"
    
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container_name"; then
        local status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{print $2}')
        if [[ $status == *"Up"* ]]; then
            print_success "$container_name is running"
            return 0
        else
            print_error "$container_name is not running properly"
            return 1
        fi
    else
        print_error "$container_name is not running"
        return 1
    fi
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # CPU usage
    local cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        print_warning "High CPU usage: ${cpu_usage}%"
    else
        print_success "CPU usage: ${cpu_usage}%"
    fi
    
    # Memory usage
    local memory_usage=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
    local memory_total=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    local memory_percent=$((memory_usage * 100 / (memory_usage + memory_total)))
    
    if [ $memory_percent -gt 85 ]; then
        print_warning "High memory usage: ${memory_percent}%"
    else
        print_success "Memory usage: ${memory_percent}%"
    fi
    
    # Disk usage
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $disk_usage -gt 90 ]; then
        print_warning "High disk usage: ${disk_usage}%"
    else
        print_success "Disk usage: ${disk_usage}%"
    fi
}

# Function to check database connectivity
check_database_connectivity() {
    print_status "Checking database connectivity..."
    
    if docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U postgres -d axf_bot_db > /dev/null 2>&1; then
        print_success "Database is accessible"
        return 0
    else
        print_error "Database is not accessible"
        return 1
    fi
}

# Function to check Redis connectivity
check_redis_connectivity() {
    print_status "Checking Redis connectivity..."
    
    if docker-compose -f docker-compose.dev.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_success "Redis is accessible"
        return 0
    else
        print_error "Redis is not accessible"
        return 1
    fi
}

# Function to show service logs
show_service_logs() {
    local service_name=$1
    local lines=${2:-50}
    
    print_status "Showing last $lines lines of logs for $service_name..."
    docker-compose -f docker-compose.dev.yml logs --tail=$lines $service_name
}

# Function to show system overview
show_system_overview() {
    print_status "System Overview"
    echo "=================="
    echo ""
    
    # Docker containers status
    echo "Docker Containers:"
    docker-compose -f docker-compose.dev.yml ps
    echo ""
    
    # System resources
    check_system_resources
    echo ""
    
    # Service health checks
    echo "Service Health Checks:"
    check_service_health "Web UI" "http://localhost:3000/api/health"
    check_service_health "App1" "http://localhost:8000/health"
    check_service_health "App2" "http://localhost:8001/health"
    check_database_connectivity
    check_redis_connectivity
    echo ""
}

# Function to show detailed monitoring
show_detailed_monitoring() {
    print_status "Detailed Monitoring Information"
    echo "===================================="
    echo ""
    
    # Get detailed health for each service
    get_detailed_health "Web UI" "http://localhost:3000/api/health?detailed=true"
    echo ""
    
    get_detailed_health "App1" "http://localhost:8000/health/detailed"
    echo ""
    
    get_detailed_health "App2" "http://localhost:8001/health/detailed"
    echo ""
    
    # Show metrics
    print_status "Service Metrics:"
    echo "App1 Metrics:"
    curl -s "http://localhost:8000/metrics" | head -20
    echo ""
    
    echo "App2 Metrics:"
    curl -s "http://localhost:8001/metrics" | head -20
    echo ""
}

# Function to show help
show_help() {
    echo "AXF Bot 0 - Monitoring Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  overview         - Show system overview"
    echo "  detailed         - Show detailed monitoring information"
    echo "  health           - Check all service health"
    echo "  resources        - Check system resources"
    echo "  database         - Check database connectivity"
    echo "  redis            - Check Redis connectivity"
    echo "  logs [service]   - Show service logs (default: all services)"
    echo "  watch            - Watch system status continuously"
    echo "  help             - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 overview"
    echo "  $0 logs app1"
    echo "  $0 watch"
}

# Function to watch system status
watch_system_status() {
    print_status "Watching system status (Press Ctrl+C to stop)..."
    echo ""
    
    while true; do
        clear
        echo "AXF Bot 0 - System Status Monitor"
        echo "================================="
        echo "Last updated: $(date)"
        echo ""
        
        show_system_overview
        
        sleep 5
    done
}

# Main script logic
case "${1:-help}" in
    overview)
        show_system_overview
        ;;
    detailed)
        show_detailed_monitoring
        ;;
    health)
        check_service_health "Web UI" "http://localhost:3000/api/health"
        check_service_health "App1" "http://localhost:8000/health"
        check_service_health "App2" "http://localhost:8001/health"
        check_database_connectivity
        check_redis_connectivity
        ;;
    resources)
        check_system_resources
        ;;
    database)
        check_database_connectivity
        ;;
    redis)
        check_redis_connectivity
        ;;
    logs)
        if [ -n "$2" ]; then
            show_service_logs "$2"
        else
            docker-compose -f docker-compose.dev.yml logs -f
        fi
        ;;
    watch)
        watch_system_status
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
