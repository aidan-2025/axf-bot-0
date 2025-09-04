#!/bin/bash

# AXF Bot 0 - Development Environment Setup Script
# This script sets up the complete development environment

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        echo "  macOS: brew install --cask docker"
        echo "  Ubuntu: sudo apt install docker.io"
        echo "  Windows: Download from https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "  macOS: brew install docker-compose"
        echo "  Ubuntu: sudo apt install docker-compose"
        exit 1
    fi
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.11+ first."
        echo "  macOS: brew install python@3.11"
        echo "  Ubuntu: sudo apt install python3.11"
        exit 1
    fi
    
    # Check Node.js
    if ! command_exists node; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        echo "  macOS: brew install node@18"
        echo "  Ubuntu: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs"
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

# Function to create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install main requirements
    pip install -r requirements.txt
    
    # Install app-specific requirements
    pip install -r app1/requirements.txt
    pip install -r app2/requirements.txt
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy debugpy
    
    print_success "Python dependencies installed"
}

# Function to install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    cd web-ui
    npm install
    cd ..
    
    print_success "Node.js dependencies installed"
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp env.development .env
        print_success "Environment file created (.env)"
        print_warning "Please update .env with your actual API keys and configuration"
    else
        print_warning "Environment file already exists"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p app1/data app1/models app1/logs
    mkdir -p app2/generated app2/backtesting app2/logs
    mkdir -p web-ui/.next
    mkdir -p logs data models
    
    print_success "Directories created"
}

# Function to set up Git hooks
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    # Create pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for AXF Bot 0

echo "Running pre-commit checks..."

# Run linting
echo "Running linting..."
python -m flake8 app1/ app2/ --max-line-length=100 --ignore=E203,W503
if [ $? -ne 0 ]; then
    echo "❌ Linting failed. Please fix the issues and try again."
    exit 1
fi

# Run type checking
echo "Running type checking..."
python -m mypy app1/ app2/ --ignore-missing-imports
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed. Please fix the issues and try again."
    exit 1
fi

# Run tests
echo "Running tests..."
python -m pytest app1/tests/ app2/tests/ --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix the issues and try again."
    exit 1
fi

echo "✅ Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    print_success "Git hooks configured"
}

# Function to build Docker images
build_docker_images() {
    print_status "Building Docker images..."
    
    # Build development images
    docker-compose -f docker-compose.dev.yml build
    
    print_success "Docker images built"
}

# Function to start development environment
start_dev_environment() {
    print_status "Starting development environment..."
    
    # Start services
    docker-compose -f docker-compose.dev.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Run database migrations
    print_status "Running database migrations..."
    python scripts/migrate.py --migrate
    
    print_success "Development environment started!"
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📊 Services available at:"
    echo "  - Web UI: http://localhost:3000"
    echo "  - App1 (Strategy Generator): http://localhost:8000"
    echo "  - App2 (MT4 EA Generator): http://localhost:8001"
    echo "  - Database Admin: http://localhost:5050 (admin@axf-bot.com / admin)"
    echo "  - Redis Admin: http://localhost:8002"
    echo ""
    echo "🔧 Development commands:"
    echo "  make dev-start    - Start development environment"
    echo "  make dev-stop     - Stop development environment"
    echo "  make dev-logs     - View logs"
    echo "  make dev-shell    - Open shell in container"
    echo "  make health-check - Check system health"
    echo ""
    echo "📚 Next steps:"
    echo "  1. Update .env file with your API keys"
    echo "  2. Run 'make health-check' to verify everything is working"
    echo "  3. Start developing!"
}

# Function to show help
show_help() {
    echo "AXF Bot 0 - Development Environment Setup"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-docker    Skip Docker image building"
    echo "  --skip-start     Skip starting the development environment"
    echo "  --help           Show this help message"
    echo ""
    echo "This script will:"
    echo "  1. Check system requirements"
    echo "  2. Create Python virtual environment"
    echo "  3. Install all dependencies"
    echo "  4. Create necessary directories"
    echo "  5. Set up Git hooks"
    echo "  6. Build Docker images"
    echo "  7. Start development environment"
    echo "  8. Run database migrations"
}

# Parse command line arguments
SKIP_DOCKER=false
SKIP_START=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-start)
            SKIP_START=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main setup process
main() {
    echo "🚀 AXF Bot 0 - Development Environment Setup"
    echo "============================================="
    echo ""
    
    check_requirements
    create_venv
    install_python_deps
    install_node_deps
    create_env_file
    create_directories
    setup_git_hooks
    
    if [ "$SKIP_DOCKER" = false ]; then
        build_docker_images
    fi
    
    if [ "$SKIP_START" = false ]; then
        start_dev_environment
    else
        print_success "Setup completed (skipped starting environment)"
        echo ""
        echo "To start the development environment, run:"
        echo "  make dev-start"
    fi
}

# Run main function
main
