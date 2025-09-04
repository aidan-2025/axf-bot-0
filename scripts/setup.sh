#!/bin/bash

# AXF Bot 0 - Project Setup Script
# This script sets up the development environment for the axf-bot-0 project

set -e  # Exit on any error

echo "🚀 Setting up AXF Bot 0 development environment..."

# Check if Python 3.11+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker detected"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker Compose detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p app1/{data,models,logs}
mkdir -p app2/{generated,backtesting,logs}
mkdir -p logs
mkdir -p data

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp env.example .env
    echo "📝 Please edit .env file with your API keys and configuration"
fi

# Initialize Taskmaster if not already done
if [ ! -f ".taskmaster/tasks/tasks.json" ]; then
    echo "🎯 Initializing Taskmaster..."
    task-master init --yes
    task-master parse-prd .taskmaster/docs/prd.md --num-tasks 25 --research
fi

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
if [ -f "venv/bin/pre-commit" ]; then
    pre-commit install
fi

echo "✅ Setup complete!"
echo ""
echo "🎉 AXF Bot 0 is ready for development!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Start the applications:"
echo "   - App1 (Strategy Generator): cd app1 && python main.py"
echo "   - App2 (MT4 EA Generator): cd app2 && python main.py"
echo "3. Or use Docker: docker-compose up -d"
echo "4. View tasks: task-master list"
echo "5. Get next task: task-master next"
echo ""
echo "Happy coding! 🚀"
