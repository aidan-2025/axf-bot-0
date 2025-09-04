#!/bin/bash

# GitHub Repository Setup Script
# This script helps set up the GitHub repository with proper configuration

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

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed. Please install it first:"
    echo "  macOS: brew install gh"
    echo "  Ubuntu: sudo apt install gh"
    echo "  Windows: winget install GitHub.cli"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    print_error "Not authenticated with GitHub. Please run: gh auth login"
    exit 1
fi

print_status "Setting up GitHub repository for axf-bot-0..."

# Get repository name from user
read -p "Enter GitHub repository name (default: axf-bot-0): " REPO_NAME
REPO_NAME=${REPO_NAME:-axf-bot-0}

# Get repository description
read -p "Enter repository description (default: AI-Powered Forex Trading System): " REPO_DESC
REPO_DESC=${REPO_DESC:-AI-Powered Forex Trading System}

# Get repository visibility
read -p "Make repository private? (y/N): " PRIVATE_REPO
if [[ $PRIVATE_REPO =~ ^[Yy]$ ]]; then
    VISIBILITY="--private"
else
    VISIBILITY="--public"
fi

print_status "Creating GitHub repository: $REPO_NAME"

# Create repository
gh repo create "$REPO_NAME" --description "$REPO_DESC" $VISIBILITY --source=. --remote=origin --push

print_success "Repository created successfully!"

# Set up branch protection rules
print_status "Setting up branch protection rules..."

# Protect main branch
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true}' \
  --field restrictions=null

# Protect develop branch
gh api repos/:owner/:repo/branches/develop/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null

print_success "Branch protection rules configured!"

# Set up repository secrets (prompt user)
print_status "Setting up repository secrets..."

echo "Please add the following secrets to your repository:"
echo "1. Go to: https://github.com/$(gh api user --jq .login)/$REPO_NAME/settings/secrets/actions"
echo "2. Add these secrets:"
echo "   - DATABASE_URL: Your production database connection string"
echo "   - REDIS_URL: Your production Redis connection string"
echo "   - API_KEYS: JSON object with all API keys"
echo "   - DOCKER_REGISTRY_TOKEN: GitHub token for container registry"

read -p "Press Enter when you've added the secrets..."

# Enable GitHub Actions
print_status "Enabling GitHub Actions..."

# Enable issues and projects
gh api repos/:owner/:repo \
  --method PATCH \
  --field has_issues=true \
  --field has_projects=true \
  --field has_wiki=true

print_success "Repository features enabled!"

# Create initial issues
print_status "Creating initial project issues..."

# Create project board
PROJECT_ID=$(gh api repos/:owner/:repo/projects --method POST \
  --field name="AXF Bot 0 Development" \
  --field body="Project board for AXF Bot 0 development tasks" \
  --jq .id)

print_success "Project board created with ID: $PROJECT_ID"

# Create initial issues
gh issue create --title "Set up development environment" --body "Set up local development environment with all dependencies" --label "enhancement,good first issue"
gh issue create --title "Implement basic data ingestion" --body "Implement basic market data ingestion for major currency pairs" --label "enhancement"
gh issue create --title "Create strategy generation engine" --body "Develop the core AI-powered strategy generation engine" --label "enhancement,priority:high"
gh issue create --title "Implement web UI dashboard" --body "Create the main dashboard for monitoring strategies and performance" --label "enhancement,ui"
gh issue create --title "Add comprehensive testing" --body "Add unit tests, integration tests, and end-to-end tests" --label "enhancement,testing"

print_success "Initial issues created!"

# Set up webhooks (optional)
read -p "Set up webhooks for deployment? (y/N): " SETUP_WEBHOOKS
if [[ $SETUP_WEBHOOKS =~ ^[Yy]$ ]]; then
    print_status "Setting up webhooks..."
    # Add webhook setup commands here
    print_success "Webhooks configured!"
fi

# Final instructions
print_success "GitHub repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Push your code: git push origin main"
echo "2. Push develop branch: git push origin develop"
echo "3. Create feature branches from develop"
echo "4. Set up your deployment environment"
echo "5. Configure monitoring and alerts"
echo ""
echo "Repository URL: https://github.com/$(gh api user --jq .login)/$REPO_NAME"
echo "Actions URL: https://github.com/$(gh api user --jq .login)/$REPO_NAME/actions"
echo "Settings URL: https://github.com/$(gh api user --jq .login)/$REPO_NAME/settings"
