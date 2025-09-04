# Git Branching Strategy

## 🌳 Branch Structure

### Main Branches
- **`main`** - Production-ready code
- **`develop`** - Integration branch for features

### Supporting Branches
- **`feature/*`** - New features and enhancements
- **`hotfix/*`** - Critical production fixes
- **`release/*`** - Release preparation
- **`bugfix/*`** - Bug fixes for develop branch

## 🔄 Workflow

### Feature Development
```bash
# 1. Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/strategy-generation-engine

# 2. Develop feature
git add .
git commit -m "feat: implement strategy generation engine"

# 3. Push and create PR
git push origin feature/strategy-generation-engine
# Create PR: feature/strategy-generation-engine → develop
```

### Hotfix Process
```bash
# 1. Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-security-fix

# 2. Fix the issue
git add .
git commit -m "fix: resolve critical security vulnerability"

# 3. Push and create PR
git push origin hotfix/critical-security-fix
# Create PR: hotfix/critical-security-fix → main
# Also create PR: hotfix/critical-security-fix → develop
```

### Release Process
```bash
# 1. Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# 2. Prepare release (version bumps, changelog)
git add .
git commit -m "chore: prepare release v1.0.0"

# 3. Merge to main and tag
git checkout main
git merge release/v1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin main --tags

# 4. Merge back to develop
git checkout develop
git merge release/v1.0.0
git push origin develop

# 5. Delete release branch
git branch -d release/v1.0.0
git push origin --delete release/v1.0.0
```

## 📋 Branch Protection Rules

### Main Branch Protection
- Require pull request reviews (2 reviewers)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to main branch
- Require linear history

### Develop Branch Protection
- Require pull request reviews (1 reviewer)
- Require status checks to pass
- Require branches to be up to date
- Allow force pushes for maintainers

## 🏷️ Naming Conventions

### Branch Names
- **Features**: `feature/description` (e.g., `feature/user-authentication`)
- **Hotfixes**: `hotfix/description` (e.g., `hotfix/login-bug`)
- **Releases**: `release/version` (e.g., `release/v1.2.0`)
- **Bugfixes**: `bugfix/description` (e.g., `bugfix/database-connection`)

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Build process or auxiliary tool changes

## 🔧 GitHub Repository Setup

### Repository Settings
1. **General**
   - Repository name: `axf-bot-0`
   - Description: `AI-Powered Forex Trading System`
   - Visibility: Private (initially)

2. **Branches**
   - Default branch: `main`
   - Branch protection rules configured

3. **Security**
   - Dependency alerts enabled
   - Security advisories enabled
   - Code scanning enabled

### Required Secrets
Add these secrets to GitHub repository settings:
- `DATABASE_URL` - Production database connection
- `REDIS_URL` - Production Redis connection
- `API_KEYS` - External API keys (JSON format)
- `DOCKER_REGISTRY_TOKEN` - Container registry access token

## 🚀 CI/CD Pipeline

### Continuous Integration (CI)
Triggers on:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

Runs:
- Code linting and formatting
- Unit tests
- Integration tests
- Security scans
- Docker build tests

### Continuous Deployment (CD)
- **Staging**: Auto-deploy on push to `develop`
- **Production**: Auto-deploy on push to `main`
- **Manual**: Workflow dispatch for manual deployments

## 📊 Code Review Process

### Pull Request Requirements
1. **Description**: Clear description of changes
2. **Testing**: Evidence of testing performed
3. **Documentation**: Updated documentation if needed
4. **Breaking Changes**: Clearly marked if applicable

### Review Checklist
- [ ] Code follows project conventions
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Database migrations included if needed

## 🔍 Monitoring and Alerts

### Branch Status
- Monitor branch health via GitHub Insights
- Track merge frequency and conflicts
- Monitor PR review times

### Deployment Status
- Staging deployment notifications
- Production deployment notifications
- Rollback procedures documented

## 📚 Additional Resources

- [Git Flow Cheat Sheet](https://danielkummer.github.io/git-flow-cheatsheet/)
- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
