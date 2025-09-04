# Contributing to AXF Bot 0

Thank you for your interest in contributing to AXF Bot 0! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Process](#contributing-process)
5. [Code Standards](#code-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Issue Reporting](#issue-reporting)
10. [Community Guidelines](#community-guidelines)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- Git
- Basic understanding of the project architecture

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/axf-bot-0.git
   cd axf-bot-0
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/aidan-2025/axf-bot-0.git
   ```

## Development Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r app1/requirements.txt
pip install -r app2/requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Environment Configuration

```bash
# Create environment file
make secrets-create

# Validate configuration
make secrets-validate
```

### 3. Start Development Environment

```bash
# Start all services
make dev-start

# Check status
make health-check
```

### 4. Verify Setup

```bash
# Run tests
make test

# Run linting
make lint

# Run type checking
make type-check
```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to express interest
- Wait for maintainer approval before starting work

### 2. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b bugfix/issue-number-description
```

### 3. Make Changes

- Follow the [code standards](#code-standards)
- Write tests for new functionality
- Update documentation as needed
- Commit changes with descriptive messages

### 4. Test Your Changes

```bash
# Run all tests
make test

# Run specific tests
make test-app1
make test-app2
make test-web-ui

# Run linting
make lint

# Run type checking
make type-check
```

### 5. Submit Pull Request

- Push your branch to your fork
- Create a pull request against the `develop` branch
- Fill out the pull request template
- Wait for review and feedback

## Code Standards

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Write docstrings for all functions, classes, and modules
- Use meaningful variable and function names
- Keep functions small and focused

**Example**:
```python
def calculate_strategy_performance(
    strategy_id: int,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, float]:
    """
    Calculate performance metrics for a strategy.
    
    Args:
        strategy_id: ID of the strategy
        start_date: Start date for calculation
        end_date: End date for calculation
        
    Returns:
        Dictionary containing performance metrics
        
    Raises:
        ValueError: If strategy_id is invalid
        DatabaseError: If database connection fails
    """
    # Implementation here
    pass
```

### JavaScript/TypeScript Code

- Use ESLint and Prettier for formatting
- Write JSDoc comments for functions
- Use meaningful variable and function names
- Follow React best practices
- Use TypeScript for type safety

**Example**:
```typescript
/**
 * Calculate strategy performance metrics
 * @param strategyId - ID of the strategy
 * @param startDate - Start date for calculation
 * @param endDate - End date for calculation
 * @returns Promise containing performance metrics
 */
async function calculateStrategyPerformance(
  strategyId: number,
  startDate: Date,
  endDate: Date
): Promise<StrategyPerformance> {
  // Implementation here
}
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build process or auxiliary tool changes

**Examples**:
```
feat(api): add strategy performance endpoint
fix(database): resolve connection timeout issue
docs(readme): update installation instructions
test(app1): add unit tests for strategy generation
```

## Testing

### Unit Tests

Write unit tests for all new functionality:

```python
# app1/tests/test_strategy_generation.py
import pytest
from app1.src.strategy_generation.engine import StrategyEngine

def test_generate_strategy():
    """Test strategy generation functionality."""
    engine = StrategyEngine()
    strategy = engine.generate_strategy(
        strategy_type="trend_following",
        parameters={"timeframe": "1H"}
    )
    
    assert strategy is not None
    assert strategy.strategy_type == "trend_following"
    assert strategy.parameters["timeframe"] == "1H"
```

### Integration Tests

Write integration tests for API endpoints:

```python
# tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from app1.main import app

client = TestClient(app)

def test_create_strategy():
    """Test strategy creation endpoint."""
    response = client.post(
        "/api/v1/strategies",
        json={
            "name": "Test Strategy",
            "strategy_type": "trend_following",
            "parameters": {"timeframe": "1H"}
        }
    )
    
    assert response.status_code == 201
    assert response.json()["name"] == "Test Strategy"
```

### Frontend Tests

Write tests for React components:

```typescript
// web-ui/tests/components/StrategyList.test.tsx
import { render, screen } from '@testing-library/react';
import { StrategyList } from '../components/StrategyList';

describe('StrategyList', () => {
  it('renders strategy list', () => {
    const strategies = [
      { id: 1, name: 'Test Strategy', type: 'trend_following' }
    ];
    
    render(<StrategyList strategies={strategies} />);
    
    expect(screen.getByText('Test Strategy')).toBeInTheDocument();
  });
});
```

## Documentation

### Code Documentation

- Write docstrings for all functions, classes, and modules
- Include type hints and parameter descriptions
- Add examples for complex functions
- Update README files when adding new features

### API Documentation

- Update OpenAPI/Swagger documentation
- Include request/response examples
- Document error codes and messages
- Add authentication requirements

### User Documentation

- Update user guides and tutorials
- Add screenshots for UI changes
- Update installation instructions
- Document configuration options

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows project standards
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Error handling is proper

### 2. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### 3. Review Process

- Maintainers will review your PR
- Address feedback promptly
- Make requested changes
- Keep PR focused and small
- Respond to comments

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g. macOS, Windows, Linux]
- Browser: [e.g. Chrome, Firefox, Safari]
- Version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
A clear description of any alternative solutions.

**Additional context**
Add any other context or screenshots.
```

## Community Guidelines

### Communication

- Be respectful and constructive
- Use clear and concise language
- Ask questions when unsure
- Help others when possible

### Code Review

- Provide constructive feedback
- Focus on the code, not the person
- Suggest improvements
- Acknowledge good work

### Getting Help

- Check existing documentation
- Search existing issues
- Ask in team chat
- Create new issue if needed

## Development Workflow

### 1. Daily Workflow

```bash
# Start development
make dev-start

# Make changes
# ... your code changes ...

# Test changes
make test

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push changes
git push origin feature/your-feature-name
```

### 2. Weekly Workflow

```bash
# Sync with upstream
git fetch upstream
git checkout develop
git merge upstream/develop

# Update your feature branch
git checkout feature/your-feature-name
git merge develop
```

### 3. Release Workflow

```bash
# Create release branch
git checkout -b release/v1.0.0

# Update version numbers
# Update CHANGELOG.md
# Update documentation

# Merge to main
git checkout main
git merge release/v1.0.0
git tag v1.0.0
git push origin main --tags
```

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributors page

## Questions?

- Check the [Development Guide](docs/DEVELOPMENT_GUIDE.md)
- Ask in the team chat
- Create an issue
- Contact maintainers

Thank you for contributing to AXF Bot 0! 🚀
