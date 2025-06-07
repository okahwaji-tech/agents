# Development Setup

This guide covers the development workflow, tools, and best practices for working with the LLM & Agentic Workflows repository.

## Development Workflow

### Daily Development Routine

1. **Activate Environment**
   ```bash
   source agents/bin/activate
   ```

2. **Update Dependencies** (if needed)
   ```bash
   uv sync
   ```

3. **Run Quality Checks**
   ```bash
   uv run poe check_all
   ```

4. **Start Development Server**
   ```bash
   # For documentation
   uv run poe docs-serve
   
   # For Jupyter Lab
   uv run jupyter lab
   ```

### Code Quality Standards

We maintain high code quality standards with automated tools:

#### Formatting with Ruff

```bash
# Format all Python files
uv run poe format

# Check formatting without making changes
uv run ruff format --check .
```

#### Linting with Ruff

```bash
# Lint all Python files
uv run poe lint

# Fix auto-fixable issues
uv run ruff check --fix .
```

#### Type Checking with MyPy

```bash
# Type check all Python files
uv run poe typecheck

# Type check specific file
uv run mypy path/to/file.py
```

#### Testing with Pytest

```bash
# Run all tests
uv run poe test

# Run specific test file
uv run pytest tests/test_specific.py

# Run with coverage
uv run pytest --cov=src
```

## Project Structure

```
agents/
├── docs/                    # Documentation source files
├── materials/               # Learning materials by week
│   └── weeks-1/            # Week 1 materials
├── code/                   # Code examples and implementations
│   └── week-1/             # Week 1 code examples
├── scripts/                # Utility scripts
├── tests/                  # Test files
├── pyproject.toml          # Project configuration
├── mkdocs.yml             # Documentation configuration
└── README.md              # Project overview
```

## Development Tools

### 1. Poethepoet (poe) Task Runner

We use `poe` for common development tasks:

```bash
# Available tasks
uv run poe --help

# Common tasks
uv run poe format      # Format code
uv run poe lint        # Lint code
uv run poe typecheck   # Type checking
uv run poe test        # Run tests
uv run poe check_all   # Run all checks
uv run poe clean_cache # Clean Python cache
```

### 2. Jupyter Lab Configuration

Optimized Jupyter Lab setup for LLM development:

```bash
# Start Jupyter Lab
uv run jupyter lab

# With specific port
uv run jupyter lab --port=8888

# With custom config
uv run jupyter lab --config=jupyter_config.py
```

### 3. Documentation Development

#### Local Development

```bash
# Serve docs locally with auto-reload
uv run poe docs-serve

# Build docs for production
uv run poe docs-build

# Deploy to GitHub Pages
uv run poe docs-deploy
```

#### Writing Documentation

- Use **Markdown** for all documentation
- Follow **Material for MkDocs** conventions
- Include **code examples** with syntax highlighting
- Add **progress tracking tables** for each week
- Use **admonitions** for important notes

Example documentation structure:

```markdown
# Week Title

## Overview
Brief description of the week's content.

## Mathematical Foundations
Mathematical concepts covered this week.

## Key Readings
Important papers and resources.

## Hands-On Deliverable
Practical implementation tasks.

## Progress Tracking
| Task | Status | Notes |
|------|--------|-------|
| ... | ... | ... |
```

## Code Standards

### Python Style Guide

We follow these conventions:

1. **PEP 8** compliance (enforced by Ruff)
2. **Type hints** for all functions and methods
3. **Docstrings** for all public functions
4. **Maximum line length**: 88 characters
5. **Import sorting**: isort-compatible

Example function:

```python
from typing import List, Optional
import torch
import logging

logger = logging.getLogger(__name__)

def process_medical_text(
    text: str,
    model: torch.nn.Module,
    max_length: Optional[int] = None
) -> List[str]:
    """
    Process medical text using the specified model.
    
    Args:
        text: Input medical text to process
        model: PyTorch model for text processing
        max_length: Maximum sequence length (optional)
        
    Returns:
        List of processed text segments
        
    Raises:
        ValueError: If text is empty or model is not initialized
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    
    logger.info(f"Processing text of length {len(text)}")
    
    # Implementation here
    return processed_segments
```

### Healthcare AI Considerations

When working with healthcare applications:

1. **Privacy**: Never commit real patient data
2. **Safety**: Add warnings for medical advice
3. **Compliance**: Follow HIPAA guidelines
4. **Bias**: Test for demographic biases
5. **Validation**: Include medical expert review

Example safety warning:

```python
import warnings

def medical_diagnosis_helper(symptoms: List[str]) -> str:
    """
    MEDICAL DISCLAIMER: This is for educational purposes only.
    Always consult qualified healthcare professionals for medical advice.
    """
    warnings.warn(
        "This tool is for educational purposes only. "
        "Do not use for actual medical diagnosis.",
        UserWarning
    )
    # Implementation here
```

## Git Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches
- `docs/*`: Documentation updates
- `week-*`: Weekly learning materials

### Commit Messages

Use conventional commit format:

```
type(scope): description

feat(week-1): add probability theory materials
fix(docs): correct navigation links
docs(readme): update installation instructions
test(week-1): add unit tests for LLM evaluation
```

### Pre-commit Hooks

Set up pre-commit hooks for quality checks:

```bash
# Install pre-commit
uv add --dev pre-commit

# Install hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

## Environment Configuration

### Environment Variables

Create `.env` file for local development:

```bash
# API Keys (optional)
OPENAI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here

# Apple Silicon Optimizations
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_ENABLE_MPS_FALLBACK=1

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

### VS Code Configuration

Recommended VS Code settings (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./agents/bin/python",
    "python.formatting.provider": "none",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    "ruff.args": ["--config=pyproject.toml"],
    "mypy.configFile": "pyproject.toml"
}
```

## Performance Monitoring

### Apple Silicon Optimization

Monitor performance with:

```python
import torch
import time
import psutil

def monitor_performance():
    """Monitor system performance during model training."""
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
```

### Memory Management

```python
import gc
import torch

def cleanup_memory():
    """Clean up GPU and system memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
```

## Troubleshooting

### Common Development Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Type Errors**: Run `uv run poe typecheck` for details
3. **Format Issues**: Run `uv run poe format` to auto-fix
4. **Memory Issues**: Use `cleanup_memory()` function
5. **Documentation Errors**: Check `mkdocs.yml` syntax

### Getting Help

- Check the [troubleshooting section](installation.md#troubleshooting)
- Review error logs in detail
- Use `uv run poe check_all` to identify issues
- Consult the project documentation

## Next Steps

- **[Apple Silicon Optimization](apple-silicon.md)** - Optimize for M3 Ultra
- **[Study Guide](../study-guide/index.md)** - Begin learning with Week 1
- **[Materials](../materials/math/index.md)** - Dive into mathematical foundations
