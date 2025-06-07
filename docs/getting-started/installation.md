# Installation

This guide will help you set up your development environment for the LLM & Agentic Workflows learning repository.

## Prerequisites

- **Python 3.11+** (Python 3.12 recommended)
- **Apple Silicon Mac** (M1/M2/M3) for optimal performance
- **Git** for version control
- **16GB+ RAM** recommended for running larger models

## Step 1: Install uv Package Manager

We use `uv` as our Python package manager for fast dependency resolution and virtual environment management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to your PATH (add this to your shell profile)
source $HOME/.local/bin/env
```

Verify the installation:

```bash
uv --version
```

## Step 2: Clone the Repository

```bash
git clone https://github.com/okahwaji-tech/agents.git
cd agents
```

## Step 3: Create Virtual Environment

Create a virtual environment named 'agents' (as per your preference):

```bash
# Create virtual environment
uv venv agents

# Activate the virtual environment
source agents/bin/activate
```

## Step 4: Install Dependencies

Install all project dependencies optimized for Apple Silicon:

```bash
# Install main dependencies
UV_PROJECT_ENVIRONMENT=agents uv sync

# Install development dependencies (optional)
uv sync --extra dev

# Install documentation dependencies (for building docs)
uv sync --extra docs
```

## Step 5: Verify Installation

Run the Apple Silicon optimization test:

```bash
python test_apple_silicon.py
```

This test will verify:

- ✅ PyTorch with MPS (Metal Performance Shaders) support
- ✅ Hugging Face Transformers with Apple Silicon optimizations
- ✅ All data science libraries properly installed
- ✅ GPU acceleration availability

Expected output:
```
✅ PyTorch MPS available: True
✅ Transformers library: 4.52.4+
✅ Apple Silicon optimizations: Enabled
✅ Memory optimization: Active
```

## Step 6: Development Tools Setup

### Code Quality Tools

Run automated checks and formatting:

```bash
# Format code
uv run poe format

# Lint code
uv run poe lint

# Type checking
uv run poe typecheck

# Run tests
uv run poe test

# Run all checks
uv run poe check_all
```

### Jupyter Lab

Start Jupyter Lab for interactive development:

```bash
uv run jupyter lab
```

### Documentation Server

Serve the documentation locally:

```bash
uv run poe docs-serve
```

The documentation will be available at `http://localhost:8000`

## Troubleshooting

### Common Issues

#### 1. MPS Not Available

If PyTorch MPS is not available:

```bash
# Check macOS version (requires macOS 12.3+)
sw_vers

# Reinstall PyTorch with MPS support
uv pip install --force-reinstall torch torchvision torchaudio
```

#### 2. Memory Issues

For memory optimization on Apple Silicon:

```bash
# Set environment variables for optimal memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 3. Package Conflicts

If you encounter dependency conflicts:

```bash
# Clean cache and reinstall
uv run poe clean_cache
uv cache clean
uv sync --reinstall
```

#### 4. Virtual Environment Issues

If the virtual environment is corrupted:

```bash
# Remove and recreate
rm -rf agents
uv venv agents
source agents/bin/activate
uv sync
```

## Environment Variables

Create a `.env` file in the project root for API keys and configuration:

```bash
# OpenAI API (optional, for advanced examples)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Token (optional, for private models)
HUGGINGFACE_TOKEN=your_hf_token_here

# Apple Silicon optimizations
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Performance Optimization

### Apple Silicon M3 Ultra Specific

For optimal performance on M3 Ultra:

```python
import torch
from accelerate import Accelerator

# Automatic device selection
accelerator = Accelerator()
device = accelerator.device  # Will use 'mps' on Apple Silicon

# Manual device selection
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Memory Management

```python
# Enable memory efficient attention
import torch.nn.functional as F
torch.backends.mps.enable_memory_efficient_attention = True

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

## Next Steps

Once installation is complete:

1. **[Development Setup](development-setup.md)** - Configure your development workflow
2. **[Apple Silicon Optimization](apple-silicon.md)** - Learn about M3 Ultra specific optimizations
3. **[Study Guide](../study-guide/index.md)** - Begin your learning journey with Week 1

## Verification Checklist

- [ ] Python 3.11+ installed
- [ ] uv package manager installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] Apple Silicon test passes
- [ ] Jupyter Lab starts without errors
- [ ] Documentation server runs locally
- [ ] Code quality tools work

If all items are checked, you're ready to begin learning! 🚀
