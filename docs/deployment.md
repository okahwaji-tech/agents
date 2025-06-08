# Documentation Deployment Guide

This guide explains how to deploy the LLM & Agentic Workflows documentation to GitHub Pages.

## 🚀 Quick Deployment

### Prerequisites

1. **GitHub repository** with documentation source
2. **GitHub Pages enabled** in repository settings
3. **Local development environment** set up

### Deploy to GitHub Pages

```bash
# Build and deploy documentation
uv run mkdocs gh-deploy

# Or using the poe task (if poethepoet is installed)
uv run poe docs-deploy
```

This command will:
- Build the documentation
- Create/update the `gh-pages` branch
- Push the built site to GitHub Pages
- Make it available at `https://okahwaji-tech.github.io/agents/`

## 🛠️ Local Development

### Serve Documentation Locally

```bash
# Start development server
uv run mkdocs serve

# Or using the poe task
uv run poe docs-serve
```

The documentation will be available at `http://localhost:8000` with auto-reload on changes.

### Build Documentation

```bash
# Build static site
uv run mkdocs build

# Or using the poe task
uv run poe docs-build
```

Built files will be in the `site/` directory.

## 📁 Project Structure

```
agents/
├── docs/                           # Documentation source
│   ├── index.md                   # Homepage
│   ├── getting-started/           # Installation and setup
│   ├── study-guide/              # Learning curriculum
│   ├── materials/                # Educational materials
│   ├── code-examples/            # Code implementations
│   ├── progress/                 # Progress tracking
│   ├── resources/                # Additional resources
│   ├── stylesheets/              # Custom CSS
│   ├── javascripts/              # Custom JavaScript
│   └── includes/                 # Reusable content
├── mkdocs.yml                     # MkDocs configuration
├── pyproject.toml                # Project dependencies
└── README.md                     # Project overview
```

## ⚙️ Configuration

### MkDocs Configuration (`mkdocs.yml`)

Key configuration sections:

```yaml
site_name: Large Language Models Learning Hub
site_url: https://okahwaji-tech.github.io/agents/

theme:
  name: material
  palette:
    # Light/dark mode toggle
    - scheme: default
      primary: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocs-jupyter          # Jupyter notebook support
  - git-revision-date-localized  # Last updated dates
  - awesome-pages           # Flexible navigation

markdown_extensions:
  - pymdownx.arithmatex     # Math support
  - pymdownx.highlight      # Code highlighting
  - pymdownx.superfences    # Code blocks
  - pymdownx.tabbed         # Tabbed content
  - admonition              # Callout boxes
  - attr_list               # Attribute lists
  - md_in_html              # Markdown in HTML
```

### Custom Styling

Custom CSS in `docs/stylesheets/extra.css`:
- Progress tracking tables
- Healthcare-specific styling
- Apple Silicon optimization callouts
- Mathematical content formatting
- Responsive design improvements

### JavaScript Enhancements

Custom JavaScript in `docs/javascripts/mathjax.js`:
- MathJax configuration for mathematical notation
- Progress tracking functionality
- Code copy buttons
- Apple Silicon performance monitoring

## 🎨 Customization

### Adding New Pages

1. **Create markdown file** in appropriate directory
2. **Add to navigation** in `mkdocs.yml`
3. **Link from other pages** as needed

Example:
```yaml
nav:
  - Home: index.md
  - New Section:
    - Overview: new-section/index.md
    - Details: new-section/details.md
```

### Custom Components

#### Progress Tables
```markdown
| Task | Status | Notes |
|------|--------|-------|
| Learn Probability | ⏳ Pending | [Materials](link) |
| Implement LLM | ✅ Complete | [Code](link) |
```

#### Medical Disclaimers
```markdown
!!! warning "Medical Disclaimer"
    This is for educational purposes only.
    Always consult healthcare professionals.
```

#### Apple Silicon Callouts
```markdown
!!! tip "Apple Silicon Optimization"
    This code is optimized for M1/M2/M3 processors.
```

### Mathematical Notation

Use MathJax for mathematical expressions:

```markdown
Inline math: $P(w|context) = \frac{P(context|w) \cdot P(w)}{P(context)}$

Block math:
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Build Failures

```bash
# Check for syntax errors
uv run mkdocs build --strict

# Validate configuration
uv run mkdocs config
```

#### 2. Missing Dependencies

```bash
# Reinstall documentation dependencies
uv sync --extra docs

# Check installed packages
uv pip list | grep mkdocs
```

#### 3. Navigation Issues

- Ensure all referenced files exist
- Check file paths in `mkdocs.yml`
- Verify markdown link syntax

#### 4. GitHub Pages Not Updating

- Check GitHub Actions logs
- Verify `gh-pages` branch exists
- Confirm GitHub Pages source is set to `gh-pages` branch

### Debug Mode

```bash
# Serve with verbose output
uv run mkdocs serve --verbose

# Build with strict mode
uv run mkdocs build --strict --verbose
```

## 📈 Performance Optimization

### Build Performance

- Use `--dirty` flag for faster rebuilds during development
- Optimize image sizes and formats
- Minimize custom JavaScript and CSS

### Site Performance

- Enable compression in hosting
- Use CDN for static assets
- Optimize images with appropriate formats
- Minimize JavaScript execution

## 🔒 Security Considerations

### Content Security

- Never include real patient data
- Use synthetic data for examples
- Include appropriate disclaimers
- Follow HIPAA guidelines for educational content

### Deployment Security

- Use HTTPS for all links
- Validate external dependencies
- Keep dependencies updated
- Monitor for security vulnerabilities

## 📊 Analytics and Monitoring

### Google Analytics

Add to `mkdocs.yml`:
```yaml
extra:
  analytics:
    provider: google
    property: GA_MEASUREMENT_ID
```

### Performance Monitoring

- Monitor page load times
- Track user engagement
- Analyze search queries
- Monitor error rates

## 🚀 Advanced Features

### Custom Plugins

Create custom MkDocs plugins for:
- Progress tracking automation
- Code example validation
- Healthcare compliance checking
- Apple Silicon optimization hints

### Integration with CI/CD

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --extra docs
      - name: Deploy docs
        run: uv run mkdocs gh-deploy --force
```

## 📝 Content Guidelines

### Writing Style

- Use clear, concise language
- Include practical examples
- Provide step-by-step instructions
- Add appropriate warnings and disclaimers

### Code Examples

- Include complete, runnable examples
- Add comments explaining key concepts
- Optimize for Apple Silicon when relevant
- Include healthcare safety considerations

### Healthcare Content

- Always include medical disclaimers
- Use synthetic or anonymized data
- Follow regulatory guidelines
- Emphasize educational purpose

---

**Need help with deployment?** Check the [troubleshooting section](#troubleshooting) or open an issue in the GitHub repository.
