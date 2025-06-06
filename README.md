# Agents Study Plan

This repository contains a detailed study guide for learning large language models (LLMs) and building agentic workflows. The full curriculum is stored in the `docs/` directory and rendered as a website using [MkDocs](https://www.mkdocs.org/).

## Getting Started

1. **Install dependencies** using `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv agents
   source agents/bin/activate
   UV_PROJECT_ENVIRONMENT=agents uv sync
   ```

2. **Run automated checks**:
   ```bash
   uv run poe check_all
   uv run poe clean_cache
   ```

## Documentation Site

Browse the study guide as a web page:

1. Install MkDocs:
   ```bash
   pip install mkdocs
   ```
2. Serve the site locally:
   ```bash
   mkdocs serve
   ```
   Open <http://127.0.0.1:8000> in your browser.
3. Build the static site (optional):
   ```bash
   mkdocs build
   ```

The site content lives in `docs/`. The original README has been moved there as `docs/index.md`.

## License

This project is licensed under the MIT License.
