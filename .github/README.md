# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating various tasks in the LLM Learning Guide repository.

## 📄 Workflows

### `docs.yml` - Documentation Deployment

**Purpose**: Automatically builds and deploys the MkDocs documentation to GitHub Pages.

**Triggers**:
- ✅ **Push to main branch**: Builds and deploys documentation
- ✅ **Pull requests**: Builds documentation for validation (no deployment)
- ✅ **Manual trigger**: Can be run manually from GitHub Actions tab

**Features**:
- 🚀 **Fast builds** using `uv` for dependency management
- 🔍 **Strict building** with `--strict` flag to catch errors
- 📝 **Full git history** for git-revision-date-localized plugin
- 🔄 **PR validation** builds docs on pull requests without deploying
- 💬 **PR comments** with build status
- 🛡️ **Secure deployment** using GitHub's recommended Pages actions

**Build Process**:
1. Checkout repository with full git history
2. Setup Python 3.11 and uv package manager
3. Create virtual environment and install dependencies
4. Build documentation with MkDocs
5. Deploy to GitHub Pages (main branch only)

**Environment Variables**:
- `JUPYTER_PLATFORM_DIRS=1`: Suppresses Jupyter platform directory warnings

## 🔧 Setup Requirements

### GitHub Repository Settings

1. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Set Source to "GitHub Actions"
   - Save settings

2. **Branch Protection** (Optional but recommended):
   - Protect main branch
   - Require status checks to pass before merging
   - Include "build" check from the docs workflow

### Local Development

The workflow uses the same dependencies as local development:
- Python 3.11+
- uv package manager
- Dependencies defined in `pyproject.toml` and `uv.lock`

## 🚀 Usage

### Automatic Deployment
- Push changes to main branch
- GitHub Actions will automatically build and deploy
- Check the Actions tab for build status and logs

### Manual Deployment
- Go to Actions tab in GitHub repository
- Select "Deploy Documentation" workflow
- Click "Run workflow" button
- Choose branch and click "Run workflow"

### Pull Request Validation
- Create a pull request
- GitHub Actions will build documentation for validation
- Check for build errors before merging
- PR will receive a comment with build status

## 🔍 Monitoring

### Build Status
- Check the Actions tab for workflow runs
- Green checkmark = successful build and deployment
- Red X = build failed (check logs for details)

### Deployment URL
- Documentation is deployed to: `https://okahwaji-tech.github.io/llm-learning-guide/`
- URL is shown in the deployment job output

### Troubleshooting
- Check workflow logs in the Actions tab
- Common issues:
  - Missing dependencies in `pyproject.toml`
  - Broken internal links (caught by `--strict` flag)
  - Invalid markdown syntax
  - Missing files referenced in navigation

## 📋 Maintenance

### Updating Dependencies
- Update `pyproject.toml` and run `uv lock` locally
- Commit the updated `uv.lock` file
- GitHub Actions will use the locked dependencies

### Workflow Updates
- Modify `.github/workflows/docs.yml` as needed
- Test changes with pull requests before merging
- Monitor the Actions tab for any issues

### Security
- Workflow uses official GitHub Actions with pinned versions
- No secrets required for public repository documentation
- Uses GitHub's recommended Pages deployment action
# GitHub Actions workflow is now active
