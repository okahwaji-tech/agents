# 🚀 Quick Start Guide

Get up and running with the LLM & Agentic AI learning guide in under 30 minutes!

## ⚡ 5-Minute Setup

### 1. Clone the Repository
```bash
git clone https://github.com/okahwaji-tech/agents.git
cd agents
```

### 2. Install Dependencies (Apple Silicon Optimized)
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv agents

# Activate environment
source agents/bin/activate  # On macOS/Linux
# agents\Scripts\activate     # On Windows

# Install all dependencies
uv pip install -e ".[dev,docs]"
```

### 3. Verify Apple Silicon Optimization
```bash
python test_apple_silicon.py
```

### 4. Start Learning!
```bash
# Serve documentation locally
mkdocs serve

# Open http://localhost:8000 in your browser
```

## 📚 Learning Path Quick Start

### Choose Your Starting Point

=== "Complete Beginner"
    **Start Here**: [Week 1 - Introduction to LLMs](study-guide/week-1/)
    
    - 📖 Begin with mathematical foundations
    - 🧮 Review probability theory and linear algebra
    - 💻 Set up your first LLM implementation
    - ⏱️ Estimated time: 15-19 hours

=== "Some ML Background"
    **Start Here**: [Week 2 - Transformer Architecture](study-guide/week-2/)
    
    - 🔍 Dive into attention mechanisms
    - 🏗️ Build transformers from scratch
    - 🏥 Explore healthcare applications
    - ⏱️ Estimated time: 15-19 hours

=== "LLM Experience"
    **Start Here**: [Week 7 - Advanced Techniques](study-guide/week-7/)
    
    - 🚀 Advanced training methods
    - 🤖 Reinforcement learning from human feedback
    - 🏥 Healthcare-specific fine-tuning
    - ⏱️ Estimated time: 15-19 hours

=== "Want to Build Agents"
    **Start Here**: [Week 13 - Agent Foundations](study-guide/week-13/)
    
    - 🤖 Agent architectures and planning
    - 🛠️ Tool use and API integration
    - 🏥 Medical agent applications
    - ⏱️ Estimated time: 15-19 hours

## 🎯 First Week Goals

{{ study_session_card("Week 1 Kickoff", "15-19 hours", [
    "Understand probability theory fundamentals",
    "Grasp information theory basics", 
    "Implement your first LLM program",
    "Explore healthcare AI considerations"
]) }}

## 🏥 Healthcare Focus Quick Start

If you're specifically interested in healthcare AI:

### Essential Healthcare AI Concepts
1. **Medical Text Processing** - Clinical notes, medical literature
2. **Safety & Compliance** - HIPAA, FDA regulations, bias mitigation
3. **Clinical Decision Support** - Diagnostic assistance, treatment recommendations
4. **Medical Knowledge Integration** - UMLS, medical ontologies

### Healthcare-Specific Resources
- [Healthcare Applications Overview](materials/healthcare/overview.md)
- [Medical AI Safety Guidelines](materials/healthcare/safety-compliance.md)
- [Clinical Decision Support Examples](code-examples/healthcare/)

## 🍎 Apple Silicon Quick Optimization

### Immediate Performance Gains
```python
import torch

# Enable MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Silicon GPU acceleration!")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Optimize memory usage
torch.mps.empty_cache()  # Clear GPU memory
```

### M3 Ultra Specific Optimizations
{{ apple_silicon_tip("The M3 Ultra has 192GB unified memory. Use larger batch sizes and models that fit entirely in memory for optimal performance.") }}

## 📊 Progress Tracking Quick Setup

### Enable Progress Tracking
1. **Automatic Saving** - Your progress is saved locally in browser storage
2. **Study Timer** - Track time spent on each topic
3. **Achievement System** - Unlock badges as you progress
4. **Analytics** - View learning velocity and patterns

### Quick Progress Check
{{ progress_bar(0, "Overall Progress", "primary") }}

## 🔧 Development Environment

### Recommended VS Code Extensions
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff"
  ]
}
```

### Jupyter Lab Setup
```bash
# Start Jupyter Lab with Apple Silicon optimizations
jupyter lab --no-browser --port=8888
```

## 🤝 Community & Support

### Get Help
- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Report bugs or request features
- 📧 **Direct Contact**: omarkahwaji@outlook.com

### Study Groups
- Join weekly study sessions
- Share progress and insights
- Collaborate on projects

## 📋 Study Schedule Templates

### Full-Time Study (40 hours/week)
- **Week 1-6**: Foundation (2 weeks each phase)
- **Week 7-12**: Advanced Techniques (2 weeks each phase)  
- **Week 13-18**: Agents (2 weeks each phase)
- **Week 19-24**: Advanced Architectures (2 weeks each phase)

### Part-Time Study (15 hours/week)
- **Week 1-6**: Foundation (1 week each)
- **Week 7-12**: Advanced Techniques (1 week each)
- **Week 13-18**: Agents (1 week each)  
- **Week 19-24**: Advanced Architectures (1 week each)

### Weekend Study (10 hours/week)
- **Month 1-2**: Foundation
- **Month 3-4**: Advanced Techniques
- **Month 5-6**: Agents
- **Month 7-8**: Advanced Architectures

## ✅ Quick Checklist

Before starting your learning journey:

- [ ] ✅ Environment set up and tested
- [ ] 🍎 Apple Silicon optimization verified
- [ ] 📚 Week 1 materials reviewed
- [ ] 📊 Progress tracking enabled
- [ ] ⏱️ Study schedule planned
- [ ] 🏥 Healthcare focus areas identified
- [ ] 🤝 Community resources bookmarked

## 🎯 Next Steps

1. **[Complete Environment Setup](getting-started/installation.md)** - Detailed installation guide
2. **[Apple Silicon Optimization](getting-started/apple-silicon.md)** - M3 Ultra specific setup
3. **[Begin Week 1](study-guide/week-1/)** - Start your learning journey
4. **[Join Community](resources/community.md)** - Connect with other learners

---

<div class="quick-start-cta">
  <h3>🚀 Ready to Begin?</h3>
  <p>You're all set! Choose your starting point and begin your journey to LLM mastery.</p>
  <div class="cta-buttons">
    <a href="study-guide/week-1/" class="cta-primary">Start Week 1 →</a>
    <a href="roadmap/" class="cta-secondary">View Full Roadmap</a>
  </div>
</div>

<style>
.quick-start-cta {
  background: linear-gradient(135deg, #4caf50, #8bc34a);
  color: white;
  padding: 2rem;
  border-radius: 12px;
  text-align: center;
  margin-top: 2rem;
}

.quick-start-cta h3 {
  margin: 0 0 1rem 0;
  font-size: 1.5rem;
}

.quick-start-cta p {
  margin: 0 0 1.5rem 0;
  opacity: 0.9;
}

.cta-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.cta-primary, .cta-secondary {
  padding: 0.75rem 1.5rem;
  border-radius: 25px;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.3s ease;
}

.cta-primary {
  background: white;
  color: #4caf50;
}

.cta-secondary {
  background: transparent;
  color: white;
  border: 2px solid white;
}

.cta-primary:hover, .cta-secondary:hover {
  transform: translateY(-2px);
  text-decoration: none;
}

@media (max-width: 768px) {
  .cta-buttons {
    flex-direction: column;
    align-items: center;
  }
  
  .cta-primary, .cta-secondary {
    width: 200px;
    text-align: center;
  }
}
</style>
