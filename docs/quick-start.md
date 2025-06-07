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
    **Start Here**: [Week 1 - Introduction to LLMs](study-guide/week-1/index.md)

    - 📖 Begin with mathematical foundations
    - 🧮 Review probability theory and linear algebra
    - 💻 Set up your first LLM implementation
    - ⏱️ Estimated time: 15-19 hours

=== "Some ML Background"
    **Start Here**: [Mathematical Foundations](materials/math/index.md)

    - 🔍 Dive into probability theory and linear algebra
    - 🏗️ Build understanding from mathematical foundations
    - 🧮 Explore information theory concepts
    - ⏱️ Estimated time: 15-19 hours

=== "LLM Experience"
    **Start Here**: [LLM Fundamentals](materials/llm/index.md)

    - 🚀 Advanced LLM concepts
    - 🤖 Word embeddings and evaluation methods
    - 🧠 Deep dive into LLM architecture
    - ⏱️ Estimated time: 15-19 hours

=== "Want to Build Agents"
    **Start Here**: [Machine Learning Foundations](materials/ml/index.md)

    - 🤖 Reinforcement learning fundamentals
    - 🛠️ Markov Decision Processes
    - 🔗 RL-LLM connections
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
- [Code Examples](code-examples/index.md) - Practical implementations
- [Learning Materials](materials/index.md) - Comprehensive study materials
- [Resources](resources/index.md) - Additional learning resources

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

## 📊 Learning Organization

### Study Structure
1. **Weekly Objectives** - Clear learning goals for each week
2. **Study Materials** - Organized by mathematical foundations, LLM concepts, and healthcare applications
3. **Code Examples** - Hands-on implementations for each concept
4. **Self-Assessment** - Checklists to verify understanding

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
- Share learning insights and discoveries
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
- [ ] 📊 Study materials organized
- [ ] ⏱️ Study schedule planned
- [ ] 🏥 Healthcare focus areas identified
- [ ] 🤝 Community resources bookmarked

## 🎯 Next Steps

1. **[Complete Environment Setup](getting-started/installation.md)** - Detailed installation guide
2. **[Apple Silicon Optimization](getting-started/apple-silicon.md)** - M3 Ultra specific setup
3. **[Begin Week 1](study-guide/week-1/index.md)** - Start your learning journey
4. **[Browse Resources](resources/index.md)** - Explore additional learning materials

---

<div class="quick-start-cta">
  <h3>🚀 Ready to Begin?</h3>
  <p>You're all set! Choose your starting point and begin your journey to LLM mastery.</p>
  <div class="cta-buttons">
    <a href="study-guide/week-1/index.md" class="cta-primary">Start Week 1 →</a>
    <a href="roadmap.md" class="cta-secondary">View Full Roadmap</a>
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
