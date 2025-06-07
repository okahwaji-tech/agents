# 💻 Code Examples

!!! info "🎯 Learning Objectives"
    Practical implementations demonstrating the mathematical concepts and healthcare applications covered in the study materials.

    - **Mathematical Foundations**: PyTorch implementations of probability, linear algebra, and optimization
    - **LLM Applications**: Healthcare-focused language model examples
    - **Reinforcement Learning**: MDP implementations and RL algorithms
    - **Apple Silicon Optimization**: M3 Ultra processor optimizations

## 🌟 Overview

The code examples provide hands-on implementations of the theoretical concepts covered in the learning materials, with a focus on healthcare applications and Apple Silicon optimization.

!!! example "What You'll Find Here"
    - **Mathematical foundations** implemented in PyTorch with healthcare examples
    - **Language model applications** for medical text processing
    - **Reinforcement learning algorithms** for healthcare decision-making
    - **Evaluation metrics** and analysis tools
    - **Apple Silicon optimizations** for M3 Ultra processors

## 📁 Actual Code Structure

```
code/week-1/
├── 📊 Mathematical Foundations
│   ├── discrete_distributions_healthcare.py
│   ├── continuous_distributions_healthcare.py
│   ├── svd_pytorch_examples.py
│   └── gradient_descent_pytorch_examples.py
├── 🤖 LLM Applications
│   ├── autoregressive_lm_healthcare.py
│   └── llm_evaluation_metrics.py
└── 🎯 Reinforcement Learning
    ├── base_mdp.py
    ├── value_iteration.py
    ├── policy_iteration.py
    ├── q_learning.py
    ├── sarsa.py
    ├── healthcare_examples.py
    └── comprehensive_demo.py
```

## 🧮 Mathematical Foundations

### Discrete Probability Distributions for Healthcare

Explore discrete probability distributions with medical applications using PyTorch.

!!! example "Healthcare Applications"
    - **Binomial distributions** for treatment success rates
    - **Poisson distributions** for patient arrival rates
    - **Categorical distributions** for symptom classification
    - **Bayesian inference** for medical diagnosis

```python title="discrete_distributions_healthcare.py"
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

class HealthcareDiscreteDistributions:
    """
    Discrete probability distributions for healthcare applications.
    Demonstrates fundamental concepts used in LLMs and medical AI.
    """

    def __init__(self, device: str = 'mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def treatment_success_analysis(self, success_rate: float, num_patients: int):
        """
        Model treatment success using binomial distribution.

        This is fundamental to understanding how LLMs model
        discrete outcomes in medical contexts.
        """
        # Create binomial distribution
        binomial = dist.Binomial(num_patients, success_rate)

        # Sample possible outcomes
        samples = binomial.sample((1000,))

        return {
            'mean_successes': binomial.mean.item(),
            'variance': binomial.variance.item(),
            'samples': samples
        }
```

**Key Features:**
- Binomial distributions for treatment outcomes
- Poisson distributions for event modeling
- Categorical distributions for classification
- Healthcare-specific probability calculations

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/discrete_distributions_healthcare.py)

### Continuous Probability Distributions

Advanced continuous distributions for medical parameter modeling.

```python title="continuous_distributions_healthcare.py"
import torch
import torch.distributions as dist
from typing import Dict, List, Tuple

class HealthcareContinuousDistributions:
    """
    Continuous probability distributions for healthcare modeling.
    Essential for understanding LLM probability calculations.
    """

    def blood_pressure_modeling(self, systolic_mean: float = 120, systolic_std: float = 15):
        """
        Model blood pressure using normal distribution.

        Demonstrates continuous probability concepts used in
        neural network outputs and LLM token probabilities.
        """
        normal_dist = dist.Normal(systolic_mean, systolic_std)

        # Calculate probabilities for different ranges
        prob_normal = self._calculate_range_probability(normal_dist, 90, 140)
        prob_high = self._calculate_range_probability(normal_dist, 140, 200)

        return {
            'normal_range_prob': prob_normal,
            'high_range_prob': prob_high,
            'distribution': normal_dist
        }
```

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/continuous_distributions_healthcare.py)

### Singular Value Decomposition (SVD) for LLMs

Demonstrate SVD operations fundamental to transformer architectures and dimensionality reduction.

!!! note "Why SVD Matters for LLMs"
    - **Attention mechanisms** use matrix decompositions
    - **Dimensionality reduction** for efficient embeddings
    - **Principal Component Analysis** for data analysis
    - **Low-rank approximations** for model compression

```python title="svd_pytorch_examples.py"
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class SVDHealthcareAnalyzer:
    """
    SVD applications in healthcare and LLM contexts.
    Demonstrates matrix decomposition techniques used in transformers.
    """

    def __init__(self, device: str = 'mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')

    def medical_data_compression(self, patient_data: torch.Tensor, rank: int = 10):
        """
        Compress patient data using SVD for efficient storage and analysis.

        This technique is used in transformer attention mechanisms
        and embedding compression.
        """
        # Perform SVD decomposition
        U, S, V = torch.svd(patient_data)

        # Low-rank approximation
        compressed = U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T

        # Calculate compression ratio
        original_size = patient_data.numel()
        compressed_size = rank * (U.shape[0] + V.shape[0] + 1)
        compression_ratio = original_size / compressed_size

        return {
            'compressed_data': compressed,
            'compression_ratio': compression_ratio,
            'singular_values': S,
            'reconstruction_error': torch.norm(patient_data - compressed).item()
        }
```

**Key Features:**
- SVD for medical data compression
- Low-rank matrix approximations
- Attention mechanism foundations
- Healthcare data analysis applications

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/svd_pytorch_examples.py)

### Gradient Descent Optimization

PyTorch implementations of optimization algorithms used in LLM training.

```python title="gradient_descent_pytorch_examples.py"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class HealthcareOptimization:
    """
    Gradient descent optimization for healthcare ML models.
    Demonstrates optimization techniques used in LLM training.
    """

    def medical_prediction_training(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Train a medical prediction model using various optimizers.

        Shows the same optimization principles used in LLM training.
        """
        model = nn.Sequential(
            nn.Linear(features.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Compare different optimizers
        optimizers = {
            'SGD': optim.SGD(model.parameters(), lr=0.01),
            'Adam': optim.Adam(model.parameters(), lr=0.001),
            'AdamW': optim.AdamW(model.parameters(), lr=0.001)
        }

        return self._train_with_optimizers(model, features, targets, optimizers)
```

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/gradient_descent_pytorch_examples.py)

## 🤖 LLM Applications

### Autoregressive Language Model for Healthcare

Complete implementation of an autoregressive language model with healthcare applications.

!!! example "What You'll Learn"
    - **Autoregressive generation** - How LLMs predict next tokens
    - **Healthcare text processing** - Medical domain applications
    - **Apple Silicon optimization** - M3 Ultra performance tuning
    - **Safety considerations** - Medical AI ethics and disclaimers

```python title="autoregressive_lm_healthcare.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Tuple, Optional
import logging

class HealthcareAutoregressiveLM:
    """
    Autoregressive Language Model for Healthcare Applications.

    Demonstrates the fundamental architecture behind modern LLMs
    with specific focus on medical text generation and safety.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        self.model, self.tokenizer = self._load_model(model_name)

        # Medical safety disclaimer
        self.medical_disclaimer = (
            "⚠️ MEDICAL DISCLAIMER: This is for educational purposes only. "
            "Always consult qualified healthcare professionals for medical advice."
        )

        # Medical keywords for content detection
        self.medical_keywords = {
            'diagnosis', 'treatment', 'medication', 'symptoms', 'disease',
            'patient', 'clinical', 'medical', 'health', 'therapy', 'prescription',
            'surgery', 'hospital', 'doctor', 'nurse', 'pharmaceutical'
        }

    def autoregressive_generation(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Dict[str, any]:
        """
        Demonstrate step-by-step autoregressive generation.

        Shows exactly how LLMs generate text token by token.
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Track generation process
        generation_log = []
        current_ids = input_ids.clone()

        self.model.eval()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model predictions for next token
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last token predictions

                # Apply temperature scaling
                logits = logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample next token
                next_token_id = torch.multinomial(probs, 1)
                next_token = self.tokenizer.decode(next_token_id.item())

                # Log this step
                generation_log.append({
                    'step': step,
                    'token': next_token,
                    'token_id': next_token_id.item(),
                    'probability': probs[next_token_id].item(),
                    'top_5_tokens': self._get_top_tokens(probs, 5)
                })

                # Add to sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)

                # Stop if we hit end token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        # Decode final sequence
        generated_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)

        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'generation_log': generation_log,
            'total_tokens': len(generation_log),
            'is_medical_content': self._is_medical_content(generated_text)
        }
```

**Key Features:**
- Step-by-step autoregressive generation
- Temperature and top-k/top-p sampling
- Medical content detection and safety
- Detailed generation logging and analysis
- Apple Silicon MPS optimization

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/autoregressive_lm_healthcare.py)

### LLM Evaluation Metrics

Comprehensive evaluation framework for language models with healthcare focus.

!!! note "Essential Evaluation Metrics"
    - **Perplexity** - How well the model predicts text
    - **BLEU Score** - Quality of generated text
    - **Medical Safety** - Healthcare-specific safety measures
    - **Bias Detection** - Fairness in medical contexts

```python title="llm_evaluation_metrics.py"
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import re

class HealthcareLLMEvaluator:
    """
    Comprehensive evaluation framework for LLMs in healthcare contexts.

    Implements key metrics used to assess language model performance
    with special attention to medical safety and bias.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model, self.tokenizer = self._load_model(model_name)

        # Medical safety keywords
        self.safety_keywords = {
            'harmful': ['overdose', 'dangerous', 'lethal', 'toxic', 'fatal'],
            'beneficial': ['safe', 'effective', 'approved', 'recommended', 'beneficial'],
            'uncertain': ['experimental', 'investigational', 'off-label', 'unproven']
        }

    def calculate_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate perplexity on a list of texts.

        Perplexity measures how well the model predicts the text.
        Lower perplexity = better language model performance.
        """
        total_log_likelihood = 0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Calculate log likelihood
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                log_likelihood = -outputs.loss.item() * inputs["input_ids"].size(1)

                total_log_likelihood += log_likelihood
                total_tokens += inputs["input_ids"].size(1)

        # Calculate perplexity
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = torch.exp(torch.tensor(-avg_log_likelihood))

        return {
            'perplexity': perplexity.item(),
            'avg_log_likelihood': avg_log_likelihood,
            'total_tokens': total_tokens
        }

    def evaluate_medical_safety(self, generated_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate safety of medical text generation.

        Analyzes generated text for potentially harmful medical advice.
        """
        safety_scores = []

        for text in generated_texts:
            score = self._assess_text_safety(text)
            safety_scores.append(score)

        return {
            'average_safety_score': np.mean(safety_scores),
            'min_safety_score': np.min(safety_scores),
            'max_safety_score': np.max(safety_scores),
            'safety_pass_rate': np.mean([s >= 0.7 for s in safety_scores]),
            'individual_scores': safety_scores
        }
```

**Key Features:**
- Perplexity calculation for model quality assessment
- Medical safety evaluation framework
- Bias detection in healthcare contexts
- Comprehensive reporting and visualization tools

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/llm_evaluation_metrics.py)

## 🎯 Reinforcement Learning for Healthcare

### Base MDP Implementation

Foundation classes for Markov Decision Processes in healthcare contexts.

!!! example "RL in Healthcare"
    - **Treatment planning** - Optimal therapy sequences
    - **Drug dosing** - Personalized medication schedules
    - **Resource allocation** - Hospital bed management
    - **Clinical decision support** - Evidence-based recommendations

```python title="base_mdp.py"
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

class HealthcareMDP(ABC):
    """
    Base class for healthcare Markov Decision Processes.

    Provides the fundamental structure for modeling medical
    decision-making problems as MDPs.
    """

    def __init__(self,
                 states: List[str],
                 actions: List[str],
                 gamma: float = 0.95):
        """
        Initialize healthcare MDP.

        Args:
            states: List of possible patient states
            actions: List of possible medical interventions
            gamma: Discount factor for future rewards
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.n_states = len(states)
        self.n_actions = len(actions)

        # State and action mappings
        self.state_to_idx = {state: idx for idx, state in enumerate(states)}
        self.action_to_idx = {action: idx for idx, action in enumerate(actions)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

        # Initialize transition probabilities and rewards
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

        # Medical safety constraints
        self.safety_constraints = {}
        self.medical_disclaimer = (
            "⚠️ MEDICAL DISCLAIMER: This is for educational purposes only. "
            "Always consult qualified healthcare professionals."
        )

    @abstractmethod
    def define_transition_probabilities(self) -> np.ndarray:
        """Define transition probabilities P(s'|s,a)."""
        pass

    @abstractmethod
    def define_reward_function(self) -> np.ndarray:
        """Define reward function R(s,a)."""
        pass

    def add_safety_constraint(self, state: str, forbidden_actions: List[str]):
        """Add medical safety constraints."""
        if state in self.state_to_idx:
            self.safety_constraints[state] = forbidden_actions
```

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/RL/base_mdp.py)

### Value Iteration Algorithm

Classic dynamic programming solution for MDPs.

```python title="value_iteration.py"
import numpy as np
import torch
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class ValueIteration:
    """
    Value Iteration algorithm for solving healthcare MDPs.

    Implements the classic dynamic programming approach
    to find optimal policies for medical decision-making.
    """

    def __init__(self, mdp, tolerance: float = 1e-6, max_iterations: int = 1000):
        self.mdp = mdp
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Initialize value function
        self.V = np.zeros(mdp.n_states)
        self.policy = np.zeros(mdp.n_states, dtype=int)

        # Convergence tracking
        self.convergence_history = []

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve MDP using value iteration.

        Returns:
            V: Optimal value function
            policy: Optimal policy
        """
        for iteration in range(self.max_iterations):
            V_old = self.V.copy()

            # Value iteration update
            for s in range(self.mdp.n_states):
                # Calculate Q-values for all actions
                q_values = []
                for a in range(self.mdp.n_actions):
                    q_value = self.mdp.R[s, a] + self.mdp.gamma * np.sum(
                        self.mdp.P[s, a, :] * self.V
                    )
                    q_values.append(q_value)

                # Update value function
                self.V[s] = max(q_values)
                self.policy[s] = np.argmax(q_values)

            # Check convergence
            delta = np.max(np.abs(self.V - V_old))
            self.convergence_history.append(delta)

            if delta < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

        return self.V, self.policy
```

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/RL/value_iteration.py)

### Q-Learning Implementation

Model-free reinforcement learning for healthcare applications.

```python title="q_learning.py"
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class HealthcareQLearning:
    """
    Q-Learning implementation for healthcare decision-making.

    Demonstrates model-free RL that can learn optimal policies
    without knowing the environment dynamics.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 gamma: float = 0.95):

        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []

    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy policy.

        Balances exploration vs exploitation in medical decision-making.
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.Q[state, :])

    def update_q_value(self,
                      state: int,
                      action: int,
                      reward: float,
                      next_state: int):
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        best_next_action = np.argmax(self.Q[next_state, :])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]

        self.Q[state, action] += self.lr * td_error
```

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/RL/q_learning.py)

### Healthcare Examples

Real-world medical decision-making scenarios using RL.

```python title="healthcare_examples.py"
import numpy as np
from typing import Dict, List, Tuple
from base_mdp import HealthcareMDP

class DrugDosingMDP(HealthcareMDP):
    """
    MDP for optimal drug dosing decisions.

    Models the problem of determining appropriate medication
    dosages based on patient response and side effects.
    """

    def __init__(self):
        # Define states: patient condition levels
        states = [
            "critical_low", "low", "normal", "high", "critical_high"
        ]

        # Define actions: dosage adjustments
        actions = [
            "decrease_large", "decrease_small", "maintain",
            "increase_small", "increase_large"
        ]

        super().__init__(states, actions, gamma=0.9)

        # Define medical constraints
        self.add_safety_constraint("critical_low", ["decrease_large", "decrease_small"])
        self.add_safety_constraint("critical_high", ["increase_large", "increase_small"])

        # Initialize transition probabilities and rewards
        self.define_transition_probabilities()
        self.define_reward_function()
```

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/RL/healthcare_examples.py)

## 🚀 Getting Started

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/okahwaji-tech/agents.git
   cd agents
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv agents
   source agents/bin/activate  # On Windows: agents\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run examples**:
   ```bash
   cd code/week-1
   python discrete_distributions_healthcare.py
   python autoregressive_lm_healthcare.py
   python RL/comprehensive_demo.py
   ```

### Apple Silicon Optimization

For M3 Ultra users, ensure MPS is properly configured:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## 📚 Next Steps

!!! tip "Continue Learning"
    - Explore the [Mathematical Foundations](../materials/math/index.md) for deeper theory
    - Check out [LLM Fundamentals](../materials/llm/index.md) for advanced concepts
    - Review [Study Guide Week 1](../study-guide/week-1/index.md) for structured learning
    - Follow the [Study Guide](../study-guide/index.md) for structured learning

## 🔗 Additional Resources

- **GitHub Repository**: [okahwaji-tech/agents](https://github.com/okahwaji-tech/agents)
- **Documentation**: [Full Documentation](../materials/index.md)
- **Community**: [Discussions and Issues](https://github.com/okahwaji-tech/agents/discussions)
            torch.mps.empty_cache()
        
        import gc
        gc.collect()
        
        logger.info("Memory cleanup completed")
    
    @staticmethod
    def monitor_performance() -> Dict[str, float]:
        """Monitor system performance metrics."""
        return {
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'mps_available': torch.backends.mps.is_available()
        }
```

**Key Features:**
- MPS environment configuration
- Memory management utilities
- Performance monitoring tools
- Apple Silicon specific optimizations

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/apple_silicon_utils.py)

## 🚀 Getting Started

### Running the Examples

1. **Set up your environment:**
   ```bash
   source agents/bin/activate
   cd code/week-1
   ```

2. **Run the mathematical foundations:**
   ```bash
   python mathematical_foundations/probability_distributions.py
   python mathematical_foundations/information_theory.py
   ```

3. **Try your first LLM program:**
   ```bash
   python llm_applications/first_llm_program.py
   ```

4. **Analyze healthcare applications:**
   ```bash
   python healthcare_examples/medical_text_analysis.py
   ```

### Example Notebooks

Interactive Jupyter notebooks are available for each topic:

- `week1_mathematical_foundations.ipynb`
- `week1_first_llm_program.ipynb`
- `week1_healthcare_analysis.ipynb`
- `week1_evaluation_metrics.ipynb`

## 📚 Learning Path

### Recommended Order

1. **Start with mathematical foundations** to understand the theory
2. **Implement your first LLM program** to see concepts in action
3. **Explore healthcare applications** to understand domain-specific challenges
4. **Run evaluation metrics** to measure and understand model performance

### Key Learning Outcomes

After working through these examples, you'll understand:

- ✅ How probability theory applies to language modeling
- ✅ Why cross-entropy is used as the loss function
- ✅ How vector operations work in embedding spaces
- ✅ How to implement and evaluate LLMs on Apple Silicon
- ✅ Healthcare-specific considerations for AI safety

## 🔗 Related Resources

- **[Week 1 Study Guide](../study-guide/week-1/index.md)** - Theoretical foundations
- **[Mathematical Materials](../materials/math/index.md)** - Detailed explanations
- **[Learning Materials](../materials/index.md)** - Comprehensive study resources

---

**Ready to code?** Start with the [mathematical foundations](https://github.com/okahwaji-tech/agents/tree/main/code/week-1/mathematical_foundations) and work your way through each example systematically!
