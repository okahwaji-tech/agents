# Week 1 Code Examples

This page showcases the practical implementations for Week 1, demonstrating the mathematical concepts and healthcare applications covered in the study materials.

## 🎯 Overview

The Week 1 code examples focus on:

- **Mathematical foundations** implemented in PyTorch
- **First LLM applications** with healthcare focus
- **Apple Silicon optimization** for M3 Ultra processors
- **Evaluation metrics** and analysis tools

## 📁 Code Structure

```
code/week-1/
├── mathematical_foundations/
│   ├── probability_distributions.py
│   ├── information_theory.py
│   ├── linear_algebra_examples.py
│   └── rl_foundations.py
├── llm_applications/
│   ├── first_llm_program.py
│   ├── healthcare_text_generation.py
│   └── model_evaluation.py
├── healthcare_examples/
│   ├── medical_text_analysis.py
│   ├── clinical_note_processing.py
│   └── safety_evaluation.py
└── utils/
    ├── apple_silicon_utils.py
    ├── visualization_tools.py
    └── evaluation_metrics.py
```

## 🧮 Mathematical Foundations

### Probability Distributions for Healthcare

Explore discrete and continuous probability distributions with medical applications.

```python title="probability_distributions.py"
import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class MedicalProbabilityAnalyzer:
    """Analyze probability distributions in medical contexts."""
    
    def __init__(self, device: str = 'mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
    
    def calculate_conditional_probability(
        self, 
        symptoms: List[str], 
        disease: str,
        prior_data: dict
    ) -> float:
        """
        Calculate P(disease|symptoms) using Bayes' theorem.
        
        Example of how LLMs learn conditional relationships.
        """
        # Implementation here
        pass
    
    def language_model_probability(
        self, 
        sequence: List[str]
    ) -> float:
        """
        Calculate probability of a text sequence using chain rule.
        
        P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * ... * P(wn|w1...wn-1)
        """
        # Implementation here
        pass
```

**Key Features:**
- Bayes' theorem for medical diagnosis
- Chain rule probability for text sequences
- Visualization of probability distributions
- Healthcare-specific examples

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/probability_distributions.py)

### Information Theory Metrics

Implement entropy, cross-entropy, and perplexity calculations.

```python title="information_theory.py"
import torch
import torch.nn.functional as F
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class InformationTheoryMetrics:
    """Calculate information theory metrics for LLM evaluation."""
    
    def calculate_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy: H(X) = -∑ P(x) log P(x)
        
        Lower entropy = more predictable = better for specific tasks
        Higher entropy = more uncertain = better for creative tasks
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        return -torch.sum(probabilities * torch.log(probabilities + eps))
    
    def calculate_cross_entropy(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate cross-entropy loss used in LLM training.
        
        This is the actual loss function that trains language models!
        """
        return F.cross_entropy(predictions, targets)
    
    def calculate_perplexity(self, cross_entropy: torch.Tensor) -> torch.Tensor:
        """
        Calculate perplexity: 2^(cross-entropy)
        
        Lower perplexity = better language model performance
        """
        return torch.pow(2, cross_entropy)
```

**Key Features:**
- Entropy calculation for uncertainty measurement
- Cross-entropy loss implementation
- Perplexity computation for model evaluation
- Medical text analysis examples

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/information_theory.py)

### Linear Algebra for Embeddings

Demonstrate vector operations fundamental to LLMs.

```python title="linear_algebra_examples.py"
import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class EmbeddingAnalyzer:
    """Analyze vector embeddings and similarity measures."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def cosine_similarity(
        self, 
        vector1: torch.Tensor, 
        vector2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate cosine similarity between two vectors.
        
        Most common similarity measure in NLP/LLMs.
        """
        return F.cosine_similarity(vector1, vector2, dim=-1)
    
    def medical_concept_similarity(
        self, 
        concept1: str, 
        concept2: str,
        embeddings: Dict[str, torch.Tensor]
    ) -> float:
        """
        Calculate similarity between medical concepts.
        
        Example: similarity between "diabetes" and "insulin"
        """
        if concept1 not in embeddings or concept2 not in embeddings:
            raise ValueError(f"Concepts not found in embeddings")
        
        sim = self.cosine_similarity(
            embeddings[concept1], 
            embeddings[concept2]
        )
        return sim.item()
```

**Key Features:**
- Cosine similarity for semantic relationships
- Medical concept similarity analysis
- Vector arithmetic demonstrations
- Embedding visualization tools

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/linear_algebra_examples.py)

## 🤖 LLM Applications

### First LLM Program

Your first complete LLM application optimized for Apple Silicon.

```python title="first_llm_program.py"
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class HealthcareLLMGenerator:
    """First LLM application with healthcare focus."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model, self.tokenizer = self._load_model(model_name)
        
        # Medical disclaimer
        self.medical_disclaimer = (
            "MEDICAL DISCLAIMER: This is for educational purposes only. "
            "Always consult qualified healthcare professionals for medical advice."
        )
    
    def _load_model(self, model_name: str) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
        """Load GPT-2 model optimized for Apple Silicon."""
        logger.info(f"Loading model: {model_name}")
        
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token
        tokenizer.pad_token = tokenizer.eos_token
        
        # Move to MPS for Apple Silicon acceleration
        model = model.to(self.device)
        
        logger.info(f"Model loaded on device: {self.device}")
        return model, tokenizer
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        add_disclaimer: bool = True
    ) -> str:
        """
        Generate text with specified parameters.
        
        Args:
            prompt: Input text to continue
            max_length: Maximum tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = random)
            top_k: Only consider top k tokens for sampling
            add_disclaimer: Add medical disclaimer for healthcare content
        """
        # Add medical disclaimer for healthcare-related prompts
        if add_disclaimer and self._is_medical_content(prompt):
            prompt = f"{self.medical_disclaimer}\n\n{prompt}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def _is_medical_content(self, text: str) -> bool:
        """Check if text contains medical content."""
        medical_keywords = [
            'diagnosis', 'treatment', 'medication', 'symptoms', 'disease',
            'patient', 'clinical', 'medical', 'health', 'therapy'
        ]
        return any(keyword in text.lower() for keyword in medical_keywords)
```

**Key Features:**
- Apple Silicon MPS optimization
- Healthcare safety disclaimers
- Temperature and top-k sampling
- Medical content detection
- Comprehensive error handling

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/first_llm_program.py)

### Healthcare Text Analysis

Analyze how LLMs process medical text differently from general text.

```python title="healthcare_text_analysis.py"
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class MedicalTextAnalyzer:
    """Analyze LLM behavior on medical vs. general text."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def analyze_confidence_patterns(
        self, 
        medical_texts: List[str], 
        general_texts: List[str]
    ) -> Dict[str, float]:
        """
        Compare model confidence on medical vs. general text.
        
        Returns confidence metrics for both text types.
        """
        medical_confidence = self._calculate_average_confidence(medical_texts)
        general_confidence = self._calculate_average_confidence(general_texts)
        
        return {
            'medical_confidence': medical_confidence,
            'general_confidence': general_confidence,
            'confidence_ratio': medical_confidence / general_confidence
        }
    
    def extract_medical_entities(self, text: str) -> List[str]:
        """
        Extract potential medical entities from text.
        
        Simple implementation for educational purposes.
        """
        # This is a simplified version - real medical NER is much more complex
        medical_terms = []
        
        # Basic medical term patterns
        medical_keywords = [
            'diabetes', 'hypertension', 'medication', 'treatment',
            'diagnosis', 'symptoms', 'patient', 'clinical'
        ]
        
        words = text.lower().split()
        for word in words:
            if any(keyword in word for keyword in medical_keywords):
                medical_terms.append(word)
        
        return medical_terms
```

**Key Features:**
- Confidence analysis on medical text
- Medical entity extraction
- Comparison with general text processing
- Visualization of model behavior patterns

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/healthcare_text_analysis.py)

## 📊 Evaluation and Metrics

### LLM Evaluation Framework

Comprehensive evaluation metrics for language models.

```python title="model_evaluation.py"
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class LLMEvaluator:
    """Comprehensive evaluation framework for LLMs."""
    
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def calculate_perplexity(
        self, 
        model: torch.nn.Module, 
        tokenizer, 
        texts: List[str]
    ) -> float:
        """
        Calculate perplexity on a list of texts.
        
        Lower perplexity = better language model performance.
        """
        total_loss = 0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Accumulate
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item()
    
    def evaluate_medical_safety(
        self, 
        model, 
        tokenizer, 
        test_prompts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model safety on medical prompts.
        
        Checks for potentially harmful medical advice.
        """
        safety_scores = []
        
        for prompt in test_prompts:
            generated = self._generate_response(model, tokenizer, prompt)
            safety_score = self._assess_medical_safety(generated)
            safety_scores.append(safety_score)
        
        return {
            'average_safety_score': np.mean(safety_scores),
            'min_safety_score': np.min(safety_scores),
            'safety_pass_rate': np.mean([s > 0.7 for s in safety_scores])
        }
```

**Key Features:**
- Perplexity calculation for model quality
- Medical safety evaluation
- Bias detection in healthcare contexts
- Comprehensive reporting tools

[View Full Implementation →](https://github.com/okahwaji-tech/agents/blob/main/code/week-1/model_evaluation.py)

## 🛠️ Utilities and Tools

### Apple Silicon Optimization

Utilities for maximizing performance on M3 Ultra processors.

```python title="apple_silicon_utils.py"
import torch
import psutil
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class AppleSiliconOptimizer:
    """Optimization utilities for Apple Silicon processors."""
    
    @staticmethod
    def setup_mps_environment() -> Dict[str, Any]:
        """Configure optimal MPS environment settings."""
        # Set environment variables for optimal performance
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Enable memory efficient attention
        if hasattr(torch.backends.mps, 'enable_memory_efficient_attention'):
            torch.backends.mps.enable_memory_efficient_attention = True
        
        return {
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built(),
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
        }
    
    @staticmethod
    def cleanup_memory():
        """Clean up MPS and system memory."""
        if torch.backends.mps.is_available():
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

- **[Week 1 Study Guide](../../study-guide/week-1/index.md)** - Theoretical foundations
- **[Mathematical Materials](../../materials/week-1/mathematical-foundations.md)** - Detailed explanations
- **[Progress Tracking](../../progress/index.md)** - Monitor your learning

---

**Ready to code?** Start with the [mathematical foundations](https://github.com/okahwaji-tech/agents/tree/main/code/week-1/mathematical_foundations) and work your way through each example systematically!
