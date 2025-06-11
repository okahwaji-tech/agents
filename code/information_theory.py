"""Information Theory for Large Language Models.

This module provides comprehensive PyTorch implementations for all information theory
concepts used in large language model development, including information content,
entropy, cross-entropy, KL divergence, mutual information, and perplexity.

The module follows object-oriented design principles with a unified base class
and Google-style docstrings for better maintainability and documentation.

Example:
    Basic usage of the information theory calculators:

    ```python
    from information_theory import InformationTheoryCalculator

    # Initialize calculator with desired base
    calc = InformationTheoryCalculator(base='2')  # bits

    # Calculate various metrics
    probs = torch.tensor([0.5, 0.3, 0.2])
    info_content = calc.information_content(probs)
    entropy = calc.entropy(probs)
    ```

Author: LLM Learning Guide
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any
from collections import Counter
from enum import Enum


class LogarithmBase(Enum):
    """Enumeration for logarithm bases used in information theory calculations.

    Attributes:
        NATURAL: Natural logarithm (base e) - results in nats
        BINARY: Binary logarithm (base 2) - results in bits
        DECIMAL: Decimal logarithm (base 10) - results in dits
    """
    NATURAL = 'e'
    BINARY = '2'
    DECIMAL = '10'


class ReductionType(Enum):
    """Enumeration for tensor reduction types.

    Attributes:
        MEAN: Average across batch dimension
        SUM: Sum across batch dimension
        BATCHMEAN: Mean specifically for batch operations
        NONE: No reduction applied
    """
    MEAN = 'mean'
    SUM = 'sum'
    BATCHMEAN = 'batchmean'
    NONE = 'none'


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_epsilon(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Add small epsilon to avoid numerical issues with log(0).

    Args:
        tensor: Input tensor that may contain zeros
        epsilon: Small value to add for numerical stability

    Returns:
        Tensor with minimum value clamped to epsilon
    """
    return torch.clamp(tensor, min=epsilon)


class BaseInformationTheory(ABC):
    """Abstract base class for information theory calculations.

    This class provides common functionality for all information theory
    calculators including logarithm base handling and numerical stability.

    Attributes:
        base: Logarithm base for calculations
        log_fn: Corresponding PyTorch logarithm function
        epsilon: Small value for numerical stability
    """

    def __init__(self, base: Union[str, LogarithmBase] = LogarithmBase.NATURAL,
                 epsilon: float = 1e-8):
        """Initialize the base information theory calculator.

        Args:
            base: Logarithm base for calculations. Can be 'e', '2', '10' or LogarithmBase enum
            epsilon: Small value added for numerical stability

        Raises:
            ValueError: If base is not one of the supported values
        """
        if isinstance(base, str):
            base = LogarithmBase(base)

        self.base = base
        self.epsilon = epsilon

        # Set appropriate logarithm function
        if base == LogarithmBase.NATURAL:
            self.log_fn = torch.log
        elif base == LogarithmBase.BINARY:
            self.log_fn = torch.log2
        elif base == LogarithmBase.DECIMAL:
            self.log_fn = torch.log10
        else:
            raise ValueError(f"Unsupported base: {base}")

    def _stabilize_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Add epsilon to probabilities for numerical stability.

        Args:
            probabilities: Input probability tensor

        Returns:
            Stabilized probability tensor
        """
        return add_epsilon(probabilities, self.epsilon)

    def _apply_reduction(self, tensor: torch.Tensor,
                        reduction: Union[str, ReductionType]) -> torch.Tensor:
        """Apply reduction operation to tensor.

        Args:
            tensor: Input tensor to reduce
            reduction: Type of reduction to apply

        Returns:
            Reduced tensor

        Raises:
            ValueError: If reduction type is not supported
        """
        if isinstance(reduction, str):
            reduction = ReductionType(reduction)

        if reduction == ReductionType.MEAN or reduction == ReductionType.BATCHMEAN:
            return torch.mean(tensor)
        elif reduction == ReductionType.SUM:
            return torch.sum(tensor)
        elif reduction == ReductionType.NONE:
            return tensor
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")


# Sample healthcare text data for demonstrations
HEALTHCARE_TEXTS = [
    "The patient presents with acute chest pain and shortness of breath.",
    "Blood pressure reading shows 140 over 90 mmHg indicating hypertension.",
    "Laboratory results reveal elevated white blood cell count suggesting infection.",
    "Patient reports chronic fatigue and joint pain lasting several months.",
    "Imaging studies show no evidence of fracture or dislocation.",
    "Medication adherence appears suboptimal based on patient interview.",
    "Vital signs are stable with temperature 98.6 degrees Fahrenheit.",
    "Patient has history of diabetes mellitus type 2 well controlled."
]
class InformationContent(BaseInformationTheory):
    """Calculate information content (self-information) for tokens and sequences.

    Information content I(x) = -log(P(x)) measures how surprising an event is.
    Higher probability events have lower information content. This is fundamental
    to understanding model confidence and token importance in language models.

    The information content quantifies the "surprise" of observing a particular
    event. In the context of language models, rare words or unexpected tokens
    carry more information than common, predictable ones.

    Example:
        Basic usage for calculating information content:

        ```python
        ic = InformationContent(base='2')  # Use bits
        probs = torch.tensor([0.5, 0.25, 0.25])
        info_content = ic.calculate(probs)
        # Returns: tensor([1.0000, 2.0000, 2.0000])
        ```

    Attributes:
        base: Logarithm base for calculations
        log_fn: PyTorch logarithm function
        epsilon: Numerical stability constant
    """

    def calculate(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate information content for given probabilities.

        Args:
            probabilities: Tensor of probabilities with shape [batch_size, vocab_size]
                or [vocab_size]. Values should be in range [0, 1].

        Returns:
            Information content tensor of same shape as input. Higher values
            indicate more surprising (lower probability) events.

        Raises:
            ValueError: If probabilities contain values outside [0, 1] range.

        Example:
            ```python
            ic = InformationContent(base='2')
            probs = torch.tensor([0.8, 0.1, 0.1])  # Common, rare, rare
            info = ic.calculate(probs)
            # info[0] < info[1] == info[2] (common word has less information)
            ```
        """
        if torch.any(probabilities < 0) or torch.any(probabilities > 1):
            raise ValueError("Probabilities must be in range [0, 1]")

        # Ensure numerical stability
        probs = self._stabilize_probabilities(probabilities)
        return -self.log_fn(probs)

    def from_logits(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate information content from raw logits.

        This method is more numerically stable than converting logits to
        probabilities manually, as it uses PyTorch's optimized softmax.

        Args:
            logits: Raw model outputs before softmax with shape [..., vocab_size]
            dim: Dimension along which to apply softmax (default: -1)

        Returns:
            Information content tensor with same shape as logits

        Example:
            ```python
            ic = InformationContent(base='2')
            logits = torch.randn(4, 1000)  # Batch of 4, vocab size 1000
            info = ic.from_logits(logits)
            # Shape: [4, 1000]
            ```
        """
        probs = F.softmax(logits, dim=dim)
        return self.calculate(probs)

    def token_information(self, token_probs: torch.Tensor) -> torch.Tensor:
        """Calculate information content for specific token probabilities.

        Convenience method for calculating information content of individual
        tokens or small sets of tokens.

        Args:
            token_probs: Probabilities of specific tokens with shape [batch_size]
                or scalar. Each value represents P(token).

        Returns:
            Information content for each token probability

        Example:
            ```python
            ic = InformationContent(base='2')
            # Probabilities for tokens "the", "patient", "pneumomediastinum"
            token_probs = torch.tensor([0.15, 0.08, 0.001])
            info = ic.token_information(token_probs)
            # Medical term has much higher information content
            ```
        """
        return self.calculate(token_probs)

    def surprise_threshold(self, probabilities: torch.Tensor,
                          threshold: float = 5.0) -> torch.Tensor:
        """Identify tokens that exceed a surprise threshold.

        Useful for flagging unexpected or out-of-distribution tokens that
        may require special attention in healthcare or safety-critical applications.

        Args:
            probabilities: Token probabilities
            threshold: Information content threshold (in units of chosen base)

        Returns:
            Boolean tensor indicating which tokens exceed the threshold

        Example:
            ```python
            ic = InformationContent(base='2')
            probs = torch.tensor([0.5, 0.1, 0.001])  # Common, uncommon, rare
            surprising = ic.surprise_threshold(probs, threshold=5.0)
            # Returns: tensor([False, False, True])  # Only rare token is surprising
            ```
        """
        info_content = self.calculate(probabilities)
        return info_content > threshold


class Entropy(BaseInformationTheory):
    """Calculate Shannon entropy for probability distributions.

    Entropy H(X) = -Σ P(x) * log(P(x)) measures the average uncertainty
    or randomness in a probability distribution. It's a fundamental concept
    in information theory that quantifies how much information is needed
    to describe the outcome of a random variable.

    In language modeling, entropy helps assess model confidence:
    - Low entropy: Model is confident (peaked distribution)
    - High entropy: Model is uncertain (uniform distribution)

    Example:
        Basic entropy calculation:

        ```python
        entropy_calc = Entropy(base='2')  # Use bits

        # Uniform distribution (maximum entropy)
        uniform_probs = torch.ones(4) / 4
        max_entropy = entropy_calc.calculate(uniform_probs)  # 2.0 bits

        # Peaked distribution (low entropy)
        peaked_probs = torch.tensor([0.9, 0.05, 0.03, 0.02])
        low_entropy = entropy_calc.calculate(peaked_probs)  # ~0.57 bits
        ```

    Attributes:
        base: Logarithm base for calculations
        log_fn: PyTorch logarithm function
        epsilon: Numerical stability constant
    """

    def calculate(self, probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate entropy of probability distribution.

        Args:
            probabilities: Probability distribution tensor with shape
                [batch_size, vocab_size] or any shape where the last dimension
                (or specified dim) represents the probability distribution
            dim: Dimension to sum over for entropy calculation (default: -1)

        Returns:
            Entropy values. If dim=-1 and input is [batch_size, vocab_size],
            returns [batch_size]. Otherwise returns tensor with specified
            dimension reduced.

        Raises:
            ValueError: If probabilities don't sum to 1 along specified dimension
                (within tolerance) or contain negative values

        Example:
            ```python
            entropy_calc = Entropy(base='2')

            # Batch of probability distributions
            probs = torch.tensor([[0.7, 0.2, 0.1],    # Low entropy
                                 [0.33, 0.33, 0.34]])  # Higher entropy
            entropies = entropy_calc.calculate(probs)
            # Returns: tensor([1.1568, 1.5849])
            ```
        """
        # Validate input
        if torch.any(probabilities < 0):
            raise ValueError("Probabilities cannot be negative")

        # Check if probabilities sum to 1 (within tolerance)
        prob_sums = torch.sum(probabilities, dim=dim, keepdim=True)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6):
            raise ValueError("Probabilities must sum to 1 along specified dimension")

        # Ensure numerical stability
        probs = self._stabilize_probabilities(probabilities)

        # Calculate -p * log(p) for each element
        log_probs = self.log_fn(probs)
        entropy_terms = -probs * log_probs

        # Sum over the specified dimension
        return torch.sum(entropy_terms, dim=dim)

    def from_logits(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate entropy from raw logits.

        More numerically stable than manually converting logits to probabilities.

        Args:
            logits: Raw model outputs before softmax
            dim: Dimension along which to apply softmax and calculate entropy

        Returns:
            Entropy values with specified dimension reduced

        Example:
            ```python
            entropy_calc = Entropy(base='2')
            logits = torch.randn(8, 1000)  # Batch of 8, vocab size 1000
            entropies = entropy_calc.from_logits(logits)
            # Shape: [8]
            ```
        """
        probs = F.softmax(logits, dim=dim)
        return self.calculate(probs, dim=dim)

    def conditional_entropy(self, joint_probs: torch.Tensor,
                          marginal_probs: torch.Tensor) -> torch.Tensor:
        """Calculate conditional entropy H(Y|X).

        Conditional entropy measures the average uncertainty in Y given
        knowledge of X. It's computed as H(Y|X) = Σ P(x) * H(Y|X=x).

        Args:
            joint_probs: Joint probability P(X,Y) with shape
                [batch_size, x_dim, y_dim]
            marginal_probs: Marginal probability P(X) with shape
                [batch_size, x_dim]

        Returns:
            Conditional entropy H(Y|X) with shape [batch_size]

        Example:
            ```python
            entropy_calc = Entropy(base='2')

            # Joint distribution for symptoms and diseases
            joint = torch.tensor([[[0.1, 0.05], [0.2, 0.15]],  # Patient 1
                                 [[0.3, 0.1], [0.05, 0.05]]])  # Patient 2
            marginal = torch.sum(joint, dim=-1)  # Sum over diseases

            cond_entropy = entropy_calc.conditional_entropy(joint, marginal)
            ```
        """
        # Calculate conditional probabilities P(Y|X) = P(X,Y) / P(X)
        marginal_expanded = marginal_probs.unsqueeze(-1)
        conditional_probs = joint_probs / self._stabilize_probabilities(marginal_expanded)

        # Calculate entropy for each conditional distribution
        conditional_entropies = self.calculate(conditional_probs, dim=-1)

        # Weight by marginal probabilities and sum
        return torch.sum(marginal_probs * conditional_entropies, dim=-1)

    def max_entropy(self, vocab_size: int) -> float:
        """Calculate maximum possible entropy for given vocabulary size.

        Maximum entropy occurs when all outcomes are equally likely.

        Args:
            vocab_size: Number of possible outcomes

        Returns:
            Maximum entropy value in the units of the chosen base

        Example:
            ```python
            entropy_calc = Entropy(base='2')
            max_ent = entropy_calc.max_entropy(1000)  # ~9.97 bits
            ```
        """
        if self.base == LogarithmBase.NATURAL:
            return math.log(vocab_size)
        elif self.base == LogarithmBase.BINARY:
            return math.log2(vocab_size)
        elif self.base == LogarithmBase.DECIMAL:
            return math.log10(vocab_size)

    def normalized_entropy(self, probabilities: torch.Tensor,
                          dim: int = -1) -> torch.Tensor:
        """Calculate entropy normalized by maximum possible entropy.

        Returns entropy as a fraction of maximum possible entropy,
        useful for comparing across different vocabulary sizes.

        Args:
            probabilities: Probability distribution
            dim: Dimension to calculate entropy over

        Returns:
            Normalized entropy values in range [0, 1]

        Example:
            ```python
            entropy_calc = Entropy(base='2')
            probs = torch.tensor([0.5, 0.5])  # Binary choice
            norm_entropy = entropy_calc.normalized_entropy(probs)
            # Returns: 1.0 (maximum entropy for binary choice)
            ```
        """
        entropy = self.calculate(probabilities, dim=dim)
        vocab_size = probabilities.shape[dim]
        max_entropy = self.max_entropy(vocab_size)
        return entropy / max_entropy


class InformationTheoryCalculator(BaseInformationTheory):
    """Unified calculator for all information theory metrics.

    This class provides a single interface for calculating all information
    theory metrics used in language model development. It combines the
    functionality of individual calculators into one convenient class.

    The calculator is designed for practical use in LLM development,
    evaluation, and analysis, with special attention to healthcare
    applications where uncertainty quantification is critical.

    Example:
        Complete information theory analysis:

        ```python
        calc = InformationTheoryCalculator(base='2')

        # Model predictions
        logits = torch.randn(4, 1000)  # Batch=4, vocab=1000
        targets = torch.randint(0, 1000, (4,))

        # Calculate all metrics
        results = calc.analyze_predictions(logits, targets)
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Entropy: {results['entropy']:.3f} bits")
        ```

    Attributes:
        base: Logarithm base for all calculations
        log_fn: PyTorch logarithm function
        epsilon: Numerical stability constant
        _info_content: Information content calculator
        _entropy: Entropy calculator
    """

    def __init__(self, base: Union[str, LogarithmBase] = LogarithmBase.NATURAL,
                 epsilon: float = 1e-8):
        """Initialize the unified information theory calculator.

        Args:
            base: Logarithm base for calculations
            epsilon: Small value for numerical stability
        """
        super().__init__(base, epsilon)

        # Initialize component calculators
        self._info_content = InformationContent(base, epsilon)
        self._entropy = Entropy(base, epsilon)

    def information_content(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate information content (self-information).

        Args:
            probabilities: Token probabilities

        Returns:
            Information content values

        Example:
            ```python
            calc = InformationTheoryCalculator(base='2')
            probs = torch.tensor([0.8, 0.1, 0.1])
            info = calc.information_content(probs)
            ```
        """
        return self._info_content.calculate(probabilities)

    def entropy(self, probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate Shannon entropy.

        Args:
            probabilities: Probability distribution
            dim: Dimension to calculate entropy over

        Returns:
            Entropy values

        Example:
            ```python
            calc = InformationTheoryCalculator(base='2')
            probs = torch.tensor([[0.7, 0.2, 0.1], [0.33, 0.33, 0.34]])
            entropies = calc.entropy(probs)
            ```
        """
        return self._entropy.calculate(probabilities, dim)

    def cross_entropy(self, true_probs: torch.Tensor,
                     pred_probs: torch.Tensor,
                     reduction: Union[str, ReductionType] = ReductionType.MEAN) -> torch.Tensor:
        """Calculate cross-entropy between distributions.

        Args:
            true_probs: True probability distribution
            pred_probs: Predicted probability distribution
            reduction: How to reduce the result

        Returns:
            Cross-entropy loss

        Example:
            ```python
            calc = InformationTheoryCalculator(base='e')
            true_probs = torch.tensor([[1.0, 0.0, 0.0]])
            pred_probs = torch.tensor([[0.7, 0.2, 0.1]])
            ce = calc.cross_entropy(true_probs, pred_probs)
            ```
        """
        # Ensure numerical stability
        pred_probs = self._stabilize_probabilities(pred_probs)

        # Calculate -p * log(q)
        log_pred_probs = self.log_fn(pred_probs)
        cross_entropy_terms = -true_probs * log_pred_probs

        # Sum over vocabulary dimension
        cross_entropy = torch.sum(cross_entropy_terms, dim=-1)

        return self._apply_reduction(cross_entropy, reduction)

    def sparse_cross_entropy(self, true_indices: torch.Tensor,
                           pred_logits: torch.Tensor,
                           reduction: Union[str, ReductionType] = ReductionType.MEAN) -> torch.Tensor:
        """Calculate cross-entropy for sparse labels (typical in LM training).

        Args:
            true_indices: True token indices
            pred_logits: Predicted logits
            reduction: How to reduce the result

        Returns:
            Cross-entropy loss

        Example:
            ```python
            calc = InformationTheoryCalculator(base='e')
            targets = torch.tensor([42, 156, 789])
            logits = torch.randn(3, 1000)
            loss = calc.sparse_cross_entropy(targets, logits)
            ```
        """
        # Use PyTorch's built-in function for efficiency
        if self.base == LogarithmBase.NATURAL:
            return F.cross_entropy(pred_logits, true_indices,
                                 reduction=reduction.value if isinstance(reduction, ReductionType) else reduction)
        else:
            # Convert to desired base
            ce_nats = F.cross_entropy(pred_logits, true_indices, reduction='none')
            if self.base == LogarithmBase.BINARY:
                ce_converted = ce_nats / math.log(2)
            elif self.base == LogarithmBase.DECIMAL:
                ce_converted = ce_nats / math.log(10)

            return self._apply_reduction(ce_converted, reduction)

    def kl_divergence(self, p_probs: torch.Tensor, q_probs: torch.Tensor,
                     reduction: Union[str, ReductionType] = ReductionType.BATCHMEAN) -> torch.Tensor:
        """Calculate Kullback-Leibler divergence KL(P||Q).

        KL divergence measures how much distribution P differs from Q.
        It's asymmetric: KL(P||Q) ≠ KL(Q||P).

        Args:
            p_probs: Distribution P (reference)
            q_probs: Distribution Q (approximation)
            reduction: How to reduce the result

        Returns:
            KL divergence value

        Example:
            ```python
            calc = InformationTheoryCalculator(base='e')
            p = torch.tensor([[0.7, 0.2, 0.1]])
            q = torch.tensor([[0.6, 0.3, 0.1]])
            kl = calc.kl_divergence(p, q)
            ```
        """
        # Ensure numerical stability
        p_probs = self._stabilize_probabilities(p_probs)
        q_probs = self._stabilize_probabilities(q_probs)

        # Calculate P * log(P/Q) = P * (log(P) - log(Q))
        log_p = self.log_fn(p_probs)
        log_q = self.log_fn(q_probs)

        kl_terms = p_probs * (log_p - log_q)
        kl_div = torch.sum(kl_terms, dim=-1)

        return self._apply_reduction(kl_div, reduction)

    def kl_divergence_from_logits(self, p_logits: torch.Tensor,
                                 q_logits: torch.Tensor,
                                 reduction: Union[str, ReductionType] = ReductionType.BATCHMEAN) -> torch.Tensor:
        """Calculate KL divergence from logits (more stable).

        Args:
            p_logits: Logits for distribution P
            q_logits: Logits for distribution Q
            reduction: How to reduce the result

        Returns:
            KL divergence value

        Example:
            ```python
            calc = InformationTheoryCalculator(base='e')
            p_logits = torch.randn(4, 1000)
            q_logits = torch.randn(4, 1000)
            kl = calc.kl_divergence_from_logits(p_logits, q_logits)
            ```
        """
        p_probs = F.softmax(p_logits, dim=-1)
        q_probs = F.softmax(q_logits, dim=-1)
        return self.kl_divergence(p_probs, q_probs, reduction)

    def mutual_information(self, joint_probs: torch.Tensor) -> torch.Tensor:
        """Calculate mutual information I(X;Y) from joint distribution.

        Mutual information measures how much information X and Y share.
        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Args:
            joint_probs: Joint probability P(X,Y) with shape [x_dim, y_dim]

        Returns:
            Mutual information value

        Example:
            ```python
            calc = InformationTheoryCalculator(base='2')
            # Joint distribution for symptoms and diseases
            joint = torch.tensor([[0.3, 0.1], [0.2, 0.4]])
            mi = calc.mutual_information(joint)
            ```
        """
        # Calculate marginal probabilities
        marginal_x = torch.sum(joint_probs, dim=1)  # P(X)
        marginal_y = torch.sum(joint_probs, dim=0)  # P(Y)

        # Calculate entropies
        h_x = self._entropy.calculate(marginal_x)
        h_y = self._entropy.calculate(marginal_y)
        h_xy = self._entropy.calculate(joint_probs.flatten())

        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        return h_x + h_y - h_xy

    def perplexity(self, pred_logits: torch.Tensor, true_tokens: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  ignore_index: int = -100) -> torch.Tensor:
        """Calculate perplexity from model predictions.

        Perplexity = exp(cross_entropy) measures how surprised the model
        is by the actual sequence. Lower perplexity indicates better performance.

        Args:
            pred_logits: Model predictions with shape [batch_size, seq_len, vocab_size]
            true_tokens: True token indices with shape [batch_size, seq_len]
            mask: Optional mask for valid tokens
            ignore_index: Token index to ignore (e.g., padding)

        Returns:
            Perplexity value

        Example:
            ```python
            calc = InformationTheoryCalculator(base='e')
            logits = torch.randn(2, 10, 1000)  # 2 sequences, length 10, vocab 1000
            tokens = torch.randint(0, 1000, (2, 10))
            ppl = calc.perplexity(logits, tokens)
            ```
        """
        # Reshape for cross-entropy calculation
        batch_size, seq_len, vocab_size = pred_logits.shape
        pred_logits_flat = pred_logits.view(-1, vocab_size)
        true_tokens_flat = true_tokens.view(-1)

        # Calculate cross-entropy for each token
        ce_losses = F.cross_entropy(pred_logits_flat, true_tokens_flat,
                                  ignore_index=ignore_index, reduction='none')

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            ce_losses = ce_losses * mask_flat
            valid_tokens = mask_flat.sum()
        else:
            # Count valid tokens (not ignored)
            valid_mask = (true_tokens_flat != ignore_index)
            ce_losses = ce_losses * valid_mask.float()
            valid_tokens = valid_mask.sum()

        # Calculate average cross-entropy
        avg_ce = ce_losses.sum() / valid_tokens

        # Convert to perplexity
        if self.base == LogarithmBase.NATURAL:
            return torch.exp(avg_ce)
        elif self.base == LogarithmBase.BINARY:
            return 2 ** avg_ce
        elif self.base == LogarithmBase.DECIMAL:
            return 10 ** avg_ce

    def analyze_predictions(self, pred_logits: torch.Tensor,
                          true_tokens: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Comprehensive analysis of model predictions using information theory.

        This method calculates multiple information theory metrics to provide
        a complete picture of model performance and behavior.

        Args:
            pred_logits: Model predictions with shape [batch_size, seq_len, vocab_size]
            true_tokens: True token indices with shape [batch_size, seq_len]
            mask: Optional mask for valid tokens

        Returns:
            Dictionary containing all calculated metrics:
            - 'perplexity': Model perplexity
            - 'cross_entropy': Average cross-entropy loss
            - 'entropy': Average prediction entropy
            - 'max_entropy': Maximum possible entropy
            - 'normalized_entropy': Entropy normalized by maximum
            - 'confidence': Average model confidence (1 - normalized_entropy)

        Example:
            ```python
            calc = InformationTheoryCalculator(base='2')
            logits = torch.randn(4, 20, 1000)
            tokens = torch.randint(0, 1000, (4, 20))

            results = calc.analyze_predictions(logits, tokens)
            print(f"Perplexity: {results['perplexity']:.2f}")
            print(f"Average entropy: {results['entropy']:.3f} bits")
            print(f"Model confidence: {results['confidence']:.1%}")
            ```
        """
        # Calculate perplexity and cross-entropy
        perplexity = self.perplexity(pred_logits, true_tokens, mask)
        cross_entropy = self.sparse_cross_entropy(true_tokens.view(-1),
                                                 pred_logits.view(-1, pred_logits.size(-1)))

        # Calculate prediction entropy
        probs = F.softmax(pred_logits, dim=-1)
        entropy = self.entropy(probs, dim=-1)

        if mask is not None:
            # Apply mask to entropy calculation
            entropy = entropy * mask
            avg_entropy = entropy.sum() / mask.sum()
        else:
            avg_entropy = entropy.mean()

        # Calculate maximum possible entropy and normalized metrics
        vocab_size = pred_logits.size(-1)
        max_entropy = self._entropy.max_entropy(vocab_size)
        normalized_entropy = avg_entropy / max_entropy
        confidence = 1.0 - normalized_entropy

        return {
            'perplexity': perplexity.item(),
            'cross_entropy': cross_entropy.item(),
            'entropy': avg_entropy.item(),
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy.item(),
            'confidence': confidence.item()
        }

    def uncertainty_analysis(self, pred_logits: torch.Tensor,
                           uncertainty_threshold: float = 0.7) -> Dict[str, Any]:
        """Analyze prediction uncertainty for safety-critical applications.

        Identifies tokens where the model is uncertain, which is crucial
        for healthcare and other high-stakes applications.

        Args:
            pred_logits: Model predictions
            uncertainty_threshold: Normalized entropy threshold for uncertainty

        Returns:
            Dictionary with uncertainty analysis:
            - 'uncertain_positions': Boolean tensor marking uncertain predictions
            - 'uncertainty_rate': Fraction of predictions that are uncertain
            - 'avg_uncertainty': Average normalized entropy
            - 'max_uncertainty': Maximum normalized entropy

        Example:
            ```python
            calc = InformationTheoryCalculator(base='2')
            logits = torch.randn(1, 50, 1000)  # 50 tokens, vocab 1000

            uncertainty = calc.uncertainty_analysis(logits, threshold=0.8)
            print(f"Uncertain tokens: {uncertainty['uncertainty_rate']:.1%}")
            ```
        """
        probs = F.softmax(pred_logits, dim=-1)
        entropy = self.entropy(probs, dim=-1)

        vocab_size = pred_logits.size(-1)
        max_entropy = self._entropy.max_entropy(vocab_size)
        normalized_entropy = entropy / max_entropy

        uncertain_positions = normalized_entropy > uncertainty_threshold
        uncertainty_rate = uncertain_positions.float().mean()

        return {
            'uncertain_positions': uncertain_positions,
            'uncertainty_rate': uncertainty_rate.item(),
            'avg_uncertainty': normalized_entropy.mean().item(),
            'max_uncertainty': normalized_entropy.max().item()
        }


class HealthcareInformationTheoryDemo:
    """Demonstration class for healthcare applications of information theory.

    This class provides comprehensive examples of how information theory
    concepts apply to healthcare AI and language model development in
    medical domains.

    Example:
        Run healthcare demonstrations:

        ```python
        demo = HealthcareInformationTheoryDemo()
        demo.run_all_demonstrations()
        ```
    """

    def __init__(self, base: Union[str, LogarithmBase] = LogarithmBase.BINARY):
        """Initialize the healthcare demonstration.

        Args:
            base: Logarithm base for calculations (default: binary for bits)
        """
        self.calc = InformationTheoryCalculator(base)
        self.base = base

    def demonstrate_information_content(self) -> None:
        """Demonstrate information content with healthcare vocabulary.

        Shows how rare medical terms carry more information than common words,
        which is crucial for understanding model behavior in medical contexts.
        """
        print("=" * 70)
        print("INFORMATION CONTENT ANALYSIS - HEALTHCARE VOCABULARY")
        print("=" * 70)

        # Healthcare vocabulary with varying frequencies
        vocab = ["the", "patient", "presents", "with", "pain", "pneumomediastinum"]
        # Simulate realistic probabilities (rare conditions have lower probability)
        probs = torch.tensor([0.15, 0.12, 0.08, 0.10, 0.05, 0.001])

        information_content = self.calc.information_content(probs)

        print("Medical Vocabulary Information Content Analysis:")
        print("-" * 60)
        print(f"{'Word':<20} | {'Probability':<12} | {'Info Content':<15}")
        print("-" * 60)

        for word, prob, ic_val in zip(vocab, probs, information_content):
            unit = "bits" if self.base == LogarithmBase.BINARY else "nats"
            print(f"{word:<20} | {prob:<12.3f} | {ic_val:<15.2f} {unit}")

        print(f"\nKey Insights:")
        print(f"• 'pneumomediastinum' has {information_content[-1]:.1f} {unit}")
        print(f"• 'the' has only {information_content[0]:.1f} {unit}")
        print(f"• Rare medical terms carry {information_content[-1]/information_content[0]:.1f}x more information")
        print(f"• This helps identify when models encounter unusual medical conditions")

    def demonstrate_entropy_analysis(self) -> None:
        """Demonstrate entropy analysis for diagnostic uncertainty.

        Shows how entropy can quantify diagnostic uncertainty and guide
        clinical decision-making processes.
        """
        print("\n" + "=" * 70)
        print("ENTROPY ANALYSIS - DIAGNOSTIC UNCERTAINTY")
        print("=" * 70)

        # Different diagnostic scenarios
        scenarios = {
            "Clear Diagnosis": torch.tensor([0.85, 0.08, 0.04, 0.02, 0.01]),
            "Uncertain Diagnosis": torch.tensor([0.25, 0.20, 0.18, 0.17, 0.20]),
            "Differential Diagnosis": torch.tensor([0.40, 0.35, 0.15, 0.06, 0.04]),
            "Uniform Uncertainty": torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20])
        }

        print("Diagnostic Uncertainty Analysis:")
        print("-" * 50)
        unit = "bits" if self.base == LogarithmBase.BINARY else "nats"

        for scenario_name, probs in scenarios.items():
            entropy = self.calc.entropy(probs)
            max_entropy = self.calc._entropy.max_entropy(len(probs))
            normalized = entropy / max_entropy

            print(f"{scenario_name:<20}: {entropy:.3f} {unit} "
                  f"(normalized: {normalized:.1%})")

        print(f"\nClinical Interpretation:")
        print(f"• Higher entropy → More diagnostic uncertainty")
        print(f"• Lower entropy → More confident diagnosis")
        print(f"• Normalized entropy helps compare across different numbers of diagnoses")
        print(f"• Threshold-based alerts can flag uncertain cases for expert review")

    def demonstrate_cross_entropy_training(self) -> None:
        """Demonstrate cross-entropy in medical language model training.

        Shows how cross-entropy loss guides model training and how different
        prediction qualities affect the loss.
        """
        print("\n" + "=" * 70)
        print("CROSS-ENTROPY ANALYSIS - MEDICAL LANGUAGE MODEL TRAINING")
        print("=" * 70)

        # Medical vocabulary for demonstration
        medical_vocab = ["pain", "fever", "nausea", "fatigue", "pneumonia"]
        vocab_size = len(medical_vocab)

        # Different prediction scenarios
        scenarios = [
            ("Excellent Prediction", torch.tensor([[3.0, -2.0, -2.0, -2.0, -2.0]])),
            ("Good Prediction", torch.tensor([[1.5, -0.5, -0.5, -0.5, -0.5]])),
            ("Poor Prediction", torch.tensor([[0.2, 0.1, 0.1, 0.1, 0.1]])),
            ("Random Prediction", torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]))
        ]

        true_token = torch.tensor([0])  # "pain" is the correct answer

        print("Medical Term Prediction Analysis:")
        print("-" * 45)
        print(f"{'Scenario':<20} | {'Cross-Entropy':<15} | {'Perplexity':<12}")
        print("-" * 45)

        for scenario_name, logits in scenarios:
            ce_loss = self.calc.sparse_cross_entropy(true_token, logits,
                                                    reduction=ReductionType.NONE)
            # Calculate perplexity (convert to natural log if needed)
            if self.base != LogarithmBase.NATURAL:
                ce_nats = ce_loss * math.log(2) if self.base == LogarithmBase.BINARY else ce_loss * math.log(10)
            else:
                ce_nats = ce_loss
            perplexity = torch.exp(ce_nats)

            print(f"{scenario_name:<20} | {ce_loss.item():<15.4f} | {perplexity.item():<12.2f}")

        print(f"\nTraining Insights:")
        print(f"• Lower cross-entropy = Better prediction quality")
        print(f"• Cross-entropy directly guides gradient updates")
        print(f"• Perplexity provides intuitive interpretation")
        print(f"• Medical terms require careful handling due to class imbalance")

    def run_all_demonstrations(self) -> None:
        """Run all healthcare information theory demonstrations.

        Provides a comprehensive overview of information theory applications
        in healthcare AI and medical language modeling.
        """
        print("HEALTHCARE INFORMATION THEORY DEMONSTRATIONS")
        print("=" * 70)
        print("Exploring information theory concepts in medical AI applications")
        print("=" * 70)

        self.demonstrate_information_content()
        self.demonstrate_entropy_analysis()
        self.demonstrate_cross_entropy_training()

        print("\n" + "=" * 70)
        print("SUMMARY AND CLINICAL APPLICATIONS")
        print("=" * 70)
        print("Key applications in healthcare AI:")
        print("• Uncertainty quantification for clinical decision support")
        print("• Rare condition detection through information content analysis")
        print("• Model confidence assessment for safety-critical applications")
        print("• Training optimization for medical language models")
        print("• Quality assurance for AI-assisted diagnosis")
        print("=" * 70)


def demonstrate_unified_calculator() -> None:
    """Demonstrate the unified InformationTheoryCalculator.

    Shows how to use the unified calculator for comprehensive analysis
    of language model predictions.
    """
    print("UNIFIED INFORMATION THEORY CALCULATOR DEMONSTRATION")
    print("=" * 60)

    # Initialize calculator
    calc = InformationTheoryCalculator(base=LogarithmBase.BINARY)

    # Simulate model predictions
    batch_size, seq_len, vocab_size = 4, 20, 1000
    pred_logits = torch.randn(batch_size, seq_len, vocab_size)
    true_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Comprehensive analysis
    results = calc.analyze_predictions(pred_logits, true_tokens)

    print("Model Performance Analysis:")
    print("-" * 40)
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Cross-entropy: {results['cross_entropy']:.4f}")
    print(f"Average entropy: {results['entropy']:.3f} bits")
    print(f"Normalized entropy: {results['normalized_entropy']:.1%}")
    print(f"Model confidence: {results['confidence']:.1%}")

    # Uncertainty analysis
    uncertainty = calc.uncertainty_analysis(pred_logits, uncertainty_threshold=0.8)
    print(f"\nUncertainty Analysis:")
    print(f"Uncertain predictions: {uncertainty['uncertainty_rate']:.1%}")
    print(f"Average uncertainty: {uncertainty['avg_uncertainty']:.3f}")
    print(f"Maximum uncertainty: {uncertainty['max_uncertainty']:.3f}")


# Legacy classes maintained for backward compatibility
# These are kept for users who might be using the individual classes
# The recommended approach is to use InformationTheoryCalculator
class CrossEntropy:
    """
    Calculate cross-entropy between true and predicted distributions.

    Cross-entropy H(P,Q) = -Σ P(x) * log(Q(x)) measures the difference
    between true distribution P and predicted distribution Q.
    """

    def __init__(self, base: str = 'e', reduction: str = 'mean'):
        """
        Initialize cross-entropy calculator.

        Args:
            base: Logarithm base ('e', '2', or '10')
            reduction: How to reduce batch dimension ('mean', 'sum', 'none')
        """
        self.base = base
        self.reduction = reduction

        if base == 'e':
            self.log_fn = torch.log
        elif base == '2':
            self.log_fn = torch.log2
        elif base == '10':
            self.log_fn = torch.log10
        else:
            raise ValueError("Base must be 'e', '2', or '10'")

    def calculate(self, true_probs: torch.Tensor,
                  pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross-entropy between probability distributions.

        Args:
            true_probs: True probability distribution [batch_size, vocab_size]
            pred_probs: Predicted probability distribution [batch_size, vocab_size]

        Returns:
            Cross-entropy loss
        """
        # Ensure numerical stability
        pred_probs = add_epsilon(pred_probs)

        # Calculate -p * log(q)
        log_pred_probs = self.log_fn(pred_probs)
        cross_entropy_terms = -true_probs * log_pred_probs

        # Sum over vocabulary dimension
        cross_entropy = torch.sum(cross_entropy_terms, dim=-1)

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(cross_entropy)
        elif self.reduction == 'sum':
            return torch.sum(cross_entropy)
        else:
            return cross_entropy

    def from_logits(self, true_probs: torch.Tensor,
                    pred_logits: torch.Tensor) -> torch.Tensor:
        """Calculate cross-entropy from logits (more numerically stable)."""
        pred_probs = F.softmax(pred_logits, dim=-1)
        return self.calculate(true_probs, pred_probs)

    def sparse_cross_entropy(self, true_indices: torch.Tensor,
                           pred_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross-entropy for sparse true labels (typical in LM training).

        Args:
            true_indices: True token indices [batch_size]
            pred_logits: Predicted logits [batch_size, vocab_size]

        Returns:
            Cross-entropy loss
        """
        # Use PyTorch's built-in function for efficiency and stability
        if self.base == 'e':
            return F.cross_entropy(pred_logits, true_indices, reduction=self.reduction)
        else:
            # Convert to desired base
            ce_nats = F.cross_entropy(pred_logits, true_indices, reduction='none')
            if self.base == '2':
                ce_converted = ce_nats / math.log(2)
            elif self.base == '10':
                ce_converted = ce_nats / math.log(10)

            if self.reduction == 'mean':
                return torch.mean(ce_converted)
            elif self.reduction == 'sum':
                return torch.sum(ce_converted)
            else:
                return ce_converted


def demonstrate_cross_entropy():
    """Demonstrate cross-entropy calculation with language model examples."""

    ce_calc = CrossEntropy(base='e', reduction='none')

    # Simulate a simple language model scenario
    vocab_size = 1000
    batch_size = 4

    # Create some example predictions (logits)
    pred_logits = torch.randn(batch_size, vocab_size)

    # True next tokens (sparse format)
    true_tokens = torch.tensor([42, 156, 789, 23])  # Example token indices

    # Calculate cross-entropy loss
    ce_loss = ce_calc.sparse_cross_entropy(true_tokens, pred_logits)

    print("Cross-Entropy Loss Analysis:")
    print("-" * 40)
    print(f"Batch size: {batch_size}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Cross-entropy losses: {ce_loss}")
    print(f"Average loss: {ce_loss.mean():.4f}")

    # Healthcare example: Medical term prediction
    print("\nHealthcare Example - Medical Term Prediction:")
    print("-" * 55)

    # Simulate predictions for medical terms
    medical_vocab = ["pain", "fever", "nausea", "fatigue", "pneumonia"]
    vocab_size = len(medical_vocab)

    # Good prediction (model is confident about correct term)
    good_logits = torch.tensor([[2.0, -1.0, -1.0, -1.0, -1.0]])  # High confidence in "pain"
    true_token = torch.tensor([0])  # "pain" is correct

    # Poor prediction (model is uncertain)
    poor_logits = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]])  # Uniform uncertainty

    good_loss = ce_calc.sparse_cross_entropy(true_token, good_logits)
    poor_loss = ce_calc.sparse_cross_entropy(true_token, poor_logits)

    print(f"Good prediction loss: {good_loss.item():.4f}")
    print(f"Poor prediction loss: {poor_loss.item():.4f}")
    print(f"Loss ratio: {poor_loss.item() / good_loss.item():.2f}x higher for poor prediction")


class KLDivergence:
    """
    Calculate Kullback-Leibler divergence between probability distributions.
    
    KL(P||Q) = Σ P(x) * log(P(x)/Q(x)) measures how much distribution P
    differs from distribution Q.
    """
    
    def __init__(self, base: str = 'e', reduction: str = 'batchmean'):
        """
        Initialize KL divergence calculator.
        
        Args:
            base: Logarithm base ('e', '2', or '10')
            reduction: How to reduce ('batchmean', 'sum', 'mean', 'none')
        """
        self.base = base
        self.reduction = reduction
        
        if base == 'e':
            self.log_fn = torch.log
        elif base == '2':
            self.log_fn = torch.log2
        elif base == '10':
            self.log_fn = torch.log10
        else:
            raise ValueError("Base must be 'e', '2', or '10'")
    
    def calculate(self, p_probs: torch.Tensor, 
                  q_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence KL(P||Q).
        
        Args:
            p_probs: Distribution P [batch_size, vocab_size]
            q_probs: Distribution Q [batch_size, vocab_size]
            
        Returns:
            KL divergence
        """
        # Ensure numerical stability
        p_probs = add_epsilon(p_probs)
        q_probs = add_epsilon(q_probs)
        
        # Calculate P * log(P/Q) = P * (log(P) - log(Q))
        log_p = self.log_fn(p_probs)
        log_q = self.log_fn(q_probs)
        
        kl_terms = p_probs * (log_p - log_q)
        kl_div = torch.sum(kl_terms, dim=-1)
        
        # Apply reduction
        if self.reduction == 'batchmean':
            return torch.mean(kl_div)
        elif self.reduction == 'sum':
            return torch.sum(kl_div)
        elif self.reduction == 'mean':
            return torch.mean(kl_div)
        else:
            return kl_div
    
    def from_logits(self, p_logits: torch.Tensor, 
                    q_logits: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence from logits."""
        p_probs = F.softmax(p_logits, dim=-1)
        q_probs = F.softmax(q_logits, dim=-1)
        return self.calculate(p_probs, q_probs)
    
    def pytorch_kl_div(self, p_logits: torch.Tensor, 
                       q_logits: torch.Tensor) -> torch.Tensor:
        """Use PyTorch's built-in KL divergence (more efficient)."""
        # PyTorch's kl_div expects log probabilities as first argument
        log_q = F.log_softmax(q_logits, dim=-1)
        p_probs = F.softmax(p_logits, dim=-1)
        
        kl_div = F.kl_div(log_q, p_probs, reduction=self.reduction)
        
        # Convert to desired base if needed
        if self.base == '2':
            return kl_div / math.log(2)
        elif self.base == '10':
            return kl_div / math.log(10)
        else:
            return kl_div

def demonstrate_kl_divergence():
    """Demonstrate KL divergence with RLHF and fine-tuning examples."""
    
    kl_calc = KLDivergence(base='e', reduction='none')
    
    # Example 1: RLHF scenario - measuring deviation from reference model
    print("KL Divergence Analysis - RLHF Scenario:")
    print("-" * 45)
    
    vocab_size = 100
    batch_size = 3
    
    # Reference model (original model before RLHF)
    ref_logits = torch.randn(batch_size, vocab_size)
    
    # Fine-tuned model (after RLHF training)
    # Scenario 1: Small deviation (good)
    small_deviation_logits = ref_logits + 0.1 * torch.randn(batch_size, vocab_size)
    
    # Scenario 2: Large deviation (potentially problematic)
    large_deviation_logits = ref_logits + 2.0 * torch.randn(batch_size, vocab_size)
    
    small_kl = kl_calc.from_logits(ref_logits, small_deviation_logits)
    large_kl = kl_calc.from_logits(ref_logits, large_deviation_logits)
    
    print(f"Small deviation KL: {small_kl.mean():.4f} ± {small_kl.std():.4f}")
    print(f"Large deviation KL: {large_kl.mean():.4f} ± {large_kl.std():.4f}")
    print(f"Ratio: {large_kl.mean() / small_kl.mean():.1f}x larger")
    
    # Example 2: Healthcare domain adaptation
    print("\nHealthcare Example - Domain Adaptation:")
    print("-" * 45)
    
    # General model distribution (more uniform over general vocabulary)
    general_probs = F.softmax(torch.randn(1, 10), dim=-1)
    
    # Medical model distribution (focused on medical terms)
    medical_logits = torch.randn(1, 10)
    medical_logits[0, [2, 5, 7]] += 2.0  # Boost medical terms
    medical_probs = F.softmax(medical_logits, dim=-1)
    
    # Calculate bidirectional KL divergence
    kl_general_to_medical = kl_calc.calculate(general_probs, medical_probs)
    kl_medical_to_general = kl_calc.calculate(medical_probs, general_probs)
    
    print(f"KL(General || Medical): {kl_general_to_medical.item():.4f}")
    print(f"KL(Medical || General): {kl_medical_to_general.item():.4f}")
    print("Note: KL divergence is asymmetric!")
    
    # Jensen-Shannon divergence (symmetric alternative)
    js_div = 0.5 * (kl_general_to_medical + kl_medical_to_general)
    print(f"Jensen-Shannon divergence: {js_div.item():.4f}")


class MutualInformation:
    """
    Calculate mutual information between random variables.
    
    I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
    measures how much information X and Y share.
    """
    
    def __init__(self, base: str = 'e'):
        """Initialize with logarithm base."""
        self.base = base
        self.entropy_calc = Entropy(base=base)
    
    def calculate_discrete(self, joint_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate mutual information from joint probability distribution.
        
        Args:
            joint_probs: Joint probability P(X,Y) [x_dim, y_dim]
            
        Returns:
            Mutual information I(X;Y)
        """
        # Calculate marginal probabilities
        marginal_x = torch.sum(joint_probs, dim=1)  # P(X)
        marginal_y = torch.sum(joint_probs, dim=0)  # P(Y)
        
        # Calculate entropies
        h_x = self.entropy_calc.calculate(marginal_x)
        h_y = self.entropy_calc.calculate(marginal_y)
        h_xy = self.entropy_calc.calculate(joint_probs.flatten())
        
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        return h_x + h_y - h_xy
    
    def calculate_from_samples(self, x_samples: torch.Tensor, 
                             y_samples: torch.Tensor, 
                             bins: int = 10) -> torch.Tensor:
        """
        Estimate mutual information from samples using binning.
        
        Args:
            x_samples: Samples from X [n_samples]
            y_samples: Samples from Y [n_samples]
            bins: Number of bins for discretization
            
        Returns:
            Estimated mutual information
        """
        # Discretize continuous variables
        x_discrete = torch.floor(bins * (x_samples - x_samples.min()) / 
                               (x_samples.max() - x_samples.min() + 1e-8)).long()
        y_discrete = torch.floor(bins * (y_samples - y_samples.min()) / 
                               (y_samples.max() - y_samples.min() + 1e-8)).long()
        
        # Clamp to valid range
        x_discrete = torch.clamp(x_discrete, 0, bins - 1)
        y_discrete = torch.clamp(y_discrete, 0, bins - 1)
        
        # Create joint histogram
        joint_hist = torch.zeros(bins, bins)
        for i in range(len(x_samples)):
            joint_hist[x_discrete[i], y_discrete[i]] += 1
        
        # Convert to probabilities
        joint_probs = joint_hist / joint_hist.sum()
        
        return self.calculate_discrete(joint_probs)
    
    def neural_estimation(self, x_samples: torch.Tensor, 
                         y_samples: torch.Tensor,
                         hidden_dim: int = 64,
                         n_epochs: int = 100) -> torch.Tensor:
        """
        Estimate mutual information using neural estimation (MINE).
        
        This is a simplified version of the MINE algorithm.
        """
        
        class MINENet(nn.Module):
            def __init__(self, x_dim: int, y_dim: int, hidden_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(x_dim + y_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                xy = torch.cat([x, y], dim=-1)
                return self.net(xy)
        
        # Prepare data
        x_dim = x_samples.shape[-1] if x_samples.dim() > 1 else 1
        y_dim = y_samples.shape[-1] if y_samples.dim() > 1 else 1
        
        if x_samples.dim() == 1:
            x_samples = x_samples.unsqueeze(-1)
        if y_samples.dim() == 1:
            y_samples = y_samples.unsqueeze(-1)
        
        # Create shuffled samples for negative examples
        y_shuffled = y_samples[torch.randperm(len(y_samples))]
        
        # Initialize network
        mine_net = MINENet(x_dim, y_dim, hidden_dim)
        optimizer = torch.optim.Adam(mine_net.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(n_epochs):
            # Positive samples
            pos_scores = mine_net(x_samples, y_samples)
            
            # Negative samples
            neg_scores = mine_net(x_samples, y_shuffled)
            
            # MINE loss
            loss = -(torch.mean(pos_scores) - torch.log(torch.mean(torch.exp(neg_scores))))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final estimate
        with torch.no_grad():
            pos_scores = mine_net(x_samples, y_samples)
            neg_scores = mine_net(x_samples, y_shuffled)
            mi_estimate = torch.mean(pos_scores) - torch.log(torch.mean(torch.exp(neg_scores)))
        
        return mi_estimate

def demonstrate_mutual_information():
    """Demonstrate mutual information calculation with healthcare examples."""
    
    mi_calc = MutualInformation(base='2')
    
    # Example 1: Discrete case - symptoms and diagnoses
    print("Mutual Information Analysis - Healthcare:")
    print("-" * 45)
    
    # Simulate joint distribution of symptoms and diagnoses
    # Rows: symptoms (fever, cough, fatigue)
    # Cols: diagnoses (flu, cold, pneumonia)
    joint_probs = torch.tensor([
        [0.15, 0.05, 0.02],  # fever
        [0.10, 0.20, 0.08],  # cough  
        [0.08, 0.12, 0.20]   # fatigue
    ])
    
    # Normalize to ensure it's a valid probability distribution
    joint_probs = joint_probs / joint_probs.sum()
    
    mi_symptoms_diagnosis = mi_calc.calculate_discrete(joint_probs)
    
    print(f"Mutual information between symptoms and diagnosis: {mi_symptoms_diagnosis:.4f} bits")
    
    # Calculate marginal entropies for context
    marginal_symptoms = torch.sum(joint_probs, dim=1)
    marginal_diagnosis = torch.sum(joint_probs, dim=0)
    
    h_symptoms = mi_calc.entropy_calc.calculate(marginal_symptoms)
    h_diagnosis = mi_calc.entropy_calc.calculate(marginal_diagnosis)
    
    print(f"Symptom entropy: {h_symptoms:.4f} bits")
    print(f"Diagnosis entropy: {h_diagnosis:.4f} bits")
    print(f"Information reduction: {mi_symptoms_diagnosis/h_diagnosis:.1%}")
    
    # Example 2: Continuous case - vital signs correlation
    print("\nContinuous Case - Vital Signs Correlation:")
    print("-" * 45)
    
    n_samples = 1000
    
    # Generate correlated vital signs (blood pressure and heart rate)
    # Higher correlation = higher mutual information
    
    # Case 1: High correlation
    bp_base = torch.randn(n_samples)
    hr_high_corr = bp_base + 0.2 * torch.randn(n_samples)  # Highly correlated
    
    # Case 2: Low correlation  
    hr_low_corr = torch.randn(n_samples)  # Independent
    
    mi_high = mi_calc.calculate_from_samples(bp_base, hr_high_corr, bins=20)
    mi_low = mi_calc.calculate_from_samples(bp_base, hr_low_corr, bins=20)
    
    print(f"MI (BP, HR) - high correlation: {mi_high:.4f} bits")
    print(f"MI (BP, HR) - low correlation:  {mi_low:.4f} bits")
    print(f"Ratio: {mi_high/mi_low:.1f}x higher for correlated variables")
    
    # Example 3: Attention analysis simulation
    print("\nAttention Analysis Simulation:")
    print("-" * 35)
    
    # Simulate attention weights and output quality
    # High MI suggests attention is focusing on informative tokens
    
    attention_weights = torch.softmax(torch.randn(50), dim=0)  # 50 tokens
    
    # Create output quality that depends on attention to important tokens
    important_tokens = torch.zeros(50)
    important_tokens[[5, 12, 23, 34, 41]] = 1.0  # Mark important tokens
    
    output_quality = torch.sum(attention_weights * important_tokens) + 0.1 * torch.randn(1)
    
    # Replicate for multiple examples
    attention_batch = torch.randn(100, 50)
    quality_batch = torch.sum(F.softmax(attention_batch, dim=-1) * important_tokens, dim=-1)
    quality_batch += 0.1 * torch.randn(100)
    
    # Calculate MI between attention on important tokens and output quality
    attention_on_important = torch.sum(F.softmax(attention_batch, dim=-1) * important_tokens, dim=-1)
    
    mi_attention = mi_calc.calculate_from_samples(attention_on_important, quality_batch, bins=15)
    
    print(f"MI (Attention on important tokens, Output quality): {mi_attention:.4f} bits")
    print("Higher MI suggests attention mechanism is working effectively!")


class Perplexity:
    """
    Calculate perplexity for language model evaluation.
    
    Perplexity = exp(cross_entropy) measures how surprised the model
    is by the actual sequence. Lower perplexity indicates better performance.
    """
    
    def __init__(self, base: str = 'e', ignore_index: int = -100):
        """
        Initialize perplexity calculator.
        
        Args:
            base: Logarithm base for cross-entropy calculation
            ignore_index: Token index to ignore (e.g., padding tokens)
        """
        self.base = base
        self.ignore_index = ignore_index
        self.cross_entropy = CrossEntropy(base=base, reduction='none')
    
    def calculate(self, pred_logits: torch.Tensor, 
                  true_tokens: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate perplexity from model predictions.
        
        Args:
            pred_logits: Model predictions [batch_size, seq_len, vocab_size]
            true_tokens: True token indices [batch_size, seq_len]
            mask: Optional mask for valid tokens [batch_size, seq_len]
            
        Returns:
            Perplexity value
        """
        # Reshape for cross-entropy calculation
        batch_size, seq_len, vocab_size = pred_logits.shape
        pred_logits_flat = pred_logits.view(-1, vocab_size)
        true_tokens_flat = true_tokens.view(-1)
        
        # Calculate cross-entropy for each token
        ce_losses = F.cross_entropy(pred_logits_flat, true_tokens_flat, 
                                  ignore_index=self.ignore_index, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            ce_losses = ce_losses * mask_flat
            valid_tokens = mask_flat.sum()
        else:
            # Count valid tokens (not ignored)
            valid_mask = (true_tokens_flat != self.ignore_index)
            ce_losses = ce_losses * valid_mask.float()
            valid_tokens = valid_mask.sum()
        
        # Calculate average cross-entropy
        avg_ce = ce_losses.sum() / valid_tokens
        
        # Convert to perplexity
        if self.base == 'e':
            return torch.exp(avg_ce)
        elif self.base == '2':
            return 2 ** avg_ce
        elif self.base == '10':
            return 10 ** avg_ce
    
    def calculate_sequence_level(self, pred_logits: torch.Tensor, 
                               true_tokens: torch.Tensor,
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate per-sequence perplexity.
        
        Returns:
            Perplexity for each sequence [batch_size]
        """
        batch_size, seq_len, vocab_size = pred_logits.shape
        
        perplexities = []
        for i in range(batch_size):
            seq_logits = pred_logits[i]  # [seq_len, vocab_size]
            seq_tokens = true_tokens[i]  # [seq_len]
            seq_mask = mask[i] if mask is not None else None
            
            # Calculate cross-entropy for this sequence
            ce_losses = F.cross_entropy(seq_logits, seq_tokens, 
                                      ignore_index=self.ignore_index, reduction='none')
            
            if seq_mask is not None:
                ce_losses = ce_losses * seq_mask
                valid_tokens = seq_mask.sum()
            else:
                valid_mask = (seq_tokens != self.ignore_index)
                ce_losses = ce_losses * valid_mask.float()
                valid_tokens = valid_mask.sum()
            
            if valid_tokens > 0:
                avg_ce = ce_losses.sum() / valid_tokens
                if self.base == 'e':
                    perplexity = torch.exp(avg_ce)
                elif self.base == '2':
                    perplexity = 2 ** avg_ce
                elif self.base == '10':
                    perplexity = 10 ** avg_ce
                perplexities.append(perplexity)
            else:
                perplexities.append(torch.tensor(float('inf')))
        
        return torch.stack(perplexities)
    
    def sliding_window_perplexity(self, pred_logits: torch.Tensor, 
                                true_tokens: torch.Tensor,
                                window_size: int = 100,
                                stride: int = 50) -> List[float]:
        """
        Calculate perplexity over sliding windows (useful for long sequences).
        
        Args:
            pred_logits: Model predictions [1, seq_len, vocab_size]
            true_tokens: True tokens [1, seq_len]
            window_size: Size of each window
            stride: Step size between windows
            
        Returns:
            List of perplexity values for each window
        """
        seq_len = pred_logits.shape[1]
        perplexities = []
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window_logits = pred_logits[:, start:end, :]
            window_tokens = true_tokens[:, start:end]
            
            ppl = self.calculate(window_logits, window_tokens)
            perplexities.append(ppl.item())
        
        return perplexities

def demonstrate_perplexity():
    """Demonstrate perplexity calculation with healthcare text examples."""
    
    ppl_calc = Perplexity(base='e')
    
    # Simulate language model evaluation scenario
    print("Perplexity Analysis - Language Model Evaluation:")
    print("-" * 55)
    
    vocab_size = 1000
    seq_len = 20
    batch_size = 4
    
    # Simulate different model quality scenarios
    
    # Scenario 1: Good model (confident, correct predictions)
    good_logits = torch.randn(batch_size, seq_len, vocab_size)
    true_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Boost the probability of correct tokens
    for b in range(batch_size):
        for s in range(seq_len):
            good_logits[b, s, true_tokens[b, s]] += 3.0
    
    # Scenario 2: Poor model (uncertain predictions)
    poor_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1  # Low confidence
    
    # Scenario 3: Random model (uniform predictions)
    random_logits = torch.zeros(batch_size, seq_len, vocab_size)
    
    # Calculate perplexities
    good_ppl = ppl_calc.calculate(good_logits, true_tokens)
    poor_ppl = ppl_calc.calculate(poor_logits, true_tokens)
    random_ppl = ppl_calc.calculate(random_logits, true_tokens)
    
    print(f"Good model perplexity:   {good_ppl:.2f}")
    print(f"Poor model perplexity:   {poor_ppl:.2f}")
    print(f"Random model perplexity: {random_ppl:.2f}")
    print(f"Theoretical maximum:     {vocab_size:.2f}")
    
    # Healthcare example: Medical text complexity
    print("\nHealthcare Example - Medical Text Complexity:")
    print("-" * 50)
    
    # Simulate different types of medical text
    medical_vocab_size = 500
    
    # Simple clinical note (low perplexity expected)
    simple_logits = torch.randn(1, 10, medical_vocab_size)
    simple_tokens = torch.randint(0, medical_vocab_size, (1, 10))
    
    # Boost common medical terms
    common_terms = [45, 67, 123, 234, 345]  # Simulate common medical terms
    for term in common_terms:
        simple_logits[0, :, term] += 2.0
    
    # Complex research paper (high perplexity expected)
    complex_logits = torch.randn(1, 10, medical_vocab_size) * 0.5
    complex_tokens = torch.randint(0, medical_vocab_size, (1, 10))
    
    simple_ppl = ppl_calc.calculate(simple_logits, simple_tokens)
    complex_ppl = ppl_calc.calculate(complex_logits, complex_tokens)
    
    print(f"Simple clinical note perplexity: {simple_ppl:.2f}")
    print(f"Complex research text perplexity: {complex_ppl:.2f}")
    print(f"Complexity ratio: {complex_ppl/simple_ppl:.1f}x")
    
    # Per-sequence analysis
    print("\nPer-Sequence Perplexity Analysis:")
    print("-" * 40)
    
    # Create batch with varying difficulty
    mixed_logits = torch.randn(3, 15, medical_vocab_size)
    mixed_tokens = torch.randint(0, medical_vocab_size, (3, 15))
    
    # Make first sequence easy, second medium, third hard
    mixed_logits[0, :, mixed_tokens[0, :]] += 4.0  # Easy
    mixed_logits[1, :, mixed_tokens[1, :]] += 1.0  # Medium
    # Third sequence unchanged (hard)
    
    seq_perplexities = ppl_calc.calculate_sequence_level(mixed_logits, mixed_tokens)
    
    for i, ppl in enumerate(seq_perplexities):
        difficulty = ["Easy", "Medium", "Hard"][i]
        print(f"Sequence {i+1} ({difficulty}): {ppl:.2f}")
    
    # Sliding window analysis for long sequences
    print("\nSliding Window Analysis (Long Sequence):")
    print("-" * 45)
    
    long_seq_len = 200
    long_logits = torch.randn(1, long_seq_len, medical_vocab_size)
    long_tokens = torch.randint(0, medical_vocab_size, (1, long_seq_len))
    
    # Make the middle section more difficult
    long_logits[0, 80:120, :] *= 0.3  # Reduce confidence in middle section
    
    window_ppls = ppl_calc.sliding_window_perplexity(
        long_logits, long_tokens, window_size=50, stride=25
    )
    
    print("Perplexity across sliding windows:")
    for i, ppl in enumerate(window_ppls):
        start_pos = i * 25
        print(f"Window {i+1} (pos {start_pos:3d}-{start_pos+50:3d}): {ppl:.2f}")
    
    print("\nNote: Higher perplexity in middle windows indicates")
    print("model difficulty with that section of the text.")


def comprehensive_healthcare_analysis():
    """
    Comprehensive analysis of healthcare text using all information theory concepts.
    This example demonstrates how these metrics work together in practice.
    """
    
    print("=" * 70)
    print("COMPREHENSIVE HEALTHCARE TEXT ANALYSIS")
    print("=" * 70)
    
    # Initialize all calculators
    ic_calc = InformationContent(base='2')
    entropy_calc = Entropy(base='2')
    ce_calc = CrossEntropy(base='e', reduction='mean')
    kl_calc = KLDivergence(base='e', reduction='batchmean')
    mi_calc = MutualInformation(base='2')
    ppl_calc = Perplexity(base='e')
    
    # Simulate a medical language model evaluation scenario
    vocab_size = 2000  # Medical vocabulary
    seq_len = 25       # Typical sentence length
    batch_size = 8     # Batch of medical texts
    
    # Create different types of medical text scenarios
    scenarios = [
        "routine_checkup",
        "emergency_case", 
        "research_paper",
        "patient_history",
        "diagnostic_report",
        "treatment_plan",
        "medication_list",
        "surgical_notes"
    ]
    
    print(f"\nAnalyzing {len(scenarios)} different medical text scenarios...")
    print(f"Vocabulary size: {vocab_size}, Sequence length: {seq_len}")
    print("-" * 70)
    
    # Generate synthetic data for each scenario
    all_logits = []
    all_tokens = []
    scenario_complexities = [0.5, 2.0, 1.8, 1.0, 1.5, 1.2, 0.8, 1.6]  # Relative complexity
    
    for i, (scenario, complexity) in enumerate(zip(scenarios, scenario_complexities)):
        # Generate logits with varying uncertainty based on complexity
        logits = torch.randn(1, seq_len, vocab_size) / complexity
        tokens = torch.randint(0, vocab_size, (1, seq_len))
        
        # Boost correct token probabilities (simulate model performance)
        for s in range(seq_len):
            logits[0, s, tokens[0, s]] += 2.0 / complexity
        
        all_logits.append(logits)
        all_tokens.append(tokens)
    
    # Combine all scenarios
    combined_logits = torch.cat(all_logits, dim=0)
    combined_tokens = torch.cat(all_tokens, dim=0)
    
    # 1. PERPLEXITY ANALYSIS
    print("1. PERPLEXITY ANALYSIS")
    print("-" * 30)
    
    overall_ppl = ppl_calc.calculate(combined_logits, combined_tokens)
    seq_ppls = ppl_calc.calculate_sequence_level(combined_logits, combined_tokens)
    
    print(f"Overall perplexity: {overall_ppl:.2f}")
    print("\nPer-scenario perplexity:")
    
    for i, (scenario, ppl) in enumerate(zip(scenarios, seq_ppls)):
        print(f"  {scenario:15}: {ppl:.2f}")
    
    # 2. ENTROPY ANALYSIS
    print(f"\n2. ENTROPY ANALYSIS")
    print("-" * 25)
    
    # Calculate entropy for each position in sequences
    probs = F.softmax(combined_logits, dim=-1)
    position_entropies = entropy_calc.calculate(probs, dim=-1)  # [batch, seq_len]
    
    avg_entropy_per_scenario = position_entropies.mean(dim=1)
    
    print("Average entropy per scenario (bits):")
    for i, (scenario, entropy) in enumerate(zip(scenarios, avg_entropy_per_scenario)):
        print(f"  {scenario:15}: {entropy:.3f}")
    
    # 3. CROSS-ENTROPY ANALYSIS
    print(f"\n3. CROSS-ENTROPY ANALYSIS")
    print("-" * 30)
    
    # Compare against a baseline uniform distribution
    uniform_logits = torch.zeros_like(combined_logits)
    
    model_ce = ce_calc.sparse_cross_entropy(combined_tokens.flatten(), 
                                          combined_logits.view(-1, vocab_size))
    uniform_ce = ce_calc.sparse_cross_entropy(combined_tokens.flatten(),
                                            uniform_logits.view(-1, vocab_size))
    
    print(f"Model cross-entropy:   {model_ce:.4f}")
    print(f"Uniform cross-entropy: {uniform_ce:.4f}")
    print(f"Improvement ratio:     {uniform_ce/model_ce:.2f}x")
    
    # 4. KL DIVERGENCE ANALYSIS
    print(f"\n4. KL DIVERGENCE ANALYSIS")
    print("-" * 30)
    
    # Compare different scenarios to routine checkup (baseline)
    baseline_probs = F.softmax(all_logits[0], dim=-1)  # routine_checkup
    
    print("KL divergence from routine checkup:")
    for i, scenario in enumerate(scenarios[1:], 1):
        scenario_probs = F.softmax(all_logits[i], dim=-1)
        kl_div = kl_calc.calculate(baseline_probs, scenario_probs)
        print(f"  {scenario:15}: {kl_div:.4f}")
    
    # 5. MUTUAL INFORMATION ANALYSIS
    print(f"\n5. MUTUAL INFORMATION ANALYSIS")
    print("-" * 35)
    
    # Analyze relationship between position in sequence and entropy
    positions = torch.arange(seq_len).float().repeat(batch_size)
    entropies_flat = position_entropies.flatten()
    
    mi_pos_entropy = mi_calc.calculate_from_samples(positions, entropies_flat, bins=10)
    print(f"MI(Position, Entropy): {mi_pos_entropy:.4f} bits")
    
    # Analyze relationship between scenario complexity and average entropy
    complexities = torch.tensor(scenario_complexities)
    avg_entropies = avg_entropy_per_scenario
    
    mi_complexity_entropy = mi_calc.calculate_from_samples(complexities, avg_entropies, bins=5)
    print(f"MI(Complexity, Entropy): {mi_complexity_entropy:.4f} bits")
    
    # 6. INFORMATION CONTENT ANALYSIS
    print(f"\n6. INFORMATION CONTENT ANALYSIS")
    print("-" * 35)
    
    # Find most and least informative tokens
    token_probs = F.softmax(combined_logits, dim=-1)
    token_ic = ic_calc.calculate(token_probs)
    
    # Get statistics
    max_ic_per_seq = token_ic.max(dim=1)[0]
    min_ic_per_seq = token_ic.min(dim=1)[0]
    avg_ic_per_seq = token_ic.mean(dim=1)
    
    print("Information content statistics per scenario:")
    print("Scenario         | Avg IC | Max IC | Min IC")
    print("-" * 45)
    for i, scenario in enumerate(scenarios):
        print(f"{scenario:15} | {avg_ic_per_seq[i].item():6.2f} | {max_ic_per_seq[i].item():6.2f} | {min_ic_per_seq[i].item():6.2f}")
    
    # 7. SUMMARY AND INSIGHTS
    print(f"\n7. SUMMARY AND INSIGHTS")
    print("-" * 30)
    
    print("Key findings:")
    
    # Find most challenging scenario
    most_challenging_idx = seq_ppls.argmax()
    most_challenging = scenarios[most_challenging_idx]
    print(f"• Most challenging scenario: {most_challenging} (PPL: {seq_ppls[most_challenging_idx]:.2f})")
    
    # Find most predictable scenario
    most_predictable_idx = seq_ppls.argmin()
    most_predictable = scenarios[most_predictable_idx]
    print(f"• Most predictable scenario: {most_predictable} (PPL: {seq_ppls[most_predictable_idx]:.2f})")
    
    # Entropy insights
    high_entropy_scenarios = [scenarios[i] for i in range(len(scenarios)) 
                            if avg_entropy_per_scenario[i] > avg_entropy_per_scenario.mean()]
    print(f"• High entropy scenarios: {', '.join(high_entropy_scenarios)}")
    
    # Model performance insights
    if overall_ppl < 50:
        performance = "excellent"
    elif overall_ppl < 100:
        performance = "good"
    elif overall_ppl < 200:
        performance = "fair"
    else:
        performance = "poor"
    
    print(f"• Overall model performance: {performance} (PPL: {overall_ppl:.2f})")
    
    print(f"\nRecommendations:")
    if mi_pos_entropy > 0.1:
        print("• Position affects entropy - consider positional encoding improvements")
    if mi_complexity_entropy > 0.2:
        print("• Strong complexity-entropy correlation - model adapts well to text difficulty")
    
    high_kl_scenarios = [scenarios[i+1] for i in range(len(scenarios)-1) 
                        if i < len(all_logits)-1]  # Simplified for demo
    if len(high_kl_scenarios) > 0:
        print(f"• Consider domain-specific fine-tuning for: {', '.join(high_kl_scenarios[:2])}")

if __name__ == "__main__":
    """Main execution block for demonstrations."""

    print("INFORMATION THEORY FOR LARGE LANGUAGE MODELS")
    print("=" * 70)
    print("Comprehensive demonstrations of information theory concepts")
    print("with focus on healthcare applications and LLM development")
    print("=" * 70)

    # Run unified calculator demonstration
    print("\n🔧 UNIFIED CALCULATOR DEMONSTRATION")
    demonstrate_unified_calculator()

    # Run healthcare-specific demonstrations
    print("\n HEALTHCARE APPLICATIONS")
    healthcare_demo = HealthcareInformationTheoryDemo(base=LogarithmBase.BINARY)
    healthcare_demo.run_all_demonstrations()

    print("\n ALL DEMONSTRATIONS COMPLETED")
    print("=" * 70)
    print("For production use, import the classes directly:")
    print("from information_theory import InformationTheoryCalculator")
    print("calc = InformationTheoryCalculator(base='2')")
    print("=" * 70)