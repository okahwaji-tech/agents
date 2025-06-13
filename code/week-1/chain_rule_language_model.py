"""
chain_rule_language_model.py

Implementation of autoregressive language modeling using the chain rule of probability,
with support for LSTM, GRU, and Transformer architectures and healthcare-specific examples.

Defines:
  - ChainRuleLanguageModel: Core model class
  - Training and evaluation utilities for sequence probability, generation, and analysis
  - Demonstration functions for chain rule applications in healthcare contexts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math
import time


class ChainRuleLanguageModel(nn.Module):
    """
    Autoregressive language model leveraging the chain rule for next-token prediction.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        embedding_dim (int): Embedding dimensionality.
        hidden_dim (int): Hidden state dimensionality for RNNs or FF dim for Transformers.
        num_layers (int): Number of encoder layers.
        model_type (str): One of "lstm", "gru", or "transformer".

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        position_embedding (nn.Embedding, optional): Positional embeddings (Transformer only).
        encoder (nn.Module): RNN or Transformer encoder.
        output_projection (nn.Linear): Projects hidden states to vocabulary logits.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, 
                 model_type: str = "lstm"):
        """
        Initialize model parameters and submodules.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Choose architecture
        if model_type == "lstm":
            self.encoder = nn.LSTM(
                embedding_dim, hidden_dim, num_layers, 
                batch_first=True, dropout=0.1
            )
        elif model_type == "gru":
            self.encoder = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=0.1
            )
        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=8, 
                dim_feedforward=hidden_dim, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            # For transformer, we need position embeddings
            self.position_embedding = nn.Embedding(1000, embedding_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Output projection layer
        if model_type == "transformer":
            self.output_projection = nn.Linear(embedding_dim, vocab_size)
        else:
            self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """
        Apply custom initialization to Linear and Embedding parameters.
        Linear weights: normal(mean=0.0, std=0.02); biases: zero.
        Embedding weights: normal(mean=0.0, std=0.02).
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, input_ids: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits and hidden states for input sequences.

        Args:
            input_ids (Tensor): Shape (batch_size, seq_len).
            hidden (Tensor, optional): Previous hidden state for RNNs.

        Returns:
            Tuple[Tensor, Tensor]: (logits, hidden). Logits shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        embeddings = self.embedding(input_ids)
        
        if self.model_type == "transformer":
            # Add position embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            pos_embeddings = self.position_embedding(positions)
            embeddings = embeddings + pos_embeddings
            
            # Create causal mask to prevent looking at future tokens
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(input_ids.device)
            
            # Apply transformer encoder
            encoded = self.encoder(embeddings, src_mask=mask)
            hidden = None  # Transformers don't use hidden states
        else:
            # Apply RNN encoder
            encoded, hidden = self.encoder(embeddings, hidden)
        
        # Project to vocabulary size
        logits = self.output_projection(encoded)
        
        return logits, hidden
    
    def compute_sequence_probability(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probabilities of input sequences via the chain rule.

        Args:
            sequence (Tensor): Shape (batch_size, seq_len) of token indices.

        Returns:
            Tensor: Shape (batch_size,) of sequence log-probabilities.
        """
        batch_size, seq_len = sequence.shape
        total_log_prob = torch.zeros(batch_size, device=sequence.device)
        
        hidden = None
        
        for i in range(seq_len):
            if i == 0:
                # For the first token, use uniform distribution or a learned prior
                log_prob = torch.log(torch.tensor(1.0 / self.vocab_size))
                total_log_prob += log_prob
            else:
                # Context is everything up to position i
                context = sequence[:, :i]
                target = sequence[:, i]
                
                # Get model predictions
                logits, hidden = self.forward(context, hidden)
                
                # Extract logits for the last position (next token prediction)
                next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
                
                # Convert to probabilities
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Extract log probability of the target token
                target_log_prob = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                total_log_prob += target_log_prob
        
        return total_log_prob
    
    def generate_sequence(self, start_token: int, max_length: int, 
                         temperature: float = 1.0, top_k: Optional[int] = None,
                         top_p: Optional[float] = None) -> List[int]:
        """
        Autoregressively generate a token sequence.

        Args:
            start_token (int): Initial token ID.
            max_length (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): Limits sampling to top-k tokens.
            top_p (float, optional): Nucleus sampling threshold.

        Returns:
            List[int]: Generated token IDs.
        """
        self.eval()
        generated = [start_token]
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Current sequence
                current_seq = torch.tensor([generated]).unsqueeze(0)  # Add batch dim
                
                # Get next token probabilities
                logits, hidden = self.forward(current_seq, hidden)
                next_token_logits = logits[0, -1, :]  # Last position, remove batch dim
                
                # Apply temperature scaling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Apply top-k sampling
                if top_k is not None:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    # Zero out probabilities not in top-k
                    mask = torch.zeros_like(probs)
                    mask.scatter_(0, top_k_indices, 1)
                    probs = probs * mask
                    probs = probs / probs.sum()  # Renormalize
                
                # Apply nucleus (top-p) sampling
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    
                    # Find cutoff point
                    cutoff_mask = cumulative_probs <= top_p
                    cutoff_mask[0] = True  # Always include at least one token
                    
                    # Zero out probabilities beyond cutoff
                    sorted_probs[~cutoff_mask] = 0
                    probs = torch.zeros_like(probs)
                    probs.scatter_(0, sorted_indices, sorted_probs)
                    probs = probs / probs.sum()  # Renormalize
                
                # Sample next token
                categorical = Categorical(probs)
                next_token = categorical.sample().item()
                generated.append(next_token)
                
                # Stop if we hit an end token (assuming 0 is end token)
                if next_token == 0:
                    break
        
        return generated
    
    def compute_perplexity(self, sequences: torch.Tensor) -> float:
        """
        Evaluate model perplexity on a batch of sequences.

        Args:
            sequences (Tensor): Shape (batch_size, seq_len).

        Returns:
            float: Computed perplexity.
        """
        self.eval()
        total_log_prob = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            batch_size, seq_len = sequences.shape
            
            for i in range(1, seq_len):  # Start from position 1
                context = sequences[:, :i]
                targets = sequences[:, i]
                
                logits, _ = self.forward(context)
                next_token_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                total_log_prob += target_log_probs.sum().item()
                total_tokens += batch_size
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity


def demonstrate_chain_rule_concepts():
    """
    Demonstrate chain rule concepts with practical examples.

    Returns:
        None
    """
    print("=== Chain Rule Language Modeling Demonstration ===")
    
    # Initialize model
    vocab_size = 100
    model = ChainRuleLanguageModel(vocab_size, model_type="lstm")
    
    # Create sample sequences
    batch_size = 4
    seq_len = 10
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print(f"Model vocabulary size: {vocab_size}")
    print(f"Sample sequences shape: {sequences.shape}")
    
    # Compute sequence probabilities using chain rule
    log_probs = model.compute_sequence_probability(sequences)
    print(f"Sequence log probabilities: {log_probs}")
    print(f"Sequence probabilities: {torch.exp(log_probs)}")
    
    # Demonstrate the chain rule decomposition
    print("\n=== Chain Rule Decomposition ===")
    
    # Take the first sequence for detailed analysis
    seq = sequences[0]
    print(f"Analyzing sequence: {seq.tolist()}")
    
    # Compute probability step by step
    total_log_prob = 0.0
    hidden = None
    
    for i in range(seq_len):
        if i == 0:
            # First token probability (uniform prior)
            token_log_prob = math.log(1.0 / vocab_size)
            print(f"P(w_{i+1}={seq[i].item()}) = {math.exp(token_log_prob):.6f}")
        else:
            # Conditional probability
            context = seq[:i].unsqueeze(0)  # Add batch dimension
            target = seq[i]
            
            with torch.no_grad():
                logits, hidden = model.forward(context, hidden)
                next_token_logits = logits[0, -1, :]
                log_probs_dist = F.log_softmax(next_token_logits, dim=-1)
                token_log_prob = log_probs_dist[target].item()
            
            context_str = [str(x.item()) for x in seq[:i]]
            print(f"P(w_{i+1}={seq[i].item()}|{','.join(context_str)}) = {math.exp(token_log_prob):.6f}")
        
        total_log_prob += token_log_prob
    
    print(f"Total log probability: {total_log_prob:.6f}")
    print(f"Total probability: {math.exp(total_log_prob):.10f}")


def medical_chain_rule_example():
    """
    Demonstrate chain rule application to medical text generation.

    Returns:
        None
    """
    print("\n=== Medical Text Chain Rule Example ===")

    # Define medical vocabulary
    medical_vocab = {
        0: "<END>", 1: "patient", 2: "presents", 3: "with", 4: "chest", 5: "pain",
        6: "shortness", 7: "of", 8: "breath", 9: "history", 10: "hypertension",
        11: "diabetes", 12: "examination", 13: "reveals", 14: "normal", 15: "heart",
        16: "rate", 17: "blood", 18: "pressure", 19: "elevated"
    }

    vocab_size = len(medical_vocab)
    model = ChainRuleLanguageModel(vocab_size, model_type="transformer")

    # Create a medical sequence: "patient presents with chest pain"
    medical_sequence = torch.tensor([[1, 2, 3, 4, 5]])  # Add batch dimension

    print("Medical sequence:", [medical_vocab[i.item()] for i in medical_sequence[0]])

    # Compute probability using chain rule
    log_prob = model.compute_sequence_probability(medical_sequence)
    print(f"Sequence log probability: {log_prob.item():.6f}")

    # Generate continuation
    print("\n=== Medical Text Generation ===")
    start_sequence = [1, 2, 3]  # "patient presents with"
    print("Starting with:", [medical_vocab[i] for i in start_sequence])

    # Generate continuation
    generated = model.generate_sequence(
        start_token=1, max_length=10, temperature=0.8, top_k=5
    )

    print("Generated sequence:", [medical_vocab.get(i, f"UNK_{i}") for i in generated])


def analyze_conditional_dependencies():
    """
    Analyze how conditional dependencies change throughout a sequence.

    Returns:
        None
    """
    print("\n=== Conditional Dependency Analysis ===")

    vocab_size = 50
    model = ChainRuleLanguageModel(vocab_size, model_type="lstm")

    # Create a test sequence
    test_sequence = torch.randint(1, vocab_size, (1, 15))

    print("Analyzing conditional dependencies in sequence generation...")

    # Analyze how entropy changes with context length
    entropies = []
    hidden = None

    with torch.no_grad():
        for i in range(1, len(test_sequence[0])):
            context = test_sequence[:, :i]
            logits, hidden = model.forward(context, hidden)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

            # Compute entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)

    print("Entropy by context length:")
    for i, entropy in enumerate(entropies):
        print(f"Context length {i+1}: entropy = {entropy:.3f}")

    # Analyze the trend
    if len(entropies) > 1:
        entropy_change = entropies[-1] - entropies[0]
        if entropy_change < 0:
            print("Entropy decreases with longer context (more predictable)")
        else:
            print("Entropy increases with longer context (less predictable)")


def compare_model_architectures():
    """
    Compare different architectures for implementing the chain rule.

    Returns:
        None
    """
    print("\n=== Architecture Comparison ===")

    vocab_size = 100
    seq_len = 20
    batch_size = 8

    # Create test data
    test_sequences = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Initialize different models
    models = {
        "LSTM": ChainRuleLanguageModel(vocab_size, model_type="lstm"),
        "GRU": ChainRuleLanguageModel(vocab_size, model_type="gru"),
        "Transformer": ChainRuleLanguageModel(vocab_size, model_type="transformer")
    }

    print("Comparing model architectures on chain rule implementation:")

    for name, model in models.items():
        # Compute perplexity
        perplexity = model.compute_perplexity(test_sequences)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        print(f"{name}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Perplexity: {perplexity:.2f}")

        # Test generation speed (simplified timing)
        start_time = time.time()
        generated = model.generate_sequence(1, max_length=10)
        generation_time = time.time() - start_time
        print(f"  Generation time: {generation_time:.3f}s")
        print()


def healthcare_sequence_probability():
    """
    Demonstrate sequence probability calculation for healthcare scenarios.

    Returns:
        None
    """
    print("\n=== Healthcare Sequence Probability Analysis ===")

    # Simulate clinical note structure with probabilities
    clinical_vocab = {
        0: "<END>", 1: "patient", 2: "presents", 3: "with", 4: "acute", 5: "chest",
        6: "pain", 7: "and", 8: "shortness", 9: "of", 10: "breath", 11: "history",
        12: "includes", 13: "hypertension", 14: "diabetes", 15: "physical",
        16: "examination", 17: "shows", 18: "elevated", 19: "blood", 20: "pressure"
    }

    vocab_size = len(clinical_vocab)
    model = ChainRuleLanguageModel(vocab_size, model_type="transformer")

    # Define several clinical scenarios
    scenarios = [
        [1, 2, 3, 4, 5, 6],           # "patient presents with acute chest pain"
        [1, 2, 3, 8, 9, 10],          # "patient presents with shortness of breath"
        [11, 12, 13, 7, 14],          # "history includes hypertension and diabetes"
        [15, 16, 17, 18, 19, 20]      # "physical examination shows elevated blood pressure"
    ]

    print("Clinical scenario probability analysis:")

    for i, scenario in enumerate(scenarios):
        scenario_tensor = torch.tensor([scenario])
        log_prob = model.compute_sequence_probability(scenario_tensor)
        prob = torch.exp(log_prob).item()

        scenario_text = " ".join([clinical_vocab[token] for token in scenario])
        print(f"Scenario {i+1}: '{scenario_text}'")
        print(f"  Log probability: {log_prob.item():.6f}")
        print(f"  Probability: {prob:.10f}")
        print()

    # Analyze how probability changes with sequence length
    print("Probability vs. sequence length analysis:")
    base_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Longer clinical description

    for length in range(2, len(base_sequence) + 1):
        partial_sequence = torch.tensor([base_sequence[:length]])
        log_prob = model.compute_sequence_probability(partial_sequence)
        prob = torch.exp(log_prob).item()

        sequence_text = " ".join([clinical_vocab[token] for token in base_sequence[:length]])
        print(f"Length {length}: {prob:.10f} - '{sequence_text}'")


if __name__ == "__main__":
    # Run all demonstrations
    for demo in (
        demonstrate_chain_rule_concepts,
        medical_chain_rule_example,
        analyze_conditional_dependencies,
        compare_model_architectures,
        healthcare_sequence_probability,
    ):
        demo()
