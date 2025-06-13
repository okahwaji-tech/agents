"""
autoregressive_lm_healthcare.py

Comprehensive PyTorch implementation of an autoregressive language model
focused on healthcare applications and chain-rule probability modeling.

This module defines:
  - HealthcareAutoregressiveLM: Transformer-based autoregressive LM
  - HealthcareLMTrainer: Trainer with teacher forcing
  - Utility functions for demonstration and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import math
import json

class HealthcareAutoregressiveLM(nn.Module):
    """
    Transformer-based autoregressive language model for healthcare text.

    Args:
        vocab_size (int): Size of the token vocabulary.
        embedding_dim (int): Dimension of token and position embeddings.
        hidden_dim (int): Feed-forward network hidden size.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads per layer.
        max_seq_len (int): Maximum sequence length for position embeddings.
        dropout (float): Dropout probability.

    Attributes:
        token_embedding (nn.Embedding): Token embedding lookup.
        position_embedding (nn.Embedding): Positional embedding lookup.
        transformer_layers (ModuleList): List of TransformerEncoderLayer modules.
        output_projection (nn.Linear): Final projection to vocabulary logits.
        medical_vocab (Dict[int, str]): Demonstration medical vocabulary.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        """
        Initialize the autoregressive LM components and weights.

        See class docstring for argument descriptions.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Medical vocabulary for demonstrations
        self.medical_vocab = self._create_medical_vocabulary()
    
    def _init_weights(self):
        """
        Initialize weights for Linear and Embedding modules.

        Linear weights: normal(mean=0.0, std=0.02)
        Biases: zeros
        Embedding weights: normal(mean=0.0, std=0.02)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_medical_vocabulary(self) -> Dict[int, str]:
        """
        Build a demonstration medical vocabulary mapping indices to terms.

        Returns:
            Dict[int, str]: Index-to-token mapping for medical terms.
        """
        medical_terms = [
            # Basic tokens
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            
            # Common medical terms
            "patient", "diagnosis", "treatment", "therapy", "medication",
            "symptoms", "examination", "history", "procedure", "surgery",
            
            # Anatomy
            "heart", "lung", "brain", "liver", "kidney", "chest", "abdomen",
            "head", "neck", "back", "arm", "leg", "hand", "foot",
            
            # Conditions
            "diabetes", "hypertension", "pneumonia", "asthma", "arthritis",
            "cancer", "infection", "inflammation", "fracture", "stroke",
            
            # Symptoms
            "pain", "fever", "cough", "nausea", "vomiting", "dizziness",
            "fatigue", "weakness", "swelling", "bleeding", "rash",
            
            # Descriptors
            "acute", "chronic", "severe", "mild", "moderate", "normal",
            "abnormal", "elevated", "decreased", "stable", "improving",
            
            # Actions/Procedures
            "presents", "reports", "denies", "shows", "reveals", "indicates",
            "administered", "prescribed", "performed", "scheduled",
            
            # Measurements
            "blood", "pressure", "temperature", "pulse", "rate", "level",
            "count", "result", "value", "reading", "measurement",
            
            # Time/Frequency
            "daily", "weekly", "monthly", "hours", "days", "weeks", "months",
            "morning", "evening", "night", "before", "after", "during",
            
            # Connectors
            "and", "or", "but", "with", "without", "of", "in", "on", "at",
            "to", "from", "for", "by", "as", "is", "was", "has", "had"
        ]
        
        return {i: term for i, term in enumerate(medical_terms)}
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the autoregressive LM.

        Args:
            input_ids (Tensor): Shape (batch_size, seq_len) input token indices.
            attention_mask (Tensor, optional): Shape (batch_size, seq_len) mask.

        Returns:
            Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        embeddings = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def compute_sequence_log_probability(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of sequences via the chain rule.

        Args:
            sequence (Tensor): Shape (batch_size, seq_len) input sequences.

        Returns:
            Tensor: Shape (batch_size,) log-probabilities for each sequence.
        """
        batch_size, seq_len = sequence.shape
        
        if seq_len == 1:
            # Single token - use uniform prior
            return torch.full((batch_size,), -math.log(self.vocab_size))
        
        # Get model predictions for all positions
        logits = self.forward(sequence[:, :-1])  # Exclude last token from input
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for target tokens
        targets = sequence[:, 1:]  # Exclude first token from targets
        target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        
        # Sum log probabilities (chain rule in log space)
        sequence_log_prob = target_log_probs.sum(dim=1)
        
        # Add log probability of first token (uniform prior)
        first_token_log_prob = -math.log(self.vocab_size)
        sequence_log_prob += first_token_log_prob
        
        return sequence_log_prob
    
    def generate_text(self, 
                     prompt: Union[str, List[int]], 
                     max_length: int = 50,
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     do_sample: bool = True) -> List[int]:
        """
        Generate text using autoregressive sampling.
        
        Args:
            prompt: Starting prompt (string or token IDs)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            generated_tokens: List of generated token IDs
        """
        self.eval()
        
        # Convert prompt to token IDs if string
        prompt_tokens = self._simple_tokenize(prompt) if isinstance(prompt, str) else prompt
        
        generated = prompt_tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length - len(prompt_tokens)):
                # Current sequence
                current_seq = torch.tensor([generated])
                
                # Get next token logits
                logits = self.forward(current_seq)
                next_token_logits = logits[0, -1, :]  # Last position
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Apply sampling strategies
                if top_k is not None:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    mask = torch.zeros_like(probs)
                    mask.scatter_(0, top_k_indices, 1)
                    probs = probs * mask
                    probs = probs / probs.sum()
                
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    cutoff_mask = cumulative_probs <= top_p
                    cutoff_mask[0] = True  # Always include top token
                    sorted_probs[~cutoff_mask] = 0
                    probs = torch.zeros_like(probs)
                    probs.scatter_(0, sorted_indices, sorted_probs)
                    probs = probs / probs.sum()
                
                # Sample or select next token
                if do_sample:
                    categorical = Categorical(probs)
                    next_token = categorical.sample().item()
                else:
                    next_token = probs.argmax().item()
                
                generated.append(next_token)
                
                # Stop if EOS token
                if next_token == 3:  # EOS token
                    break
        
        return generated
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """
        Simple whitespace tokenizer mapping to medical vocabulary IDs.

        Args:
            text (str): Raw input text.

        Returns:
            List[int]: Token ID sequence where unknown words map to the UNK token.
        """
        # Convert text to lowercase and split
        words = text.lower().split()
        
        # Map to token IDs
        token_ids = []
        reverse_vocab = {v: k for k, v in self.medical_vocab.items()}
        
        for word in words:
            if word in reverse_vocab:
                token_ids.append(reverse_vocab[word])
            else:
                token_ids.append(1)  # UNK token
        
        return token_ids
    
    def analyze_attention_patterns(self, sequence: torch.Tensor) -> Dict:
        """
        Analyze self-attention distributions across model layers.

        Args:
            sequence (Tensor): Shape (batch_size, seq_len) input tokens.

        Returns:
            Dict[str, Any]: Statistics including attention entropy and concentration.
        """
        self.eval()
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            # Extract attention weights from transformer layer
            # This is a simplified version - actual implementation depends on PyTorch version
            if hasattr(module, 'self_attn'):
                attention_weights.append(output[1] if len(output) > 1 else None)
        
        # Register hooks
        hooks = []
        for layer in self.transformer_layers:
            hook = layer.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = self.forward(sequence)
            
            # Analyze attention patterns
            analysis = {
                'num_layers': len(self.transformer_layers),
                'sequence_length': sequence.shape[1],
                'attention_entropy': [],
                'attention_concentration': []
            }
            
            # Compute attention statistics for each layer
            for layer_idx, attn_weights in enumerate(attention_weights):
                if attn_weights is not None:
                    # Compute entropy of attention distributions
                    entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
                    analysis['attention_entropy'].append(entropy.mean().item())
                    
                    # Compute concentration (max attention weight)
                    concentration = attn_weights.max(dim=-1)[0]
                    analysis['attention_concentration'].append(concentration.mean().item())
        
        finally:
            [hook.remove() for hook in hooks]
        
        return analysis
    
    def compute_perplexity(self, sequences: torch.Tensor) -> float:
        """
        Compute perplexity for a batch of sequences.

        Args:
            sequences (Tensor): Shape (batch_size, seq_len) input tokens.

        Returns:
            float: Perplexity value.
        """
        self.eval()
        
        with torch.no_grad():
            # Compute log probabilities
            log_probs = self.compute_sequence_log_probability(sequences)
            
            # Compute average log probability per token
            total_tokens = (sequences.shape[1] - 1) * sequences.shape[0]  # Exclude first token
            avg_log_prob = log_probs.sum().item() / total_tokens
            
            # Convert to perplexity
            perplexity = math.exp(-avg_log_prob)
        
        return perplexity

class HealthcareLMTrainer:
    """
    Trainer for HealthcareAutoregressiveLM using teacher forcing.

    Args:
        model (HealthcareAutoregressiveLM): Model to train.
        learning_rate (float): Optimizer learning rate.

    Attributes:
        optimizer (Optimizer): AdamW optimizer.
        criterion (nn.CrossEntropyLoss): Loss function ignoring padding token.
    """
    
    def __init__(self, model: HealthcareAutoregressiveLM, learning_rate: float = 1e-4):
        """
        Initialize trainer with model, optimizer, and loss.
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform one training step using teacher forcing.
        
        Args:
            batch: Batch of sequences of shape (batch_size, seq_len)
            
        Returns:
            loss: Training loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare inputs and targets for teacher forcing
        inputs = batch[:, :-1]  # All tokens except last
        targets = batch[:, 1:]  # All tokens except first
        
        # Forward pass
        logits = self.model(inputs)
        
        # Compute loss (chain rule in practice)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def demonstrate_chain_rule_calculation():
    """
    Demonstrate explicit chain rule calculation for medical sequences.
    
    Returns:
        None
    """
    print("=== Chain Rule Calculation Demonstration ===")
    
    # Create model
    vocab_size = 100
    model = HealthcareAutoregressiveLM(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
    
    # Medical sequence: "patient presents with chest pain"
    # Using simplified token IDs
    medical_sequence = torch.tensor([[4, 45, 67, 11, 20]])  # Shape: (1, 5)
    
    print(f"Medical sequence shape: {medical_sequence.shape}")
    print(f"Sequence tokens: {medical_sequence[0].tolist()}")
    
    # Compute sequence probability using chain rule
    log_prob = model.compute_sequence_log_probability(medical_sequence)
    prob = torch.exp(log_prob)
    
    print(f"\nChain rule calculation:")
    print(f"  Log probability: {log_prob.item():.6f}")
    print(f"  Probability: {prob.item():.10f}")
    
    # Step-by-step breakdown
    print(f"\nStep-by-step breakdown:")
    
    # P(w1) - uniform prior
    first_token_log_prob = -math.log(vocab_size)
    print(f"  P(w1={medical_sequence[0, 0].item()}) = {math.exp(first_token_log_prob):.6f}")
    
    # Conditional probabilities P(wi | w1, ..., wi-1)
    total_log_prob = first_token_log_prob
    
    for i in range(1, medical_sequence.shape[1]):
        context = medical_sequence[:, :i]
        target = medical_sequence[:, i]
        
        with torch.no_grad():
            logits = model(context)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            token_log_prob = log_probs[0, target.item()].item()
        
        total_log_prob += token_log_prob
        context_tokens = medical_sequence[0, :i].tolist()
        
        print(f"  P(w{i+1}={target.item()}|{context_tokens}) = {math.exp(token_log_prob):.6f}")
    
    print(f"\nTotal log probability: {total_log_prob:.6f}")
    print(f"Total probability: {math.exp(total_log_prob):.10f}")

def demonstrate_medical_text_generation():
    """
    Demonstrate medical text generation using autoregressive sampling.
    
    Returns:
        None
    """
    print("\n=== Medical Text Generation ===")
    
    # Create model
    vocab_size = len(HealthcareAutoregressiveLM(100).medical_vocab)
    model = HealthcareAutoregressiveLM(vocab_size, embedding_dim=128, hidden_dim=256)
    
    # Medical prompts
    prompts = [
        "patient presents with",
        "history of",
        "examination reveals",
        "diagnosis is",
        "treatment includes"
    ]
    
    print(f"Generating medical text with different sampling strategies:")
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Greedy generation
        greedy_tokens = model.generate_text(prompt, max_length=10, do_sample=False)
        greedy_text = " ".join([model.medical_vocab.get(t, f"UNK_{t}") for t in greedy_tokens])
        print(f"  Greedy: {greedy_text}")
        
        # Random sampling
        random_tokens = model.generate_text(prompt, max_length=10, temperature=1.0)
        random_text = " ".join([model.medical_vocab.get(t, f"UNK_{t}") for t in random_tokens])
        print(f"  Random: {random_text}")
        
        # Top-k sampling
        topk_tokens = model.generate_text(prompt, max_length=10, top_k=10)
        topk_text = " ".join([model.medical_vocab.get(t, f"UNK_{t}") for t in topk_tokens])
        print(f"  Top-k: {topk_text}")

def analyze_medical_sequence_probabilities():
    """
    Analyze probabilities of different medical sequences.
    
    Returns:
        None
    """
    print("\n=== Medical Sequence Probability Analysis ===")
    
    # Create model
    vocab_size = 100
    model = HealthcareAutoregressiveLM(vocab_size, embedding_dim=128, hidden_dim=256)
    
    # Different medical sequences with varying plausibility
    sequences = [
        [4, 45, 67, 11, 20],      # "patient presents with chest pain" - plausible
        [4, 45, 67, 15, 30],      # "patient presents with head fever" - less plausible
        [20, 11, 67, 45, 4],      # "pain chest with presents patient" - implausible order
        [4, 50, 25, 35, 40],      # Random medical terms
        [4, 4, 4, 4, 4]           # Repetitive sequence
    ]
    
    sequence_descriptions = [
        "Plausible medical sequence",
        "Less plausible combination",
        "Implausible word order",
        "Random medical terms",
        "Repetitive sequence"
    ]
    
    print(f"Comparing sequence probabilities:")
    
    for seq, desc in zip(sequences, sequence_descriptions):
        seq_tensor = torch.tensor([seq])
        log_prob = model.compute_sequence_log_probability(seq_tensor)
        prob = torch.exp(log_prob)
        perplexity = math.exp(-log_prob.item() / len(seq))
        
        print(f"\n{desc}:")
        print(f"  Sequence: {seq}")
        print(f"  Log probability: {log_prob.item():.6f}")
        print(f"  Probability: {prob.item():.10f}")
        print(f"  Perplexity: {perplexity:.3f}")

def demonstrate_temperature_effects():
    """
    Demonstrate how temperature affects generation diversity.
    
    Returns:
        None
    """
    print("\n=== Temperature Effects on Generation ===")
    
    # Create model
    vocab_size = 100
    model = HealthcareAutoregressiveLM(vocab_size, embedding_dim=128, hidden_dim=256)
    
    prompt = "patient diagnosis"
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    print(f"Prompt: '{prompt}'")
    print(f"Generating with different temperatures:")
    
    for temp in temperatures:
        print(f"\nTemperature = {temp}:")
        
        # Generate multiple samples to show diversity
        for i in range(3):
            tokens = model.generate_text(prompt, max_length=8, temperature=temp)
            text = " ".join([model.medical_vocab.get(t, f"UNK_{t}") for t in tokens])
            print(f"  Sample {i+1}: {text}")

def medical_language_model_training_demo():
    """
    Demonstrate training a medical language model with chain rule optimization.
    
    Returns:
        None
    """
    print("\n=== Medical Language Model Training Demo ===")
    
    # Create model and trainer
    vocab_size = 100
    model = HealthcareAutoregressiveLM(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
    trainer = HealthcareLMTrainer(model, learning_rate=1e-3)
    
    # Generate synthetic medical training data
    batch_size = 8
    seq_len = 20
    num_batches = 10
    
    print(f"Training configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of batches: {num_batches}")
    
    # Training loop
    losses = []
    
    for batch_idx in range(num_batches):
        # Generate synthetic batch (random medical sequences)
        batch = torch.randint(1, vocab_size, (batch_size, seq_len))
        
        # Training step
        loss = trainer.train_step(batch)
        losses.append(loss)
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}: loss = {loss:.4f}")
    
    print(f"\nTraining completed!")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")
    
    # Evaluate perplexity
    test_batch = torch.randint(1, vocab_size, (batch_size, seq_len))
    perplexity = model.compute_perplexity(test_batch)
    print(f"  Test perplexity: {perplexity:.2f}")

def main():
    """
    Run all autoregressive language modeling demonstrations.
    """
    print("Healthcare Autoregressive Language Modeling: Chain Rule Implementation")
    print("=" * 80)
    
    # Run all demonstrations
    for demo in (
        demonstrate_chain_rule_calculation,
        demonstrate_medical_text_generation,
        analyze_medical_sequence_probabilities,
        demonstrate_temperature_effects,
        medical_language_model_training_demo,
    ):
        demo()
    
    print("\n" + "=" * 80)
    print("All autoregressive language modeling demonstrations completed!")

if __name__ == "__main__":
    main()

