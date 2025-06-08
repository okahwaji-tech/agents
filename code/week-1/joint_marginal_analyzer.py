"""
Joint and Marginal Probability Distribution Analyzer

This module provides comprehensive tools for analyzing joint and marginal distributions
in language modeling contexts, with special focus on healthcare applications and
multi-dimensional probability structures.

Author: LLM Learning Guide
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import math
from typing import List, Tuple, Dict, Optional


class JointMarginalAnalyzer:
    """
    A comprehensive toolkit for analyzing joint and marginal distributions
    in language modeling contexts.
    """
    
    def __init__(self, vocab_size: int, max_seq_len: int = 50, embedding_dim: int = 128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Initialize a simple transformer-like model for demonstration
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=8, 
            batch_first=True
        )
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def compute_joint_distribution(self, sequence_length: int = 3, 
                                 context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Compute joint distribution over short sequences.
        For computational tractability, we limit to short sequences.
        
        Args:
            sequence_length: Length of sequences to analyze
            context: Optional context to condition on
            
        Returns:
            joint_probs: Tensor of shape (vocab_size^sequence_length,)
            sequence_indices: List of sequence tuples corresponding to probabilities
        """
        if sequence_length > 4:
            raise ValueError("Sequence length too large for explicit joint computation")
        
        # Generate all possible sequences of the given length
        sequences = []
        for i in range(self.vocab_size ** sequence_length):
            sequence = []
            temp = i
            for _ in range(sequence_length):
                sequence.append(temp % self.vocab_size)
                temp //= self.vocab_size
            sequences.append(tuple(sequence))
        
        # Compute probability for each sequence
        joint_probs = torch.zeros(len(sequences))
        
        with torch.no_grad():
            for idx, seq in enumerate(sequences):
                # Convert sequence to tensor
                seq_tensor = torch.tensor(seq).unsqueeze(0)  # Add batch dimension
                
                # Compute sequence probability using chain rule
                log_prob = 0.0
                for pos in range(1, sequence_length):
                    # Context is everything up to current position
                    context_seq = seq_tensor[:, :pos]
                    target_token = seq_tensor[:, pos]
                    
                    # Get model predictions
                    probs = self._get_token_probabilities(context_seq)
                    token_prob = probs[0, target_token.item()]
                    log_prob += torch.log(token_prob + 1e-10)
                
                # Add probability of first token (uniform for simplicity)
                log_prob += torch.log(torch.tensor(1.0 / self.vocab_size))
                joint_probs[idx] = torch.exp(log_prob)
        
        # Normalize to ensure valid probability distribution
        joint_probs = joint_probs / joint_probs.sum()
        
        return joint_probs, sequences
    
    def compute_marginal_distributions(self, joint_probs: torch.Tensor, 
                                     sequences: List[Tuple], 
                                     sequence_length: int) -> List[torch.Tensor]:
        """
        Compute marginal distributions for each position from joint distribution.
        
        Args:
            joint_probs: Joint probabilities for all sequences
            sequences: List of sequence tuples
            sequence_length: Length of sequences
            
        Returns:
            marginals: List of marginal distributions for each position
        """
        marginals = []
        
        for pos in range(sequence_length):
            marginal = torch.zeros(self.vocab_size)
            
            for seq_idx, seq in enumerate(sequences):
                token_at_pos = seq[pos]
                marginal[token_at_pos] += joint_probs[seq_idx]
            
            marginals.append(marginal)
        
        return marginals
    
    def analyze_token_dependencies(self, sequences: List[Tuple], 
                                 joint_probs: torch.Tensor, 
                                 pos1: int, pos2: int) -> Tuple[float, float, float]:
        """
        Analyze dependencies between tokens at two positions.
        
        Args:
            sequences: List of sequence tuples
            joint_probs: Joint probabilities
            pos1, pos2: Positions to analyze
            
        Returns:
            mutual_info: Mutual information between positions
            conditional_entropy: Conditional entropy H(pos2|pos1)
            joint_entropy: Joint entropy H(pos1, pos2)
        """
        # Compute joint distribution for the two positions
        joint_dist = torch.zeros(self.vocab_size, self.vocab_size)
        
        for seq_idx, seq in enumerate(sequences):
            token1 = seq[pos1]
            token2 = seq[pos2]
            joint_dist[token1, token2] += joint_probs[seq_idx]
        
        # Compute marginal distributions
        marginal1 = joint_dist.sum(dim=1)
        marginal2 = joint_dist.sum(dim=0)
        
        # Compute entropies
        joint_entropy = -(joint_dist * torch.log(joint_dist + 1e-10)).sum()
        marginal_entropy1 = -(marginal1 * torch.log(marginal1 + 1e-10)).sum()
        marginal_entropy2 = -(marginal2 * torch.log(marginal2 + 1e-10)).sum()
        
        # Compute mutual information
        mutual_info = marginal_entropy1 + marginal_entropy2 - joint_entropy
        
        # Compute conditional entropy H(pos2|pos1)
        conditional_entropy = joint_entropy - marginal_entropy1
        
        return mutual_info.item(), conditional_entropy.item(), joint_entropy.item()
    
    def _get_token_probabilities(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Get token probabilities from the model given context.
        
        Args:
            context_tokens: Tensor of shape (batch_size, seq_len)
            
        Returns:
            probs: Tensor of shape (batch_size, vocab_size)
        """
        batch_size, seq_len = context_tokens.shape
        
        # Create position indices
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(context_tokens)
        pos_emb = self.position_embedding(positions)
        embeddings = token_emb + pos_emb
        
        # Process through transformer
        transformer_output = self.transformer_layer(embeddings)
        
        # Use the last token's representation for prediction
        last_hidden = transformer_output[:, -1, :]
        logits = self.output_projection(last_hidden)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def analyze_position_effects(self, corpus_sequences: List[List[int]]) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Analyze how token probabilities vary by position.
        
        Args:
            corpus_sequences: List of sequences from a corpus
            
        Returns:
            position_distributions: List of token distributions for each position
            position_entropies: Entropy at each position
        """
        max_len = max(len(seq) for seq in corpus_sequences)
        position_counts = [Counter() for _ in range(max_len)]
        position_totals = [0] * max_len
        
        # Count tokens at each position
        for seq in corpus_sequences:
            for pos, token in enumerate(seq):
                position_counts[pos][token] += 1
                position_totals[pos] += 1
        
        # Convert to probability distributions
        position_distributions = []
        position_entropies = []
        
        for pos in range(max_len):
            if position_totals[pos] == 0:
                continue
                
            # Create probability distribution
            dist = torch.zeros(self.vocab_size)
            for token, count in position_counts[pos].items():
                if token < self.vocab_size:  # Ensure token is in vocabulary
                    dist[token] = count / position_totals[pos]
            
            position_distributions.append(dist)
            
            # Compute entropy
            entropy = -(dist * torch.log(dist + 1e-10)).sum()
            position_entropies.append(entropy.item())
        
        return position_distributions, position_entropies


def demonstrate_joint_marginal_concepts():
    """
    Demonstrate joint and marginal distribution concepts with examples.
    """
    print("=== Joint and Marginal Distribution Analysis ===")
    
    # Initialize analyzer with small vocabulary for tractability
    vocab_size = 6  # Small vocabulary for demonstration
    analyzer = JointMarginalAnalyzer(vocab_size)
    
    # Compute joint distribution for short sequences
    sequence_length = 3
    print(f"Computing joint distribution for sequences of length {sequence_length}")
    print(f"Total possible sequences: {vocab_size**sequence_length}")
    
    joint_probs, sequences = analyzer.compute_joint_distribution(sequence_length)
    
    # Compute marginal distributions
    marginals = analyzer.compute_marginal_distributions(joint_probs, sequences, sequence_length)
    
    print(f"\nJoint distribution computed over {len(sequences)} sequences")
    print(f"Joint probabilities sum to: {joint_probs.sum():.6f}")
    
    # Analyze marginal distributions
    print("\nMarginal distributions by position:")
    for pos, marginal in enumerate(marginals):
        entropy = -(marginal * torch.log(marginal + 1e-10)).sum()
        max_prob = marginal.max()
        print(f"Position {pos}: entropy={entropy:.3f}, max_prob={max_prob:.3f}")
    
    # Analyze dependencies between positions
    print("\nDependency analysis between positions:")
    for pos1 in range(sequence_length):
        for pos2 in range(pos1 + 1, sequence_length):
            mi, cond_ent, joint_ent = analyzer.analyze_token_dependencies(
                sequences, joint_probs, pos1, pos2
            )
            print(f"Positions {pos1}-{pos2}: MI={mi:.3f}, H(pos{pos2}|pos{pos1})={cond_ent:.3f}")


def medical_vocabulary_analysis():
    """
    Demonstrate analysis with medical vocabulary patterns.
    """
    print("\n=== Medical Vocabulary Analysis ===")

    # Simulate medical vocabulary
    medical_vocab = {
        0: "patient", 1: "chest", 2: "pain", 3: "heart", 4: "attack", 5: "diagnosis",
        6: "treatment", 7: "medication", 8: "symptoms", 9: "examination"
    }

    vocab_size = len(medical_vocab)
    analyzer = JointMarginalAnalyzer(vocab_size)

    # Simulate medical text sequences with realistic patterns
    medical_sequences = [
        [0, 1, 2],  # "patient chest pain"
        [0, 8, 3],  # "patient symptoms heart"
        [5, 3, 4],  # "diagnosis heart attack"
        [6, 7, 0],  # "treatment medication patient"
        [9, 0, 8],  # "examination patient symptoms"
        [0, 3, 2],  # "patient heart pain"
        [4, 5, 6],  # "attack diagnosis treatment"
        [1, 2, 8],  # "chest pain symptoms"
    ]

    # Analyze position effects
    position_dists, position_entropies = analyzer.analyze_position_effects(medical_sequences)

    print("Position-dependent token distributions:")
    for pos, (dist, entropy) in enumerate(zip(position_dists, position_entropies)):
        print(f"\nPosition {pos} (entropy: {entropy:.3f}):")
        # Show top tokens at this position
        top_probs, top_indices = torch.topk(dist, min(3, vocab_size))
        for prob, idx in zip(top_probs, top_indices):
            if prob > 0:
                token = medical_vocab.get(idx.item(), f"token_{idx.item()}")
                print(f"  {token}: {prob:.3f}")


def analyze_attention_as_joint_distribution():
    """
    Demonstrate how attention patterns can be viewed as joint distributions.
    """
    print("\n=== Attention as Joint Distribution ===")

    # Simulate attention weights from a transformer model
    seq_len = 8
    num_heads = 4

    # Create sample attention patterns
    attention_weights = torch.zeros(num_heads, seq_len, seq_len)

    # Head 1: Local attention (focuses on nearby tokens)
    for i in range(seq_len):
        for j in range(max(0, i-2), min(seq_len, i+3)):
            attention_weights[0, i, j] = torch.exp(-abs(i-j))
    attention_weights[0] = attention_weights[0] / attention_weights[0].sum(dim=-1, keepdim=True)

    # Head 2: Global attention (focuses on first and last tokens)
    attention_weights[1, :, 0] = 0.4  # Attention to first token
    attention_weights[1, :, -1] = 0.4  # Attention to last token
    attention_weights[1, :, 1:-1] = 0.2 / (seq_len - 2)  # Uniform over middle tokens

    # Head 3: Diagonal attention (self-attention)
    for i in range(seq_len):
        attention_weights[2, i, i] = 0.8
        attention_weights[2, i, :] = attention_weights[2, i, :] / attention_weights[2, i, :].sum()

    # Head 4: Random attention
    attention_weights[3] = torch.rand(seq_len, seq_len)
    attention_weights[3] = attention_weights[3] / attention_weights[3].sum(dim=-1, keepdim=True)

    # Analyze attention patterns as joint distributions
    print("Attention pattern analysis:")
    for head in range(num_heads):
        # Compute entropy of attention distribution for each query position
        entropies = []
        for i in range(seq_len):
            att_dist = attention_weights[head, i, :]
            entropy = -(att_dist * torch.log(att_dist + 1e-10)).sum()
            entropies.append(entropy.item())

        avg_entropy = np.mean(entropies)
        print(f"Head {head}: Average attention entropy = {avg_entropy:.3f}")

        # Compute mutual information between query and key positions
        joint_dist = attention_weights[head]  # This is already a joint distribution
        marginal_query = joint_dist.sum(dim=1)  # Marginal over key positions
        marginal_key = joint_dist.sum(dim=0)    # Marginal over query positions

        # Mutual information
        mi = 0.0
        for i in range(seq_len):
            for j in range(seq_len):
                if joint_dist[i, j] > 0:
                    mi += joint_dist[i, j] * torch.log(
                        joint_dist[i, j] / (marginal_query[i] * marginal_key[j] + 1e-10)
                    )

        print(f"Head {head}: Mutual information = {mi:.3f}")


def healthcare_sequence_modeling():
    """
    Demonstrate joint/marginal analysis for healthcare sequence modeling.
    """
    print("\n=== Healthcare Sequence Modeling ===")

    # Simulate clinical note structure
    clinical_sections = {
        "chief_complaint": [0, 1, 2],      # Patient presents with chest pain
        "history": [3, 4, 5, 6],          # History of hypertension diabetes
        "examination": [7, 8, 9],         # Physical examination findings
        "assessment": [10, 11, 12],       # Diagnosis and assessment
        "plan": [13, 14, 15]              # Treatment plan
    }

    vocab_size = 16
    analyzer = JointMarginalAnalyzer(vocab_size)

    # Generate sequences following clinical note structure
    clinical_sequences = []
    for _ in range(20):
        sequence = []
        for section, tokens in clinical_sections.items():
            # Add 1-2 tokens from each section
            num_tokens = np.random.randint(1, 3)
            selected_tokens = np.random.choice(tokens, num_tokens, replace=False)
            sequence.extend(selected_tokens)
        clinical_sequences.append(sequence)

    # Analyze position effects in clinical notes
    position_dists, position_entropies = analyzer.analyze_position_effects(clinical_sequences)

    print("Clinical note position analysis:")
    print(f"Average sequence length: {np.mean([len(seq) for seq in clinical_sequences]):.1f}")
    print(f"Position entropy variation: {np.std(position_entropies):.3f}")

    # Identify positions with low entropy (structured content)
    structured_positions = [i for i, ent in enumerate(position_entropies) if ent < np.mean(position_entropies) - np.std(position_entropies)]
    print(f"Highly structured positions: {structured_positions}")

    # Identify positions with high entropy (variable content)
    variable_positions = [i for i, ent in enumerate(position_entropies) if ent > np.mean(position_entropies) + np.std(position_entropies)]
    print(f"Highly variable positions: {variable_positions}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_joint_marginal_concepts()
    medical_vocabulary_analysis()
    analyze_attention_as_joint_distribution()
    healthcare_sequence_modeling()
