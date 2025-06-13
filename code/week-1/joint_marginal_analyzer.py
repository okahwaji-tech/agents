"""
joint_marginal_analyzer.py

Google-style toolkit for analyzing joint and marginal probability distributions
in language modeling contexts, with healthcare-focused utilities.

Defines:
  - JointMarginalAnalyzer: Core class for joint/marginal analysis.
  - Demonstration functions: joint/marginal concepts, medical vocab, attention as joint.
"""

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple, Dict, Optional


class JointMarginalAnalyzer:
    """
    Core class for joint and marginal distribution analysis in language models.

    Args:
        vocab_size (int): Token vocabulary size.
        max_seq_len (int): Maximum sequence length for positional embeddings.
        embedding_dim (int): Dimensionality of token embeddings.

    Attributes:
        token_embedding (nn.Embedding): Token embedding layer.
        position_embedding (nn.Embedding): Positional embedding layer.
        transformer_layer (nn.TransformerEncoderLayer): Transformer encoder layer.
        output_projection (nn.Linear): Projects to vocabulary logits.
    """
    
    def __init__(self, vocab_size: int, max_seq_len: int = 50, embedding_dim: int = 128):
        """
        Initialize analyzer with model components.

        Args:
            vocab_size (int): Vocabulary size.
            max_seq_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
        """
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
                                   context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
        """
        Compute joint distribution over sequences via chain rule.

        Args:
            sequence_length (int): Length of sequences to analyze.
            context (Tensor, optional): Conditioning context.

        Returns:
            Tuple[Tensor, List[Tuple[int, ...]]]: Flattened joint probabilities and sequence tuples.
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
                                       sequences: List[Tuple[int, ...]],
                                       sequence_length: int) -> List[torch.Tensor]:
        """
        Compute marginal distributions from joint probabilities.

        Args:
            joint_probs (Tensor): Joint probabilities for each sequence.
            sequences (List[Tuple[int, ...]]): List of sequence tuples.
            sequence_length (int): Sequence length.

        Returns:
            List[Tensor]: Marginal distribution for each position.
        """
        marginals = []
        
        for pos in range(sequence_length):
            marginal = torch.zeros(self.vocab_size)
            
            for seq_idx, seq in enumerate(sequences):
                token_at_pos = seq[pos]
                marginal[token_at_pos] += joint_probs[seq_idx]
            
            marginals.append(marginal)
        
        return marginals
    
    def analyze_token_dependencies(self, sequences: List[Tuple[int, ...]],
                                   joint_probs: torch.Tensor,
                                   pos1: int, pos2: int) -> Tuple[float, float, float]:
        """
        Analyze token dependencies between two positions.

        Args:
            sequences (List[Tuple[int, ...]]): Sequence tuples.
            joint_probs (Tensor): Joint probabilities tensor.
            pos1 (int): First position index.
            pos2 (int): Second position index.

        Returns:
            Tuple[float, float, float]: Mutual information, conditional entropy, joint entropy.
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
        Get conditional token probabilities from the model.

        Args:
            context_tokens (Tensor): Shape (batch_size, seq_len).

        Returns:
            Tensor: Shape (batch_size, vocab_size) probability distributions.
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
        Analyze how token distributions and entropies vary by position.

        Args:
            corpus_sequences (List[List[int]]): List of token sequences.

        Returns:
            Tuple[List[Tensor], List[float]]: Distributions and entropies per position.
        """
        max_len = max(len(seq) for seq in corpus_sequences)
        position_distributions = []
        position_entropies = []
        for pos in range(max_len):
            # Collect tokens at this position
            tokens = [seq[pos] for seq in corpus_sequences if len(seq) > pos]
            if len(tokens) == 0:
                continue
            # Compute distribution using torch.bincount
            counts = torch.bincount(torch.tensor(tokens), minlength=self.vocab_size).float()
            dist = counts / counts.sum()
            position_distributions.append(dist)
            entropy = -(dist * torch.log(dist + 1e-10)).sum()
            position_entropies.append(entropy.item())
        return position_distributions, position_entropies


def demonstrate_joint_marginal_concepts():
    """
    Demonstrate joint and marginal distribution concepts with examples.

    Returns:
        None
    """
    print("=== Joint and Marginal Distribution Analysis ===")
    vocab_size = 6  # Small vocabulary for demonstration
    analyzer = JointMarginalAnalyzer(vocab_size)
    sequence_length = 3
    print(f"Computing joint distribution for sequences of length {sequence_length}")
    print(f"Total possible sequences: {vocab_size**sequence_length}")
    joint_probs, sequences = analyzer.compute_joint_distribution(sequence_length)
    marginals = analyzer.compute_marginal_distributions(joint_probs, sequences, sequence_length)
    print(f"\nJoint distribution computed over {len(sequences)} sequences")
    print(f"Joint probabilities sum to: {joint_probs.sum():.6f}")
    print("\nMarginal distributions by position:")
    for pos, marginal in enumerate(marginals):
        entropy = -(marginal * torch.log(marginal + 1e-10)).sum()
        max_prob = marginal.max()
        print(f"Position {pos}: entropy={entropy:.3f}, max_prob={max_prob:.3f}")
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

    Returns:
        None
    """
    print("\n=== Medical Vocabulary Analysis ===")
    medical_vocab = {
        0: "patient", 1: "chest", 2: "pain", 3: "heart", 4: "attack", 5: "diagnosis",
        6: "treatment", 7: "medication", 8: "symptoms", 9: "examination"
    }
    vocab_size = len(medical_vocab)
    analyzer = JointMarginalAnalyzer(vocab_size)
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
    position_dists, position_entropies = analyzer.analyze_position_effects(medical_sequences)
    print("Position-dependent token distributions:")
    for pos, (dist, entropy) in enumerate(zip(position_dists, position_entropies)):
        print(f"\nPosition {pos} (entropy: {entropy:.3f}):")
        top_probs, top_indices = torch.topk(dist, min(3, vocab_size))
        for prob, idx in zip(top_probs, top_indices):
            if prob > 0:
                token = medical_vocab.get(idx.item(), f"token_{idx.item()}")
                print(f"  {token}: {prob:.3f}")


def analyze_attention_as_joint_distribution():
    # Demonstrate how attention matrices from different heads can be interpreted as joint distributions over query and key positions
    """
    Demonstrate how attention patterns can be viewed as joint distributions.

    Returns:
        None
    """
    print("\n=== Attention as Joint Distribution ===")
    # Define sequence length and number of attention heads for the example
    seq_len = 8
    num_heads = 4
    # Initialize tensor to hold synthetic attention weights for each head
    attention_weights = torch.zeros(num_heads, seq_len, seq_len)
    # Head 1: Local attention (focuses on nearby tokens)
    # Head 1: local attention focusing on nearby token positions
    for i in range(seq_len):
        for j in range(max(0, i-2), min(seq_len, i+3)):
            attention_weights[0, i, j] = torch.exp(-abs(i-j))
    # Normalize each row to sum to 1 (valid probability distribution)
    attention_weights[0] = attention_weights[0] / attention_weights[0].sum(dim=-1, keepdim=True)
    # Head 2: Global attention (focuses on first and last tokens)
    # Head 2: global attention with strong focus on first and last tokens
    attention_weights[1, :, 0] = 0.4  # Attention to first token
    attention_weights[1, :, -1] = 0.4  # Attention to last token
    attention_weights[1, :, 1:-1] = 0.2 / (seq_len - 2)  # Uniform over middle tokens
    # Head 3: Diagonal attention (self-attention)
    # Head 3: self-attention emphasizing the same token (identity)
    for i in range(seq_len):
        attention_weights[2, i, i] = 0.8
        attention_weights[2, i, :] = attention_weights[2, i, :] / attention_weights[2, i, :].sum()
    # Head 4: Random attention
    # Head 4: random attention pattern for contrast
    attention_weights[3] = torch.rand(seq_len, seq_len)
    attention_weights[3] = attention_weights[3] / attention_weights[3].sum(dim=-1, keepdim=True)
    # Begin analysis: compute entropy and mutual information for each head
    print("Attention pattern analysis:")
    for head in range(num_heads):
        # Compute entropy of attention distribution at each query position
        entropies = []
        for i in range(seq_len):
            att_dist = attention_weights[head, i, :]
            entropy = -(att_dist * torch.log(att_dist + 1e-10)).sum()
            entropies.append(entropy.item())
        avg_entropy = np.mean(entropies)
        # Treat the attention matrix as the joint distribution P(query_pos, key_pos)
        print(f"Head {head}: Average attention entropy = {avg_entropy:.3f}")
        joint_dist = attention_weights[head]  # This is already a joint distribution
        # Compute marginals P(query_pos) and P(key_pos)
        marginal_query = joint_dist.sum(dim=1)  # Marginal over key positions
        marginal_key = joint_dist.sum(dim=0)    # Marginal over query positions
        # Compute mutual information between query and key positions
        mi = 0.0
        for i in range(seq_len):
            for j in range(seq_len):
                if joint_dist[i, j] > 0:
                    mi += joint_dist[i, j] * torch.log(
                        joint_dist[i, j] / (marginal_query[i] * marginal_key[j] + 1e-10)
                    )
        # Output the average entropy and mutual information for this head
        print(f"Head {head}: Mutual information = {mi:.3f}")


def healthcare_sequence_modeling():
    """
    Demonstrate joint/marginal analysis for healthcare sequence modeling.

    Returns:
        None
    """
    print("\n=== Healthcare Sequence Modeling ===")
    clinical_sections = {
        "chief_complaint": [0, 1, 2],      # Patient presents with chest pain
        "history": [3, 4, 5, 6],          # History of hypertension diabetes
        "examination": [7, 8, 9],         # Physical examination findings
        "assessment": [10, 11, 12],       # Diagnosis and assessment
        "plan": [13, 14, 15]              # Treatment plan
    }
    vocab_size = 16
    analyzer = JointMarginalAnalyzer(vocab_size)
    clinical_sequences = []
    for _ in range(20):
        sequence = []
        for section, tokens in clinical_sections.items():
            num_tokens = np.random.randint(1, 3)
            selected_tokens = np.random.choice(tokens, num_tokens, replace=False)
            sequence.extend(selected_tokens)
        clinical_sequences.append(sequence)
    position_dists, position_entropies = analyzer.analyze_position_effects(clinical_sequences)
    print("Clinical note position analysis:")
    print(f"Average sequence length: {np.mean([len(seq) for seq in clinical_sequences]):.1f}")
    print(f"Position entropy variation: {np.std(position_entropies):.3f}")
    structured_positions = [i for i, ent in enumerate(position_entropies) if ent < np.mean(position_entropies) - np.std(position_entropies)]
    print(f"Highly structured positions: {structured_positions}")
    variable_positions = [i for i, ent in enumerate(position_entropies) if ent > np.mean(position_entropies) + np.std(position_entropies)]
    print(f"Highly variable positions: {variable_positions}")


if __name__ == "__main__":
    for fn in [
        demonstrate_joint_marginal_concepts,
        medical_vocabulary_analysis,
        analyze_attention_as_joint_distribution,
        healthcare_sequence_modeling,
    ]:
        fn()
