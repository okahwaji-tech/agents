"""
chain_rule_language_model.py

Implementation of conditional probability calculations for language modeling,
including Bayes’ theorem applications and sampling strategies in healthcare contexts.

Defines:
  - ConditionalProbabilityCalculator: Core class for conditional and sequence probability.
  - Utility functions for demonstration of conditional concepts and sampling strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


class ConditionalProbabilityCalculator:
    """
    Core class for computing conditional and sequence probabilities.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        embedding_dim (int): Dimensionality of token embeddings.
        hidden_dim (int): Hidden state size for the LSTM.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        lstm (nn.LSTM): LSTM encoder for context.
        output_projection (nn.Linear): Projects hidden state to token logits.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        """
        Initialize the embedding, LSTM encoder, and output projection layers.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding vector size.
            hidden_dim (int): Hidden layer size for the LSTM.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize a simple neural language model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def compute_conditional_probabilities(self, context_tokens):
        """
        Compute conditional distribution for next-token prediction.

        Args:
            context_tokens (Tensor): Shape (batch_size, seq_len) input context.

        Returns:
            Tuple[Tensor, Tensor]: 
                probabilities (batch_size, vocab_size), 
                log_probabilities (batch_size, vocab_size).
        """
        # Embed context tokens
        embeddings = self.embedding(context_tokens)
        
        # Encode context with LSTM
        lstm_output, (hidden, cell) = self.lstm(embeddings)
        
        # Project to vocabulary logits
        final_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        logits = self.output_projection(final_hidden)  # Shape: (batch_size, vocab_size)
        
        # Compute probabilities and log-probabilities
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = F.log_softmax(logits, dim=-1)
        
        return probabilities, log_probabilities
    
    def compute_sequence_probability(self, sequence_tokens):
        """
        Compute log-probability of full sequences via the chain rule.

        Args:
            sequence_tokens (Tensor): Shape (batch_size, seq_len).

        Returns:
            Tensor: Shape (batch_size,) total log-probability per sequence.
        """
        batch_size, seq_len = sequence_tokens.shape
        total_log_prob = torch.zeros(batch_size)
        
        for i in range(1, seq_len):
            # Context is everything up to position i
            context = sequence_tokens[:, :i]
            # Target is the token at position i
            target = sequence_tokens[:, i]
            
            # Compute conditional probabilities
            _, log_probs = self.compute_conditional_probabilities(context)
            
            # Extract the log probability of the target token
            target_log_prob = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            total_log_prob += target_log_prob
            
        return total_log_prob
    
    def sample_next_token(self, context_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Sample the next token given context using temperature, top-k, and top-p.

        Args:
            context_tokens (Tensor): Shape (batch_size, seq_len) input context.
            temperature (float): Sampling temperature.
            top_k (int, optional): Limits sampling to top-k tokens.
            top_p (float, optional): Nucleus sampling threshold.

        Returns:
            Tuple[Tensor, Tensor]: next_token IDs and their probabilities.
        """
        with torch.no_grad():
            # Get conditional probabilities
            probs, log_probs = self.compute_conditional_probabilities(context_tokens)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = log_probs / temperature
                probs = F.softmax(logits, dim=-1)
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                # Create a mask for non-top-k tokens
                mask = torch.zeros_like(probs)
                mask.scatter_(1, top_k_indices, 1)
                probs = probs * mask
                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find the cutoff point
                cutoff_mask = cumulative_probs <= top_p
                # Include at least one token
                cutoff_mask[:, 0] = True
                
                # Zero out probabilities beyond the cutoff
                sorted_probs[~cutoff_mask] = 0
                # Renormalize
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                # Scatter back to original order
                probs = torch.zeros_like(probs)
                probs.scatter_(1, sorted_indices, sorted_probs)
            
            # Sample from the categorical distribution
            categorical = Categorical(probs)
            next_token = categorical.sample()
            token_prob = probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            
            return next_token, token_prob


def demonstrate_conditional_probability_concepts():
    """
    Demonstrate key conditional probability concepts with concrete examples.

    Returns:
        None
    """
    # Set up a simple example
    vocab_size = 1000
    batch_size = 4
    seq_len = 10
    
    calculator = ConditionalProbabilityCalculator(vocab_size)
    
    # Generate some example context
    context = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("=== Conditional Probability Demonstration ===")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Context shape: {context.shape}")
    
    # Compute conditional probabilities
    probs, log_probs = calculator.compute_conditional_probabilities(context)
    print(f"Probability distribution shape: {probs.shape}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))}")
    
    # Analyze the distribution properties
    entropy = -(probs * log_probs).sum(dim=-1)
    max_prob = probs.max(dim=-1)[0]
    
    print(f"Average entropy: {entropy.mean().item():.3f}")
    print(f"Average max probability: {max_prob.mean().item():.3f}")
    
    # Demonstrate different sampling strategies
    print("\n=== Sampling Strategies ===")
    
    # Greedy sampling (temperature = 0 equivalent)
    greedy_token = probs.argmax(dim=-1)
    greedy_prob = probs.gather(1, greedy_token.unsqueeze(1)).squeeze(1)
    print(f"Greedy sampling - Average probability: {greedy_prob.mean().item():.3f}")
    
    # Random sampling with different temperatures
    for temp in [0.5, 1.0, 2.0]:
        sampled_token, sampled_prob = calculator.sample_next_token(context, temperature=temp)
        print(f"Temperature {temp} - Average probability: {sampled_prob.mean().item():.3f}")
    
    # Top-k sampling
    top_k_token, top_k_prob = calculator.sample_next_token(context, top_k=50)
    print(f"Top-k (k=50) - Average probability: {top_k_prob.mean().item():.3f}")
    
    # Nucleus sampling
    nucleus_token, nucleus_prob = calculator.sample_next_token(context, top_p=0.9)
    print(f"Nucleus (p=0.9) - Average probability: {nucleus_prob.mean().item():.3f}")


def demonstrate_bayes_theorem_application():
    """
    Demonstrate how Bayes' theorem applies to language modeling scenarios.

    Returns:
        None
    """
    print("\n=== Bayes' Theorem in Language Modeling ===")
    
    # Simulate a medical language modeling scenario
    # Vocabulary: 0=chest, 1=pain, 2=heart, 3=attack, 4=indigestion, 5=anxiety
    medical_vocab = ["chest", "pain", "heart", "attack", "indigestion", "anxiety"]
    vocab_size = len(medical_vocab)
    
    # Prior probabilities for different diagnoses given "chest pain"
    # P(diagnosis) - base rates in population
    diagnoses = ["heart_attack", "indigestion", "anxiety"]
    prior_probs = torch.tensor([0.1, 0.6, 0.3])  # Heart attack is rare but serious
    
    # Likelihood: P(additional_symptoms | diagnosis)
    # Symptoms: [shortness_of_breath, nausea, sweating]
    # Rows: diagnoses, Columns: symptoms
    likelihood_matrix = torch.tensor([
        [0.8, 0.7, 0.9],  # Heart attack: high likelihood of all symptoms
        [0.2, 0.8, 0.1],  # Indigestion: mainly nausea
        [0.6, 0.3, 0.4]   # Anxiety: mainly shortness of breath
    ])
    
    # Observed symptoms (binary indicators)
    observed_symptoms = torch.tensor([1.0, 1.0, 0.0])  # Shortness of breath + nausea, no sweating
    
    # Compute likelihoods for each diagnosis
    likelihoods = torch.prod(
        likelihood_matrix * observed_symptoms + (1 - likelihood_matrix) * (1 - observed_symptoms),
        dim=1
    )
    
    # Apply Bayes' theorem: P(diagnosis | symptoms) ∝ P(symptoms | diagnosis) * P(diagnosis)
    posterior_unnormalized = likelihoods * prior_probs
    posterior_probs = posterior_unnormalized / posterior_unnormalized.sum()
    
    print("Medical Diagnosis Example:")
    print("Observed: chest pain + shortness of breath + nausea")
    print("\nPrior probabilities:")
    for i, diagnosis in enumerate(diagnoses):
        print(f"  {diagnosis}: {prior_probs[i]:.3f}")
    
    print("\nLikelihoods P(symptoms | diagnosis):")
    for i, diagnosis in enumerate(diagnoses):
        print(f"  {diagnosis}: {likelihoods[i]:.3f}")
    
    print("\nPosterior probabilities P(diagnosis | symptoms):")
    for i, diagnosis in enumerate(diagnoses):
        print(f"  {diagnosis}: {posterior_probs[i]:.3f}")
    
    # Show how the posterior changes with different priors
    print("\n=== Sensitivity to Priors ===")
    alternative_priors = torch.tensor([0.05, 0.8, 0.15])  # Even lower heart attack prior
    alt_posterior_unnormalized = likelihoods * alternative_priors
    alt_posterior_probs = alt_posterior_unnormalized / alt_posterior_unnormalized.sum()
    
    print("With different priors:")
    for i, diagnosis in enumerate(diagnoses):
        print(f"  {diagnosis}: {alt_posterior_probs[i]:.3f} (was {posterior_probs[i]:.3f})")


def analyze_conditional_independence():
    """
    Analyze conditional independence assumptions in language modeling.

    Returns:
        None
    """
    print("\n=== Conditional Independence Analysis ===")

    # Simulate a scenario where we test conditional independence
    # Context: "The patient has a history of"
    # We want to test if P(diabetes | context, hypertension) = P(diabetes | context)

    # Simulate some conditional probabilities
    # P(diabetes | "patient has history of")
    p_diabetes_given_context = 0.15

    # P(diabetes | "patient has history of", hypertension=True)
    p_diabetes_given_context_and_hypertension = 0.35

    # P(diabetes | "patient has history of", hypertension=False)
    p_diabetes_given_context_and_no_hypertension = 0.08

    # P(hypertension | "patient has history of")
    p_hypertension_given_context = 0.4

    # Check if conditional independence holds
    # If independent: P(diabetes | context) should equal the weighted average
    expected_if_independent = (
        p_diabetes_given_context_and_hypertension * p_hypertension_given_context +
        p_diabetes_given_context_and_no_hypertension * (1 - p_hypertension_given_context)
    )

    print("Testing conditional independence:")
    print(f"P(diabetes | context) = {p_diabetes_given_context:.3f}")
    print(f"Expected if independent = {expected_if_independent:.3f}")
    print(f"Difference = {abs(p_diabetes_given_context - expected_if_independent):.3f}")

    if abs(p_diabetes_given_context - expected_if_independent) < 0.01:
        print("Conditional independence approximately holds")
    else:
        print("Conditional independence is violated - there are dependencies!")


def medical_term_prediction_example():
    """
    Demonstrate conditional probability in medical term prediction.

    Returns:
        None
    """
    print("\n=== Medical Term Prediction Example ===")

    # Simulate a medical language model predicting the next term
    # Context: "Patient presents with acute chest pain and"

    # Define medical vocabulary
    medical_terms = [
        "shortness_of_breath", "diaphoresis", "nausea", "palpitations",
        "dizziness", "fatigue", "anxiety", "indigestion"
    ]

    # Simulate conditional probabilities based on medical knowledge
    # These would normally be learned by the model
    conditional_probs = torch.tensor([
        0.25,  # shortness_of_breath - very common with chest pain
        0.20,  # diaphoresis - common in cardiac events
        0.15,  # nausea - moderately common
        0.12,  # palpitations - related to cardiac issues
        0.10,  # dizziness - possible
        0.08,  # fatigue - less specific
        0.06,  # anxiety - can cause chest pain
        0.04   # indigestion - differential diagnosis
    ])

    # Ensure probabilities sum to 1
    conditional_probs = conditional_probs / conditional_probs.sum()

    print("Context: 'Patient presents with acute chest pain and'")
    print("\nPredicted next terms (P(term | context)):")

    # Sort by probability for better display
    sorted_probs, sorted_indices = torch.sort(conditional_probs, descending=True)

    for i in range(len(medical_terms)):
        idx = sorted_indices[i]
        term = medical_terms[idx]
        prob = sorted_probs[i]
        print(f"  {term}: {prob:.3f}")

    # Calculate entropy to measure uncertainty
    entropy = -(conditional_probs * torch.log(conditional_probs + 1e-10)).sum()
    print(f"\nEntropy: {entropy:.3f} (lower = more certain)")

    # Simulate sampling with different strategies
    print("\nSampling strategies:")

    # Greedy (most likely)
    greedy_idx = conditional_probs.argmax()
    print(f"Greedy: {medical_terms[greedy_idx]} (p={conditional_probs[greedy_idx]:.3f})")

    # Random sampling
    categorical = Categorical(conditional_probs)
    sampled_idx = categorical.sample()
    print(f"Random sample: {medical_terms[sampled_idx]} (p={conditional_probs[sampled_idx]:.3f})")

    # Top-k sampling (k=3)
    top_k = 3
    top_k_probs, top_k_indices = torch.topk(conditional_probs, top_k)
    top_k_normalized = top_k_probs / top_k_probs.sum()
    top_k_categorical = Categorical(top_k_normalized)
    top_k_sample_idx = top_k_categorical.sample()
    actual_idx = top_k_indices[top_k_sample_idx]
    print(f"Top-{top_k} sample: {medical_terms[actual_idx]} (p={conditional_probs[actual_idx]:.3f})")


if __name__ == "__main__":
    for demo in (
        demonstrate_conditional_probability_concepts,
        demonstrate_bayes_theorem_application,
        analyze_conditional_independence,
        medical_term_prediction_example,
    ):
        demo()
