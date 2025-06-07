"""
Comprehensive PyTorch Implementation: Discrete Probability Distributions for LLMs
==============================================================================

This module provides practical implementations of discrete probability distributions
commonly used in language modeling, with a focus on healthcare applications.

Focus: LLM Applications in Healthcare
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Binomial, Multinomial, Categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import math

class HealthcareLMDistributions:
    """
    A comprehensive class for working with discrete probability distributions
    in healthcare language modeling contexts.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.medical_vocab = self._create_medical_vocabulary()
        
    def _create_medical_vocabulary(self) -> Dict[int, str]:
        """Create a sample medical vocabulary for demonstrations."""
        medical_terms = [
            "patient", "diagnosis", "treatment", "symptoms", "examination",
            "chest", "pain", "heart", "attack", "blood", "pressure",
            "diabetes", "hypertension", "medication", "dosage", "therapy",
            "surgery", "procedure", "recovery", "prognosis", "acute",
            "chronic", "severe", "mild", "moderate", "normal", "abnormal",
            "elevated", "decreased", "stable", "improving", "worsening",
            "history", "family", "medical", "surgical", "allergies",
            "laboratory", "results", "imaging", "xray", "mri", "ct",
            "ultrasound", "ecg", "ekg", "vital", "signs", "temperature",
            "pulse", "respiratory", "rate", "oxygen", "saturation"
        ]
        
        vocab = {i: term for i, term in enumerate(medical_terms)}
        # Fill remaining slots with generic tokens
        for i in range(len(medical_terms), min(self.vocab_size, 100)):
            vocab[i] = f"token_{i}"
        
        return vocab
    
    def demonstrate_bernoulli_distribution(self):
        """
        Demonstrate Bernoulli distribution for binary medical outcomes.
        Example: Presence/absence of a symptom
        """
        print("=== Bernoulli Distribution: Symptom Presence ===")
        
        # Probability of chest pain in cardiac patients
        p_chest_pain = 0.75
        
        # Create Bernoulli distribution
        bernoulli_dist = Bernoulli(probs=p_chest_pain)
        
        # Sample from the distribution
        samples = bernoulli_dist.sample((1000,))
        
        print(f"Probability of chest pain: {p_chest_pain}")
        print(f"Theoretical mean: {p_chest_pain}")
        print(f"Empirical mean: {samples.float().mean().item():.3f}")
        print(f"Theoretical variance: {p_chest_pain * (1 - p_chest_pain):.3f}")
        print(f"Empirical variance: {samples.float().var().item():.3f}")
        
        # Medical interpretation
        positive_cases = samples.sum().item()
        print(f"\nOut of 1000 cardiac patients:")
        print(f"  {positive_cases} presented with chest pain")
        print(f"  {1000 - positive_cases} did not present with chest pain")
        
        return samples
    
    def demonstrate_binomial_distribution(self):
        """
        Demonstrate Binomial distribution for counting medical events.
        Example: Number of patients with adverse reactions in a trial
        """
        print("\n=== Binomial Distribution: Adverse Drug Reactions ===")
        
        # Clinical trial parameters
        n_patients = 100
        p_adverse_reaction = 0.15
        
        # Create Binomial distribution
        binomial_dist = Binomial(total_count=n_patients, probs=p_adverse_reaction)
        
        # Sample from the distribution
        samples = binomial_dist.sample((1000,))  # 1000 trials
        
        print(f"Trial size: {n_patients} patients")
        print(f"Probability of adverse reaction per patient: {p_adverse_reaction}")
        print(f"Theoretical mean: {n_patients * p_adverse_reaction}")
        print(f"Empirical mean: {samples.float().mean().item():.3f}")
        print(f"Theoretical variance: {n_patients * p_adverse_reaction * (1 - p_adverse_reaction):.3f}")
        print(f"Empirical variance: {samples.float().var().item():.3f}")
        
        # Analyze distribution of outcomes
        unique_counts, frequencies = torch.unique(samples, return_counts=True)
        
        print(f"\nDistribution of adverse reactions across {len(samples)} trials:")
        for count, freq in zip(unique_counts[:10], frequencies[:10]):  # Show top 10
            probability = freq.item() / len(samples)
            print(f"  {count.item()} reactions: {probability:.3f} probability")
        
        return samples
    
    def demonstrate_multinomial_distribution(self):
        """
        Demonstrate Multinomial distribution for medical category classification.
        Example: Distribution of diagnoses in emergency department
        """
        print("\n=== Multinomial Distribution: Emergency Department Diagnoses ===")
        
        # Define diagnostic categories and their probabilities
        diagnoses = [
            "Cardiac", "Respiratory", "Neurological", "Gastrointestinal", 
            "Musculoskeletal", "Other"
        ]
        
        # Probabilities based on typical ED distribution
        probs = torch.tensor([0.15, 0.20, 0.10, 0.18, 0.25, 0.12])
        
        # Number of patients per day
        n_patients = 200
        
        # Create Multinomial distribution
        multinomial_dist = Multinomial(total_count=n_patients, probs=probs)
        
        # Sample from the distribution (simulate multiple days)
        samples = multinomial_dist.sample((30,))  # 30 days
        
        print(f"Patients per day: {n_patients}")
        print(f"Diagnostic categories: {diagnoses}")
        print(f"Expected probabilities: {probs.tolist()}")
        
        # Analyze average distribution
        mean_counts = samples.float().mean(dim=0)
        empirical_probs = mean_counts / n_patients
        
        print(f"\nAverage daily distribution over 30 days:")
        for i, (diagnosis, expected_prob, empirical_prob, mean_count) in enumerate(
            zip(diagnoses, probs, empirical_probs, mean_counts)
        ):
            print(f"  {diagnosis}: {mean_count:.1f} patients "
                  f"(expected: {expected_prob:.3f}, empirical: {empirical_prob:.3f})")
        
        return samples, diagnoses
    
    def demonstrate_categorical_distribution(self):
        """
        Demonstrate Categorical distribution for next-token prediction.
        Example: Predicting next medical term given context
        """
        print("\n=== Categorical Distribution: Medical Term Prediction ===")
        
        # Simulate a medical language model's output distribution
        # Context: "Patient presents with chest"
        # Possible next words and their probabilities
        
        next_words = ["pain", "discomfort", "tightness", "pressure", "burning"]
        word_indices = list(range(len(next_words)))
        
        # Probabilities from a hypothetical medical LM
        probs = torch.tensor([0.45, 0.25, 0.15, 0.10, 0.05])
        
        # Create Categorical distribution
        categorical_dist = Categorical(probs=probs)
        
        # Sample next words
        samples = categorical_dist.sample((1000,))
        
        print(f"Context: 'Patient presents with chest'")
        print(f"Possible continuations: {next_words}")
        print(f"Model probabilities: {probs.tolist()}")
        
        # Analyze sampling results
        sample_counts = torch.bincount(samples, minlength=len(next_words))
        empirical_probs = sample_counts.float() / len(samples)
        
        print(f"\nSampling results over 1000 generations:")
        for word, expected_prob, empirical_prob, count in zip(
            next_words, probs, empirical_probs, sample_counts
        ):
            print(f"  '{word}': {count.item()} times "
                  f"(expected: {expected_prob:.3f}, empirical: {empirical_prob:.3f})")
        
        # Compute entropy of the distribution
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        print(f"\nDistribution entropy: {entropy:.3f} bits")
        print(f"Perplexity: {torch.exp(entropy):.3f}")
        
        return samples, next_words
    
    def medical_text_generation_example(self):
        """
        Demonstrate how discrete distributions work together in medical text generation.
        """
        print("\n=== Medical Text Generation with Discrete Distributions ===")
        
        # Define a simple medical text generation scenario
        # Each position uses a categorical distribution over medical terms
        
        # Medical sentence templates and their probabilities
        sentence_starts = {
            "Patient": 0.4,
            "The": 0.3,
            "History": 0.2,
            "Examination": 0.1
        }
        
        # Continuation probabilities given different starts
        continuations = {
            "Patient": {
                "presents": 0.5, "reports": 0.3, "denies": 0.1, "has": 0.1
            },
            "The": {
                "patient": 0.6, "examination": 0.2, "history": 0.1, "diagnosis": 0.1
            },
            "History": {
                "reveals": 0.4, "includes": 0.3, "shows": 0.2, "indicates": 0.1
            },
            "Examination": {
                "shows": 0.4, "reveals": 0.3, "demonstrates": 0.2, "indicates": 0.1
            }
        }
        
        # Generate medical sentences
        print("Generating medical sentences using categorical distributions:")
        
        for i in range(5):
            # Sample sentence start
            start_words = list(sentence_starts.keys())
            start_probs = torch.tensor(list(sentence_starts.values()))
            start_dist = Categorical(probs=start_probs)
            start_idx = start_dist.sample()
            start_word = start_words[start_idx.item()]
            
            # Sample continuation
            cont_words = list(continuations[start_word].keys())
            cont_probs = torch.tensor(list(continuations[start_word].values()))
            cont_dist = Categorical(probs=cont_probs)
            cont_idx = cont_dist.sample()
            cont_word = cont_words[cont_idx.item()]
            
            sentence = f"{start_word} {cont_word}..."
            start_prob = start_probs[start_idx].item()
            cont_prob = cont_probs[cont_idx].item()
            joint_prob = start_prob * cont_prob
            
            print(f"  {i+1}. '{sentence}' (P = {joint_prob:.4f})")
    
    def analyze_medical_vocabulary_distribution(self):
        """
        Analyze the distribution of medical terms in a simulated corpus.
        """
        print("\n=== Medical Vocabulary Distribution Analysis ===")
        
        # Simulate a medical text corpus with Zipfian distribution
        vocab_size = 50
        medical_terms = list(self.medical_vocab.values())[:vocab_size]
        
        # Generate Zipfian frequencies (medical texts often follow this pattern)
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float)
        frequencies = 1.0 / ranks  # Zipf's law: frequency ∝ 1/rank
        frequencies = frequencies / frequencies.sum()  # Normalize
        
        # Create categorical distribution over medical vocabulary
        vocab_dist = Categorical(probs=frequencies)
        
        # Sample a large corpus
        corpus_size = 10000
        corpus_samples = vocab_dist.sample((corpus_size,))
        
        # Analyze the resulting distribution
        term_counts = torch.bincount(corpus_samples, minlength=vocab_size)
        empirical_freqs = term_counts.float() / corpus_size
        
        print(f"Simulated medical corpus: {corpus_size} tokens")
        print(f"Vocabulary size: {vocab_size} medical terms")
        print(f"\nTop 10 most frequent terms:")
        
        # Sort by frequency
        sorted_indices = torch.argsort(empirical_freqs, descending=True)
        
        for i in range(10):
            idx = sorted_indices[i].item()
            term = medical_terms[idx] if idx < len(medical_terms) else f"term_{idx}"
            expected_freq = frequencies[idx].item()
            empirical_freq = empirical_freqs[idx].item()
            count = term_counts[idx].item()
            
            print(f"  {i+1}. '{term}': {count} occurrences "
                  f"(expected: {expected_freq:.4f}, empirical: {empirical_freq:.4f})")
        
        # Compute corpus statistics
        entropy = -(empirical_freqs * torch.log(empirical_freqs + 1e-10)).sum()
        perplexity = torch.exp(entropy)
        
        print(f"\nCorpus statistics:")
        print(f"  Vocabulary entropy: {entropy:.3f} bits")
        print(f"  Vocabulary perplexity: {perplexity:.3f}")
        
        return corpus_samples, medical_terms
    
    def demonstrate_temperature_scaling(self):
        """
        Demonstrate how temperature scaling affects categorical distributions.
        """
        print("\n=== Temperature Scaling in Medical Text Generation ===")
        
        # Original logits from a medical language model
        # Context: "Patient diagnosed with"
        medical_conditions = ["diabetes", "hypertension", "pneumonia", "arthritis", "asthma"]
        logits = torch.tensor([2.5, 2.0, 1.5, 1.0, 0.5])  # Raw model outputs
        
        temperatures = [0.5, 1.0, 2.0]
        
        print(f"Medical conditions: {medical_conditions}")
        print(f"Original logits: {logits.tolist()}")
        
        for temp in temperatures:
            # Apply temperature scaling
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=0)
            
            # Create categorical distribution
            cat_dist = Categorical(probs=probs)
            
            # Sample from the distribution
            samples = cat_dist.sample((1000,))
            sample_counts = torch.bincount(samples, minlength=len(medical_conditions))
            empirical_probs = sample_counts.float() / 1000
            
            # Compute entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            
            print(f"\nTemperature = {temp}:")
            print(f"  Entropy: {entropy:.3f} bits")
            
            for i, (condition, prob, emp_prob) in enumerate(
                zip(medical_conditions, probs, empirical_probs)
            ):
                print(f"  {condition}: P={prob:.3f}, empirical={emp_prob:.3f}")
    
    def medical_sequence_probability_calculation(self):
        """
        Demonstrate sequence probability calculation for medical text.
        """
        print("\n=== Medical Sequence Probability Calculation ===")
        
        # Define a medical sequence: "Patient presents with chest pain"
        sequence = ["Patient", "presents", "with", "chest", "pain"]
        
        # Simulate conditional probabilities at each position
        # These would normally come from a trained language model
        conditional_probs = [
            {"Patient": 0.3, "The": 0.4, "History": 0.2, "Examination": 0.1},  # P(w1)
            {"presents": 0.5, "has": 0.3, "reports": 0.15, "denies": 0.05},    # P(w2|w1)
            {"with": 0.7, "a": 0.15, "symptoms": 0.1, "signs": 0.05},          # P(w3|w1,w2)
            {"chest": 0.4, "abdominal": 0.2, "back": 0.2, "head": 0.2},        # P(w4|w1,w2,w3)
            {"pain": 0.6, "discomfort": 0.2, "pressure": 0.15, "tightness": 0.05}  # P(w5|w1,w2,w3,w4)
        ]
        
        print(f"Medical sequence: {' '.join(sequence)}")
        print(f"\nStep-by-step probability calculation:")
        
        total_log_prob = 0.0
        
        for i, (word, prob_dict) in enumerate(zip(sequence, conditional_probs)):
            prob = prob_dict.get(word, 0.001)  # Small probability for unseen words
            log_prob = math.log(prob)
            total_log_prob += log_prob
            
            if i == 0:
                print(f"  P({word}) = {prob:.3f}")
            else:
                context = " ".join(sequence[:i])
                print(f"  P({word}|{context}) = {prob:.3f}")
        
        sequence_prob = math.exp(total_log_prob)
        
        print(f"\nTotal log probability: {total_log_prob:.6f}")
        print(f"Sequence probability: {sequence_prob:.10f}")
        print(f"Perplexity: {math.exp(-total_log_prob / len(sequence)):.3f}")

def main():
    """
    Run all discrete distribution demonstrations.
    """
    print("Healthcare Language Modeling: Discrete Probability Distributions")
    print("=" * 70)
    
    # Initialize the demonstration class
    demo = HealthcareLMDistributions()
    
    # Run all demonstrations
    demo.demonstrate_bernoulli_distribution()
    demo.demonstrate_binomial_distribution()
    demo.demonstrate_multinomial_distribution()
    demo.demonstrate_categorical_distribution()
    demo.medical_text_generation_example()
    demo.analyze_medical_vocabulary_distribution()
    demo.demonstrate_temperature_scaling()
    demo.medical_sequence_probability_calculation()
    
    print("\n" + "=" * 70)
    print("All discrete distribution demonstrations completed!")

if __name__ == "__main__":
    main()

