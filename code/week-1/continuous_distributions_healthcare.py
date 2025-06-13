"""
continuous_distributions_healthcare.py

Implementation of continuous probability distributions for language models
with healthcare applications.

Defines:
  - ContinuousDistributionsLM: Class demonstrating continuous distributions
  - Utility methods for Normal, MultivariateNormal, Exponential, Gamma, Beta distributions
  - Advanced demonstrations including uncertainty quantification and dosage modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Exponential, Gamma, Beta, MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional

class ContinuousDistributionsLM:
    """
    Class for demonstrating continuous probability distributions in LM applications.

    Args:
        device (str): Device identifier, e.g., "cpu" or "cuda".

    Attributes:
        device (str): Device for tensor computation.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the distribution demonstration class.

        Args:
            device (str): Device to run computations on.
        """
        self.device = device
        
    def demonstrate_normal_distribution(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Demonstrate Normal distribution for weight initialization and activations.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Xavier weights, He weights, final activations.
        """
        print("=== Normal Distribution: Neural Network Applications ===")
        
        # Weight initialization example
        input_dim = 512
        output_dim = 256
        
        # Xavier/Glorot initialization
        xavier_std = math.sqrt(2.0 / (input_dim + output_dim))
        xavier_weights = Normal(0, xavier_std).sample((output_dim, input_dim))
        
        # He initialization
        he_std = math.sqrt(2.0 / input_dim)
        he_weights = Normal(0, he_std).sample((output_dim, input_dim))
        
        print(f"Weight matrix shape: {xavier_weights.shape}")
        print(f"Xavier initialization - mean: {xavier_weights.mean():.6f}, std: {xavier_weights.std():.6f}")
        print(f"He initialization - mean: {he_weights.mean():.6f}, std: {he_weights.std():.6f}")
        
        # Simulate forward pass through multiple layers
        print(f"\nActivation analysis through network layers:")
        
        batch_size = 32
        seq_len = 128
        embedding_dim = 512
        
        # Input embeddings (normally distributed)
        input_embeddings = Normal(0, 1).sample((batch_size, seq_len, embedding_dim))
        
        # Simulate passing through transformer layers
        current_activations = input_embeddings
        
        for layer in range(6):  # 6 transformer layers
            # Linear transformation
            weight_matrix = Normal(0, xavier_std).sample((embedding_dim, embedding_dim))
            current_activations = torch.matmul(current_activations, weight_matrix.T)
            
            # Add layer normalization effect (normalize to unit variance)
            current_activations = F.layer_norm(current_activations, (embedding_dim,))
            
            # Apply activation function (GELU approximation)
            current_activations = current_activations * torch.sigmoid(1.702 * current_activations)
            
            # Analyze distribution properties
            mean_val = current_activations.mean().item()
            std_val = current_activations.std().item()
            
            print(f"  Layer {layer + 1}: mean = {mean_val:.4f}, std = {std_val:.4f}")
        
        return xavier_weights, he_weights, current_activations
    
    def demonstrate_multivariate_normal(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Demonstrate Multivariate Normal distribution for correlated embeddings and attention patterns.

        Returns:
            Tuple[Tensor, Tensor]: Embeddings tensor and correlation matrix.
        """
        print("\n=== Multivariate Normal: Correlated Embeddings ===")
        
        # Medical concept embeddings with correlations
        embedding_dim = 4
        concepts = ["heart", "cardiac", "chest", "pain"]
        
        # Define correlation structure for medical concepts
        # Heart and cardiac should be highly correlated
        # Chest and pain should be moderately correlated
        correlation_matrix = torch.tensor([
            [1.0, 0.8, 0.3, 0.2],  # heart
            [0.8, 1.0, 0.3, 0.2],  # cardiac
            [0.3, 0.3, 1.0, 0.6],  # chest
            [0.2, 0.2, 0.6, 1.0]   # pain
        ])
        
        # Convert correlation to covariance (assuming unit variance)
        covariance_matrix = correlation_matrix
        
        # Create multivariate normal distribution
        mean_vector = torch.zeros(embedding_dim)
        mvn_dist = MultivariateNormal(mean_vector, covariance_matrix)
        
        # Sample embeddings for medical concepts
        n_samples = 1000
        embeddings = mvn_dist.sample((n_samples,))
        
        print(f"Medical concepts: {concepts}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Samples generated: {n_samples}")
        
        # Analyze empirical correlations
        empirical_cov = torch.cov(embeddings.T)
        
        print(f"\nTheoretical vs Empirical Correlations:")
        for i, concept_i in enumerate(concepts):
            for j, concept_j in enumerate(concepts):
                if i <= j:
                    theoretical = correlation_matrix[i, j].item()
                    empirical = empirical_cov[i, j].item()
                    print(f"  {concept_i}-{concept_j}: theoretical={theoretical:.3f}, empirical={empirical:.3f}")
        
        # Demonstrate conditional distribution
        # Given that "heart" embedding is positive, what's the distribution of "cardiac"?
        heart_positive_mask = embeddings[:, 0] > 0
        cardiac_given_heart_positive = embeddings[heart_positive_mask, 1]
        
        print(f"\nConditional analysis:")
        print(f"P(cardiac | heart > 0): mean = {cardiac_given_heart_positive.mean():.3f}")
        print(f"P(cardiac | heart > 0): std = {cardiac_given_heart_positive.std():.3f}")
        
        return embeddings, correlation_matrix
    
    def demonstrate_exponential_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Demonstrate Exponential distribution for attention decay and temporal patterns.

        Returns:
            Tuple[Tensor, Tensor]: Attention weights and inter-event times.
        """
        print("\n=== Exponential Distribution: Attention Decay ===")
        
        # Model attention decay over sequence positions
        sequence_length = 50
        decay_rate = 0.1  # Higher rate = faster decay
        
        # Create exponential distribution for attention weights
        exp_dist = Exponential(rate=decay_rate)
        
        # Generate attention decay pattern
        positions = torch.arange(1, sequence_length + 1, dtype=torch.float)
        attention_weights = torch.exp(-decay_rate * positions)
        attention_weights = attention_weights / attention_weights.sum()  # Normalize
        
        print(f"Sequence length: {sequence_length}")
        print(f"Decay rate: {decay_rate}")
        print(f"Attention weights sum: {attention_weights.sum():.6f}")
        
        # Sample inter-event times (e.g., time between medical events)
        n_events = 100
        inter_event_times = exp_dist.sample((n_events,))
        
        print(f"\nInter-event time analysis (medical events):")
        print(f"  Mean time: {inter_event_times.mean():.3f} (theoretical: {1/decay_rate:.3f})")
        print(f"  Std time: {inter_event_times.std():.3f} (theoretical: {1/decay_rate:.3f})")
        
        # Demonstrate memoryless property
        # P(T > s + t | T > s) = P(T > t)
        s, t = 2.0, 3.0
        
        # Empirical verification
        mask_greater_s = inter_event_times > s
        conditional_times = inter_event_times[mask_greater_s] - s
        prob_greater_t_given_s = (conditional_times > t).float().mean()
        prob_greater_t = (inter_event_times > t).float().mean()
        
        print(f"\nMemoryless property verification:")
        print(f"  P(T > {t} | T > {s}) = {prob_greater_t_given_s:.3f}")
        print(f"  P(T > {t}) = {prob_greater_t:.3f}")
        print(f"  Difference: {abs(prob_greater_t_given_s - prob_greater_t):.3f}")
        
        return attention_weights, inter_event_times
    
    def demonstrate_gamma_distribution(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Demonstrate Gamma distribution for medical measurements and regularization.

        Returns:
            Tuple[Dict[str, Tensor], Tensor]: Dict of samples for each measurement, precisions tensor.
        """
        print("\n=== Gamma Distribution: Medical Measurements ===")
        
        # Model different medical measurements with Gamma distributions
        measurements = {
            "enzyme_levels": {"shape": 2.0, "rate": 0.5},      # Moderately skewed
            "reaction_times": {"shape": 1.5, "rate": 1.0},     # More skewed
            "recovery_periods": {"shape": 3.0, "rate": 0.3}    # Less skewed
        }
        samples_dict = {}
        for measurement, (shape, rate) in [(k, (v["shape"], v["rate"])) for k, v in measurements.items()]:
            gamma_dist = Gamma(concentration=shape, rate=rate)
            samples = gamma_dist.sample((1000,))
            samples_dict[measurement] = samples
            theoretical_mean = shape / rate
            theoretical_var = shape / (rate ** 2)
            empirical_mean = samples.mean().item()
            empirical_var = samples.var().item()
            print(f"\n{measurement.replace('_', ' ').title()}:")
            print(f"  Shape: {shape}, Rate: {rate}")
            print(f"  Mean - theoretical: {theoretical_mean:.3f}, empirical: {empirical_mean:.3f}")
            print(f"  Variance - theoretical: {theoretical_var:.3f}, empirical: {empirical_var:.3f}")
            mode = (shape - 1) / rate if shape > 1 else 0
            print(f"  Mode: {mode:.3f}")
            print(f"  Skewness: {(2 / math.sqrt(shape)):.3f}")
        print(f"\nGamma as precision prior in Bayesian neural networks:")
        precision_shape = 2.0
        precision_rate = 1.0
        precision_dist = Gamma(concentration=precision_shape, rate=precision_rate)
        precisions = precision_dist.sample((100,))
        weight_variances = 1.0 / precisions
        print(f"  Precision mean: {precisions.mean():.3f}")
        print(f"  Weight variance mean: {weight_variances.mean():.3f}")
        print(f"  Weight std mean: {torch.sqrt(weight_variances).mean():.3f}")
        return samples_dict, precisions
    
    def demonstrate_beta_distribution(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Demonstrate Beta distribution for modeling probabilities and attention weights.

        Returns:
            Tuple[Dict[str, Tensor], Tensor]: Dict of samples for each scenario, posterior distribution.
        """
        print("\n=== Beta Distribution: Probability Modeling ===")
        
        # Model different types of probability distributions
        probability_scenarios = {
            "diagnostic_accuracy": {"alpha": 8, "beta": 2},    # High accuracy (skewed right)
            "rare_disease_prevalence": {"alpha": 1, "beta": 9}, # Low prevalence (skewed left)
            "treatment_success": {"alpha": 3, "beta": 3},      # Moderate success (symmetric)
            "uncertain_outcome": {"alpha": 0.5, "beta": 0.5}   # U-shaped (very uncertain)
        }
        samples_dict = {}
        for scenario, (alpha, beta) in [(k, (v["alpha"], v["beta"])) for k, v in probability_scenarios.items()]:
            beta_dist = Beta(concentration1=alpha, concentration0=beta)
            samples = beta_dist.sample((1000,))
            samples_dict[scenario] = samples
            theoretical_mean = alpha / (alpha + beta)
            theoretical_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            empirical_mean = samples.mean().item()
            empirical_var = samples.var().item()
            print(f"\n{scenario.replace('_', ' ').title()}:")
            print(f"  Alpha: {alpha}, Beta: {beta}")
            print(f"  Mean - theoretical: {theoretical_mean:.3f}, empirical: {empirical_mean:.3f}")
            print(f"  Variance - theoretical: {theoretical_var:.3f}, empirical: {empirical_var:.3f}")
            if alpha > 1 and beta > 1:
                mode = (alpha - 1) / (alpha + beta - 2)
                print(f"  Mode: {mode:.3f}")
            elif alpha < 1 and beta < 1:
                print(f"  U-shaped distribution (modes at 0 and 1)")
            elif alpha < 1:
                print(f"  Mode at 0")
            elif beta < 1:
                print(f"  Mode at 1")
        print(f"\nBeta-Binomial Conjugacy Example:")
        print(f"Estimating treatment success rate with Bayesian updating")
        prior_alpha, prior_beta = 2, 2  # Weak prior, slightly optimistic
        prior_dist = Beta(concentration1=prior_alpha, concentration0=prior_beta)
        print(f"Prior: Beta({prior_alpha}, {prior_beta})")
        print(f"Prior mean success rate: {prior_alpha / (prior_alpha + prior_beta):.3f}")
        n_patients = 20
        n_successes = 14
        n_failures = n_patients - n_successes
        posterior_alpha = prior_alpha + n_successes
        posterior_beta = prior_beta + n_failures
        posterior_dist = Beta(concentration1=posterior_alpha, concentration0=posterior_beta)
        print(f"\nObserved: {n_successes} successes out of {n_patients} patients")
        print(f"Posterior: Beta({posterior_alpha}, {posterior_beta})")
        print(f"Posterior mean success rate: {posterior_alpha / (posterior_alpha + posterior_beta):.3f}")
        mle_estimate = n_successes / n_patients
        print(f"MLE estimate: {mle_estimate:.3f}")
        return samples_dict, posterior_dist
    
    def neural_network_uncertainty_quantification(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Demonstrate how continuous distributions enable uncertainty quantification in neural language models.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Sampled outputs, predictive mean, predictive std.
        """
        print("\n=== Neural Network Uncertainty Quantification ===")
        
        # Simulate a Bayesian neural network layer
        input_dim = 256
        output_dim = 128
        
        # Weight distributions (instead of point estimates)
        weight_mean = torch.zeros(output_dim, input_dim)
        weight_log_var = torch.full((output_dim, input_dim), -2.0)  # log(variance)
        weight_std = torch.exp(0.5 * weight_log_var)
        
        # Bias distributions
        bias_mean = torch.zeros(output_dim)
        bias_log_var = torch.full((output_dim,), -2.0)
        bias_std = torch.exp(0.5 * bias_log_var)
        
        print(f"Bayesian layer: {input_dim} -> {output_dim}")
        print(f"Weight std mean: {weight_std.mean():.4f}")
        print(f"Bias std mean: {bias_std.mean():.4f}")
        
        # Sample multiple weight configurations
        n_samples = 100
        input_batch = torch.randn(32, input_dim)  # Batch of inputs
        
        outputs = []
        for _ in range(n_samples):
            # Sample weights and biases
            weights = Normal(weight_mean, weight_std).sample()
            biases = Normal(bias_mean, bias_std).sample()
            
            # Forward pass
            output = torch.matmul(input_batch, weights.T) + biases
            outputs.append(output)
        
        # Stack outputs and analyze uncertainty
        outputs = torch.stack(outputs)  # Shape: (n_samples, batch_size, output_dim)
        
        # Compute predictive statistics
        predictive_mean = outputs.mean(dim=0)
        predictive_std = outputs.std(dim=0)
        
        print(f"\nPredictive uncertainty analysis:")
        print(f"  Mean output magnitude: {predictive_mean.abs().mean():.4f}")
        print(f"  Average uncertainty (std): {predictive_std.mean():.4f}")
        print(f"  Max uncertainty: {predictive_std.max():.4f}")
        print(f"  Min uncertainty: {predictive_std.min():.4f}")
        
        # Analyze uncertainty distribution
        uncertainty_flat = predictive_std.flatten()
        print(f"  Uncertainty distribution:")
        print(f"    25th percentile: {torch.quantile(uncertainty_flat, 0.25):.4f}")
        print(f"    50th percentile: {torch.quantile(uncertainty_flat, 0.50):.4f}")
        print(f"    75th percentile: {torch.quantile(uncertainty_flat, 0.75):.4f}")
        
        return outputs, predictive_mean, predictive_std
    
    def medical_dosage_modeling(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Demonstrate continuous distributions for medical dosage recommendations with uncertainty quantification.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Patient features, dosage predictions, and beta coefficients.
        """
        print("\n=== Medical Dosage Modeling with Continuous Distributions ===")
        
        # Model patient characteristics affecting dosage
        # Age, weight, kidney function, etc.
        
        # Patient population characteristics (multivariate normal)
        n_patients = 1000
        
        # Features: [age, weight, kidney_function, liver_function]
        feature_means = torch.tensor([65.0, 70.0, 80.0, 85.0])  # Typical values
        feature_cov = torch.tensor([
            [100.0, 20.0, -10.0, -5.0],   # Age variance and correlations
            [20.0, 225.0, 5.0, 10.0],     # Weight variance and correlations
            [-10.0, 5.0, 64.0, 15.0],     # Kidney function
            [-5.0, 10.0, 15.0, 49.0]      # Liver function
        ])
        
        patient_features = MultivariateNormal(feature_means, feature_cov).sample((n_patients,))
        
        print(f"Patient population: {n_patients} patients")
        print(f"Features: age, weight, kidney function, liver function")
        
        # Dosage model: log-normal distribution based on patient features
        # log(dosage) = β₀ + β₁*age + β₂*weight + β₃*kidney + β₄*liver + ε
        
        # Coefficients (learned from clinical data)
        beta_coefficients = torch.tensor([2.0, -0.01, 0.02, 0.01, 0.005])  # Including intercept
        
        # Add intercept term
        features_with_intercept = torch.cat([
            torch.ones(n_patients, 1), patient_features
        ], dim=1)
        
        # Linear combination
        linear_combination = torch.matmul(features_with_intercept, beta_coefficients)
        
        # Add noise (uncertainty in dosage prediction)
        noise_std = 0.2
        noise = Normal(0, noise_std).sample((n_patients,))
        log_dosage = linear_combination + noise
        
        # Convert to actual dosage (log-normal)
        dosage = torch.exp(log_dosage)
        
        print(f"\nDosage statistics:")
        print(f"  Mean dosage: {dosage.mean():.2f} mg")
        print(f"  Std dosage: {dosage.std():.2f} mg")
        print(f"  Median dosage: {torch.median(dosage):.2f} mg")
        print(f"  95% range: [{torch.quantile(dosage, 0.025):.2f}, {torch.quantile(dosage, 0.975):.2f}] mg")
        
        # Analyze dosage by patient subgroups
        elderly_mask = patient_features[:, 0] > 75  # Age > 75
        elderly_dosage = dosage[elderly_mask]
        young_dosage = dosage[~elderly_mask]
        
        print(f"\nAge-based analysis:")
        print(f"  Elderly (>75): mean = {elderly_dosage.mean():.2f} mg, n = {elderly_mask.sum()}")
        print(f"  Younger (≤75): mean = {young_dosage.mean():.2f} mg, n = {(~elderly_mask).sum()}")
        
        # Kidney function analysis
        low_kidney_mask = patient_features[:, 2] < 60  # Kidney function < 60
        low_kidney_dosage = dosage[low_kidney_mask]
        normal_kidney_dosage = dosage[~low_kidney_mask]
        
        print(f"\nKidney function analysis:")
        print(f"  Low function (<60): mean = {low_kidney_dosage.mean():.2f} mg, n = {low_kidney_mask.sum()}")
        print(f"  Normal function (≥60): mean = {normal_kidney_dosage.mean():.2f} mg, n = {(~low_kidney_mask).sum()}")
        
        return patient_features, dosage, beta_coefficients

def main():
    """
    Run all continuous distribution demonstrations.
    """
    print("Healthcare Language Modeling: Continuous Probability Distributions")
    print("=" * 75)
    demo = ContinuousDistributionsLM()
    for fn in [
        demo.demonstrate_normal_distribution,
        demo.demonstrate_multivariate_normal,
        demo.demonstrate_exponential_distribution,
        demo.demonstrate_gamma_distribution,
        demo.demonstrate_beta_distribution,
        demo.neural_network_uncertainty_quantification,
        demo.medical_dosage_modeling,
    ]:
        fn()
    print("\n" + "=" * 75)
    print("All continuous distribution demonstrations completed!")

if __name__ == "__main__":
    main()

