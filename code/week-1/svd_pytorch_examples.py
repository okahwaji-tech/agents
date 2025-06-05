# Comprehensive PyTorch Implementation Examples for SVD
# Healthcare-Focused Singular Value Decomposition Applications

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("COMPREHENSIVE SVD IMPLEMENTATIONS FOR HEALTHCARE APPLICATIONS")
print("=" * 80)

class BasicSVD:
    """
    Basic SVD implementation with healthcare-focused examples.
    Demonstrates fundamental SVD concepts using PyTorch.
    """
    
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None
        
    def compute_svd(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SVD of input matrix using PyTorch's built-in function.
        
        Args:
            matrix: Input tensor of shape (m, n)
            
        Returns:
            U, S, V tensors representing the SVD decomposition
        """
        U, S, V = torch.svd(matrix)
        self.U, self.S, self.V = U, S, V
        return U, S, V
    
    def reconstruct_matrix(self, k: Optional[int] = None) -> torch.Tensor:
        """
        Reconstruct matrix using first k singular values.
        
        Args:
            k: Number of singular values to use (None for all)
            
        Returns:
            Reconstructed matrix
        """
        if self.U is None:
            raise ValueError("Must compute SVD first")
            
        if k is None:
            k = len(self.S)
        
        # Truncate to first k components
        U_k = self.U[:, :k]
        S_k = self.S[:k]
        V_k = self.V[:, :k]
        
        # Reconstruct: A ≈ U_k * S_k * V_k^T
        reconstructed = U_k @ torch.diag(S_k) @ V_k.T
        return reconstructed
    
    def explained_variance_ratio(self) -> torch.Tensor:
        """
        Calculate the proportion of variance explained by each singular value.
        
        Returns:
            Tensor of explained variance ratios
        """
        if self.S is None:
            raise ValueError("Must compute SVD first")
            
        total_variance = torch.sum(self.S ** 2)
        explained_variance = (self.S ** 2) / total_variance
        return explained_variance

# Example 1: Basic SVD on Synthetic Patient Data
print("\n1. BASIC SVD ON SYNTHETIC PATIENT DATA")
print("-" * 50)

# Create synthetic patient data
# Rows: patients, Columns: biomarkers (height, weight, blood pressure, heart rate, cholesterol)
n_patients = 100
n_biomarkers = 5

# Generate correlated biomarker data
np.random.seed(42)
base_data = np.random.randn(n_patients, 2)  # Two underlying factors
biomarker_loadings = np.array([
    [1.2, 0.3],    # Height
    [1.1, 0.4],    # Weight  
    [0.8, 1.2],    # Blood pressure
    [0.6, 1.1],    # Heart rate
    [0.9, 0.8]     # Cholesterol
])

patient_data = base_data @ biomarker_loadings.T + 0.1 * np.random.randn(n_patients, n_biomarkers)

# Add meaningful structure: age effect
ages = np.random.uniform(20, 80, n_patients)
age_effect = np.outer((ages - 50) / 30, [0.1, 0.2, 0.3, -0.1, 0.2])
patient_data += age_effect

# Convert to PyTorch tensor
patient_tensor = torch.tensor(patient_data, dtype=torch.float32)

# Standardize the data
patient_mean = torch.mean(patient_tensor, dim=0)
patient_std = torch.std(patient_tensor, dim=0)
patient_standardized = (patient_tensor - patient_mean) / patient_std

print(f"Patient data shape: {patient_standardized.shape}")
print(f"Biomarkers: Height, Weight, Blood Pressure, Heart Rate, Cholesterol")

# Compute SVD
svd_basic = BasicSVD()
U, S, V = svd_basic.compute_svd(patient_standardized)

print(f"\nSVD Results:")
print(f"U (left singular vectors) shape: {U.shape}")
print(f"S (singular values) shape: {S.shape}")
print(f"V (right singular vectors) shape: {V.shape}")

# Analyze singular values
explained_var = svd_basic.explained_variance_ratio()
print(f"\nExplained variance ratio by component:")
for i, var_ratio in enumerate(explained_var):
    print(f"Component {i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")

cumulative_var = torch.cumsum(explained_var, dim=0)
print(f"\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_var):
    print(f"First {i+1} components: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Demonstrate reconstruction with different numbers of components
print(f"\nReconstruction Quality (Frobenius norm error):")
original_norm = torch.norm(patient_standardized, 'fro')
for k in [1, 2, 3, 4, 5]:
    reconstructed = svd_basic.reconstruct_matrix(k)
    error = torch.norm(patient_standardized - reconstructed, 'fro')
    relative_error = error / original_norm
    print(f"Using {k} components: {relative_error:.6f} ({relative_error*100:.4f}%)")

class HealthcarePCA:
    """
    Principal Component Analysis implementation using SVD for healthcare data.
    Provides interpretable components for medical analysis.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X: torch.Tensor) -> 'HealthcarePCA':
        """
        Fit PCA model to healthcare data.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        # Center and standardize the data
        self.mean_ = torch.mean(X, dim=0)
        self.std_ = torch.std(X, dim=0)
        X_standardized = (X - self.mean_) / self.std_
        
        # Compute SVD
        U, S, V = torch.svd(X_standardized)
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = min(X.shape)
        
        # Store components (principal directions)
        self.components_ = V[:, :self.n_components].T
        
        # Calculate explained variance
        total_variance = torch.sum(S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform data to principal component space.
        
        Args:
            X: Input data tensor
            
        Returns:
            Transformed data in PC space
        """
        if self.components_ is None:
            raise ValueError("Must fit model first")
            
        X_standardized = (X - self.mean_) / self.std_
        return X_standardized @ self.components_.T
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit model and transform data in one step.
        
        Args:
            X: Input data tensor
            
        Returns:
            Transformed data in PC space
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: torch.Tensor) -> torch.Tensor:
        """
        Transform data back to original space.
        
        Args:
            X_transformed: Data in PC space
            
        Returns:
            Data in original space
        """
        if self.components_ is None:
            raise ValueError("Must fit model first")
            
        X_reconstructed = X_transformed @ self.components_
        return X_reconstructed * self.std_ + self.mean_

# Example 2: Healthcare PCA Analysis
print("\n\n2. HEALTHCARE PCA ANALYSIS")
print("-" * 50)

# Create more realistic healthcare dataset
def generate_healthcare_data(n_patients: int = 500) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Generate synthetic healthcare dataset with realistic correlations.
    
    Returns:
        patient_data: Tensor of patient measurements
        feature_names: List of measurement names
        patient_labels: Disease severity labels
    """
    np.random.seed(42)
    
    # Define underlying health factors
    # Factor 1: Cardiovascular health
    # Factor 2: Metabolic health  
    # Factor 3: Age-related decline
    
    health_factors = np.random.randn(n_patients, 3)
    
    # Define how each measurement relates to health factors
    measurement_loadings = np.array([
        [0.8, 0.3, 0.4],   # Systolic BP
        [0.7, 0.4, 0.3],   # Diastolic BP
        [-0.6, 0.2, 0.5],  # Heart Rate Variability
        [0.2, 0.9, 0.1],   # BMI
        [0.3, 0.8, 0.2],   # Waist Circumference
        [0.1, 0.7, 0.3],   # Blood Glucose
        [0.4, 0.6, 0.4],   # Cholesterol
        [0.2, 0.1, 0.8],   # Age
        [-0.3, -0.4, 0.6], # Cognitive Score
        [0.5, 0.3, 0.7]    # Inflammation Marker
    ])
    
    # Generate measurements
    measurements = health_factors @ measurement_loadings.T
    
    # Add measurement noise
    noise = 0.3 * np.random.randn(n_patients, measurement_loadings.shape[0])
    measurements += noise
    
    # Add realistic ranges and units
    measurement_scales = np.array([120, 80, 50, 25, 90, 100, 200, 50, 100, 5])
    measurement_offsets = np.array([120, 80, 50, 25, 90, 100, 200, 50, 100, 5])
    
    measurements = measurements * measurement_scales + measurement_offsets
    
    # Ensure realistic ranges
    measurements = np.clip(measurements, 
                          [90, 60, 20, 18, 60, 70, 150, 18, 60, 1],
                          [180, 120, 80, 40, 120, 200, 300, 90, 100, 15])
    
    feature_names = [
        'Systolic_BP', 'Diastolic_BP', 'HRV', 'BMI', 'Waist_Circ',
        'Blood_Glucose', 'Cholesterol', 'Age', 'Cognitive_Score', 'CRP'
    ]
    
    # Create disease severity labels based on health factors
    disease_severity = np.sum(health_factors * [1, 1, 0.5], axis=1)
    severity_labels = torch.tensor((disease_severity > np.percentile(disease_severity, 75)).astype(int))
    
    return torch.tensor(measurements, dtype=torch.float32), feature_names, severity_labels

# Generate healthcare dataset
healthcare_data, feature_names, severity_labels = generate_healthcare_data(500)
print(f"Healthcare dataset shape: {healthcare_data.shape}")
print(f"Features: {feature_names}")
print(f"High-risk patients: {torch.sum(severity_labels).item()}/{len(severity_labels)}")

# Apply PCA
pca = HealthcarePCA(n_components=5)
healthcare_pcs = pca.fit_transform(healthcare_data)

print(f"\nPCA Results:")
print(f"Explained variance ratio:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")

cumulative_var = torch.cumsum(pca.explained_variance_ratio_, dim=0)
print(f"\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_var):
    print(f"First {i+1} PCs: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Analyze component loadings
print(f"\nPrincipal Component Loadings:")
print(f"{'Feature':<15} {'PC1':<8} {'PC2':<8} {'PC3':<8} {'PC4':<8} {'PC5':<8}")
print("-" * 65)
for i, feature in enumerate(feature_names):
    loadings = pca.components_[:, i]
    print(f"{feature:<15} {loadings[0]:.4f}  {loadings[1]:.4f}  {loadings[2]:.4f}  {loadings[3]:.4f}  {loadings[4]:.4f}")

class MedicalImageSVD:
    """
    SVD-based analysis for medical imaging data.
    Demonstrates dimensionality reduction and noise filtering for medical images.
    """
    
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None
        self.original_shape = None
        
    def fit(self, image: torch.Tensor) -> 'MedicalImageSVD':
        """
        Fit SVD to medical image data.
        
        Args:
            image: 2D image tensor
            
        Returns:
            Self for method chaining
        """
        if len(image.shape) != 2:
            raise ValueError("Image must be 2D")
            
        self.original_shape = image.shape
        self.U, self.S, self.V = torch.svd(image)
        return self
    
    def compress(self, k: int) -> torch.Tensor:
        """
        Compress image using first k singular values.
        
        Args:
            k: Number of singular values to retain
            
        Returns:
            Compressed image
        """
        if self.U is None:
            raise ValueError("Must fit SVD first")
            
        U_k = self.U[:, :k]
        S_k = self.S[:k]
        V_k = self.V[:, :k]
        
        compressed = U_k @ torch.diag(S_k) @ V_k.T
        return compressed
    
    def denoise(self, noise_threshold: float = 0.1) -> torch.Tensor:
        """
        Denoise image by removing small singular values.
        
        Args:
            noise_threshold: Threshold for removing small singular values
            
        Returns:
            Denoised image
        """
        if self.S is None:
            raise ValueError("Must fit SVD first")
            
        # Find singular values above threshold
        max_sv = torch.max(self.S)
        keep_indices = self.S > (noise_threshold * max_sv)
        k = torch.sum(keep_indices).item()
        
        return self.compress(k)
    
    def compression_ratio(self, k: int) -> float:
        """
        Calculate compression ratio for given number of components.
        
        Args:
            k: Number of singular values
            
        Returns:
            Compression ratio
        """
        if self.original_shape is None:
            raise ValueError("Must fit SVD first")
            
        original_size = self.original_shape[0] * self.original_shape[1]
        compressed_size = k * (self.original_shape[0] + self.original_shape[1] + 1)
        return original_size / compressed_size

# Example 3: Medical Image Analysis with SVD
print("\n\n3. MEDICAL IMAGE ANALYSIS WITH SVD")
print("-" * 50)

# Create synthetic medical image (e.g., chest X-ray)
def create_synthetic_medical_image(size: int = 128) -> torch.Tensor:
    """
    Create a synthetic medical image with realistic structure.
    
    Args:
        size: Image size (size x size)
        
    Returns:
        Synthetic medical image tensor
    """
    np.random.seed(42)
    
    # Create base anatomical structure
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Chest cavity (elliptical)
    chest = np.exp(-(x**2/0.8 + y**2/1.2))
    
    # Lung regions
    lung_left = np.exp(-((x+0.3)**2/0.3 + (y-0.1)**2/0.6)) * 0.7
    lung_right = np.exp(-((x-0.3)**2/0.3 + (y-0.1)**2/0.6)) * 0.7
    
    # Heart region
    heart = np.exp(-((x+0.1)**2/0.2 + (y+0.3)**2/0.3)) * 0.5
    
    # Combine structures
    image = chest + lung_left + lung_right + heart
    
    # Add some pathological features (e.g., nodules)
    nodule1 = np.exp(-((x-0.2)**2/0.05 + (y+0.2)**2/0.05)) * 0.3
    nodule2 = np.exp(-((x+0.4)**2/0.03 + (y-0.4)**2/0.03)) * 0.2
    
    image += nodule1 + nodule2
    
    # Add noise
    noise = 0.1 * np.random.randn(size, size)
    image += noise
    
    # Normalize to [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return torch.tensor(image, dtype=torch.float32)

# Generate synthetic medical image
medical_image = create_synthetic_medical_image(128)
print(f"Medical image shape: {medical_image.shape}")
print(f"Image value range: [{torch.min(medical_image):.4f}, {torch.max(medical_image):.4f}]")

# Apply SVD to medical image
image_svd = MedicalImageSVD()
image_svd.fit(medical_image)

print(f"\nSVD decomposition:")
print(f"Number of singular values: {len(image_svd.S)}")
print(f"Largest singular values: {image_svd.S[:10].tolist()}")

# Analyze compression performance
print(f"\nCompression Analysis:")
compression_levels = [5, 10, 20, 50, 100]
for k in compression_levels:
    compressed_image = image_svd.compress(k)
    
    # Calculate reconstruction error
    error = torch.norm(medical_image - compressed_image, 'fro')
    relative_error = error / torch.norm(medical_image, 'fro')
    
    # Calculate compression ratio
    comp_ratio = image_svd.compression_ratio(k)
    
    print(f"k={k:3d}: Compression ratio={comp_ratio:.2f}x, "
          f"Relative error={relative_error:.6f} ({relative_error*100:.4f}%)")

# Demonstrate denoising
denoised_image = image_svd.denoise(noise_threshold=0.05)
denoising_error = torch.norm(medical_image - denoised_image, 'fro')
relative_denoising_error = denoising_error / torch.norm(medical_image, 'fro')
print(f"\nDenoising (threshold=0.05): Relative error={relative_denoising_error:.6f}")

class HealthcareMatrixCompletion:
    """
    Matrix completion for healthcare data with missing values.
    Uses SVD-based iterative algorithms to impute missing medical measurements.
    """
    
    def __init__(self, max_rank: int = 10, max_iter: int = 100, tol: float = 1e-6):
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.tol = tol
        self.completed_matrix = None
        
    def fit(self, X: torch.Tensor, mask: torch.Tensor) -> 'HealthcareMatrixCompletion':
        """
        Complete matrix using iterative SVD.
        
        Args:
            X: Partially observed matrix
            mask: Boolean mask indicating observed entries
            
        Returns:
            Self for method chaining
        """
        # Initialize with mean imputation
        X_filled = X.clone()
        col_means = torch.nanmean(X, dim=0)
        
        for j in range(X.shape[1]):
            missing_mask = ~mask[:, j]
            X_filled[missing_mask, j] = col_means[j]
        
        # Iterative SVD completion
        for iteration in range(self.max_iter):
            X_old = X_filled.clone()
            
            # SVD of current estimate
            U, S, V = torch.svd(X_filled)
            
            # Truncate to max_rank
            rank = min(self.max_rank, len(S))
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = V[:, :rank]
            
            # Reconstruct
            X_filled = U_r @ torch.diag(S_r) @ V_r.T
            
            # Keep observed entries unchanged
            X_filled[mask] = X[mask]
            
            # Check convergence
            change = torch.norm(X_filled - X_old, 'fro')
            if change < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        self.completed_matrix = X_filled
        return self
    
    def transform(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Complete a new matrix using the fitted model.
        
        Args:
            X: Partially observed matrix
            mask: Boolean mask indicating observed entries
            
        Returns:
            Completed matrix
        """
        # For simplicity, just apply the same algorithm
        # In practice, you might want to use the learned structure
        temp_completion = HealthcareMatrixCompletion(
            self.max_rank, self.max_iter, self.tol
        )
        return temp_completion.fit(X, mask).completed_matrix

# Example 4: Missing Data Imputation in Healthcare
print("\n\n4. MISSING DATA IMPUTATION IN HEALTHCARE")
print("-" * 50)

# Create healthcare dataset with missing values
def create_missing_data_scenario(data: torch.Tensor, missing_rate: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create missing data scenario for healthcare dataset.
    
    Args:
        data: Complete healthcare data
        missing_rate: Proportion of values to make missing
        
    Returns:
        data_with_missing: Data with missing values (NaN)
        mask: Boolean mask indicating observed values
    """
    np.random.seed(42)
    
    # Create missing data pattern (MCAR - Missing Completely At Random)
    mask = torch.rand_like(data) > missing_rate
    
    # Create data with missing values
    data_with_missing = data.clone()
    data_with_missing[~mask] = float('nan')
    
    return data_with_missing, mask

# Use the healthcare data from earlier
original_data = healthcare_data[:100, :8]  # Use subset for demonstration
missing_data, observed_mask = create_missing_data_scenario(original_data, missing_rate=0.25)

print(f"Original data shape: {original_data.shape}")
print(f"Missing data rate: {(~observed_mask).float().mean():.2%}")
print(f"Missing values per feature:")
for i, feature in enumerate(feature_names[:8]):
    missing_count = (~observed_mask[:, i]).sum().item()
    print(f"  {feature}: {missing_count}/{len(observed_mask)} ({missing_count/len(observed_mask):.1%})")

# Replace NaN with zeros for SVD computation (will be masked)
missing_data_clean = missing_data.clone()
missing_data_clean[torch.isnan(missing_data_clean)] = 0

# Apply matrix completion
completion = HealthcareMatrixCompletion(max_rank=5, max_iter=50)
completed_data = completion.fit(missing_data_clean, observed_mask).completed_matrix

# Evaluate completion quality
missing_mask = ~observed_mask
if missing_mask.sum() > 0:
    # Calculate imputation error on missing values
    true_missing = original_data[missing_mask]
    imputed_missing = completed_data[missing_mask]
    
    imputation_error = torch.norm(true_missing - imputed_missing)
    relative_imputation_error = imputation_error / torch.norm(true_missing)
    
    print(f"\nImputation Results:")
    print(f"Absolute error on missing values: {imputation_error:.4f}")
    print(f"Relative error on missing values: {relative_imputation_error:.4f} ({relative_imputation_error*100:.2f}%)")
    
    # Calculate feature-wise errors
    print(f"\nFeature-wise imputation errors:")
    for i, feature in enumerate(feature_names[:8]):
        feature_missing_mask = missing_mask[:, i]
        if feature_missing_mask.sum() > 0:
            feature_true = original_data[feature_missing_mask, i]
            feature_imputed = completed_data[feature_missing_mask, i]
            feature_error = torch.norm(feature_true - feature_imputed) / torch.norm(feature_true)
            print(f"  {feature}: {feature_error:.4f} ({feature_error*100:.2f}%)")

class HealthcareTensorSVD:
    """
    Higher-order SVD for healthcare tensor data.
    Handles multi-way data such as patients × biomarkers × time.
    """
    
    def __init__(self):
        self.core_tensor = None
        self.factor_matrices = []
        self.original_shape = None
        
    def hosvd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute Higher-Order SVD (Tucker decomposition) of tensor.
        
        Args:
            tensor: Input tensor of arbitrary order
            
        Returns:
            core_tensor: Core tensor
            factor_matrices: List of factor matrices for each mode
        """
        self.original_shape = tensor.shape
        ndim = len(tensor.shape)
        
        # Compute SVD for each mode
        factor_matrices = []
        
        for mode in range(ndim):
            # Unfold tensor along mode
            unfolded = self._unfold_tensor(tensor, mode)
            
            # Compute SVD
            U, S, V = torch.svd(unfolded)
            factor_matrices.append(U)
        
        # Compute core tensor
        core_tensor = tensor.clone()
        for mode in range(ndim):
            core_tensor = self._mode_product(core_tensor, factor_matrices[mode].T, mode)
        
        self.core_tensor = core_tensor
        self.factor_matrices = factor_matrices
        
        return core_tensor, factor_matrices
    
    def _unfold_tensor(self, tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """
        Unfold tensor along specified mode.
        
        Args:
            tensor: Input tensor
            mode: Mode to unfold along
            
        Returns:
            Unfolded matrix
        """
        # Move the specified mode to the front
        dims = list(range(len(tensor.shape)))
        dims[0], dims[mode] = dims[mode], dims[0]
        
        # Permute and reshape
        permuted = tensor.permute(dims)
        unfolded = permuted.reshape(tensor.shape[mode], -1)
        
        return unfolded
    
    def _mode_product(self, tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
        """
        Compute mode-n product of tensor with matrix.
        
        Args:
            tensor: Input tensor
            matrix: Matrix to multiply
            mode: Mode for multiplication
            
        Returns:
            Result tensor
        """
        # Get tensor dimensions
        tensor_shape = list(tensor.shape)
        
        # Move the mode to the first dimension
        dims_order = [mode] + [i for i in range(len(tensor_shape)) if i != mode]
        tensor_permuted = tensor.permute(dims_order)
        
        # Reshape to matrix: mode_size x (product of other dimensions)
        mode_size = tensor_shape[mode]
        other_size = tensor.numel() // mode_size
        tensor_matrix = tensor_permuted.reshape(mode_size, other_size)
        
        # Matrix multiplication
        result_matrix = matrix @ tensor_matrix
        
        # Reshape back to tensor
        new_shape = [matrix.shape[0]] + [tensor_shape[i] for i in range(len(tensor_shape)) if i != mode]
        result_tensor = result_matrix.reshape(new_shape)
        
        # Permute back to original order
        inverse_dims = [0] * len(new_shape)
        inverse_dims[mode] = 0
        j = 1
        for i in range(len(tensor_shape)):
            if i != mode:
                inverse_dims[i] = j
                j += 1
        
        result = result_tensor.permute(inverse_dims)
        
        return result
    
    def compress_tensor(self, ranks: List[int]) -> torch.Tensor:
        """
        Compress tensor using specified ranks for each mode.
        
        Args:
            ranks: List of ranks for each mode
            
        Returns:
            Compressed tensor
        """
        if self.core_tensor is None:
            raise ValueError("Must compute HOSVD first")
        
        # Truncate factor matrices
        truncated_factors = []
        for i, rank in enumerate(ranks):
            truncated_factors.append(self.factor_matrices[i][:, :rank])
        
        # Truncate core tensor
        core_slices = [slice(None, rank) for rank in ranks]
        truncated_core = self.core_tensor[tuple(core_slices)]
        
        # Reconstruct tensor
        result = truncated_core
        for mode in range(len(ranks)):
            result = self._mode_product(result, truncated_factors[mode], mode)
        
        return result

# Example 5: Tensor Analysis for Longitudinal Healthcare Data
print("\n\n5. TENSOR ANALYSIS FOR LONGITUDINAL HEALTHCARE DATA")
print("-" * 50)

# Create synthetic longitudinal healthcare data
def create_longitudinal_data(n_patients: int = 50, n_biomarkers: int = 6, n_timepoints: int = 12) -> torch.Tensor:
    """
    Create synthetic longitudinal healthcare data.
    
    Args:
        n_patients: Number of patients
        n_biomarkers: Number of biomarkers
        n_timepoints: Number of time points
        
    Returns:
        Tensor of shape (patients, biomarkers, time)
    """
    np.random.seed(42)
    
    # Create patient-specific baseline values
    patient_baselines = np.random.randn(n_patients, n_biomarkers)
    
    # Create time-dependent trends
    time_trends = np.array([
        [0.1, -0.05, 0.02, 0.08, -0.03, 0.06],  # Linear trends
        [0.02, 0.01, -0.01, 0.03, 0.02, -0.02]  # Quadratic trends
    ])
    
    # Generate data
    data = np.zeros((n_patients, n_biomarkers, n_timepoints))
    
    for t in range(n_timepoints):
        # Linear and quadratic time effects
        time_effect = time_trends[0] * t + time_trends[1] * t**2
        
        # Add seasonal variation
        seasonal_effect = 0.1 * np.sin(2 * np.pi * t / 12) * np.array([1, 0.5, 0.8, 0.3, 0.6, 0.4])
        
        # Combine effects
        for p in range(n_patients):
            data[p, :, t] = (patient_baselines[p, :] + 
                           time_effect + 
                           seasonal_effect + 
                           0.2 * np.random.randn(n_biomarkers))
    
    return torch.tensor(data, dtype=torch.float32)

# Generate longitudinal data
longitudinal_data = create_longitudinal_data(50, 6, 12)
print(f"Longitudinal data shape: {longitudinal_data.shape}")
print(f"Dimensions: {longitudinal_data.shape[0]} patients × {longitudinal_data.shape[1]} biomarkers × {longitudinal_data.shape[2]} timepoints")

# Apply Higher-Order SVD
tensor_svd = HealthcareTensorSVD()
core_tensor, factor_matrices = tensor_svd.hosvd(longitudinal_data)

print(f"\nHOSVD Results:")
print(f"Core tensor shape: {core_tensor.shape}")
print(f"Factor matrix shapes:")
for i, factor in enumerate(factor_matrices):
    mode_names = ['Patients', 'Biomarkers', 'Time']
    print(f"  {mode_names[i]} factor: {factor.shape}")

# Analyze compression potential
original_size = torch.numel(longitudinal_data)
print(f"\nOriginal tensor size: {original_size} elements")

# Test different compression levels
compression_ranks = [
    [10, 3, 6],   # Moderate compression
    [20, 4, 8],   # Light compression
    [5, 2, 4]     # Heavy compression
]

for ranks in compression_ranks:
    compressed_tensor = tensor_svd.compress_tensor(ranks)
    
    # Calculate compression ratio
    compressed_size = (torch.numel(core_tensor[:ranks[0], :ranks[1], :ranks[2]]) + 
                      sum(factor.shape[0] * rank for factor, rank in zip(factor_matrices, ranks)))
    compression_ratio = original_size / compressed_size
    
    # Calculate reconstruction error
    error = torch.norm(longitudinal_data - compressed_tensor, 'fro')
    relative_error = error / torch.norm(longitudinal_data, 'fro')
    
    print(f"Ranks {ranks}: Compression ratio={compression_ratio:.2f}x, "
          f"Relative error={relative_error:.6f} ({relative_error*100:.4f}%)")

print("\n" + "=" * 80)
print("SVD IMPLEMENTATION EXAMPLES COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nKey Takeaways:")
print("1. Basic SVD provides optimal low-rank approximations")
print("2. PCA via SVD reveals interpretable patterns in healthcare data")
print("3. SVD enables effective medical image compression and denoising")
print("4. Matrix completion handles missing healthcare data")
print("5. Tensor SVD analyzes multi-way longitudinal medical data")
print("\nThese implementations demonstrate the versatility and power of SVD")
print("for healthcare machine learning applications.")

