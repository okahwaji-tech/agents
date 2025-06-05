# Comprehensive PyTorch Implementation Examples for Gradient Descent
# From Basic Implementation to Advanced Healthcare Applications

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=== Comprehensive Gradient Descent Implementation Examples ===")
print("This module demonstrates gradient descent from basic concepts to advanced healthcare applications")
print()

# ============================================================================
# 1. BASIC GRADIENT DESCENT FROM SCRATCH
# ============================================================================

class BasicGradientDescent:
    """
    A from-scratch implementation of gradient descent for educational purposes.
    This implementation helps understand the core mechanics without PyTorch abstractions.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        self.parameter_history = []
    
    def linear_regression_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute the cost function for linear regression (Mean Squared Error).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            theta: Parameter vector (n_features,)
        
        Returns:
            Cost value (scalar)
        """
        m = X.shape[0]
        predictions = X @ theta
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def linear_regression_gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the cost function for linear regression.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            theta: Parameter vector (n_features,)
        
        Returns:
            Gradient vector (n_features,)
        """
        m = X.shape[0]
        predictions = X @ theta
        gradient = (1 / m) * X.T @ (predictions - y)
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Fit linear regression using gradient descent.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        
        Returns:
            Tuple of (optimal_parameters, cost_history)
        """
        # Add bias term (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize parameters
        theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        print(f"Starting gradient descent with learning rate: {self.learning_rate}")
        print(f"Initial parameters: {theta}")
        
        for iteration in range(self.max_iterations):
            # Compute cost and gradient
            cost = self.linear_regression_cost(X_with_bias, y, theta)
            gradient = self.linear_regression_gradient(X_with_bias, y, theta)
            
            # Store history
            self.cost_history.append(cost)
            self.parameter_history.append(theta.copy())
            
            # Update parameters
            theta_new = theta - self.learning_rate * gradient
            
            # Check for convergence
            if np.linalg.norm(theta_new - theta) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            theta = theta_new
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}, Gradient norm = {np.linalg.norm(gradient):.6f}")
        
        print(f"Final parameters: {theta}")
        print(f"Final cost: {self.cost_history[-1]:.6f}")
        
        return theta, self.cost_history

# Demonstrate basic gradient descent
print("1. BASIC GRADIENT DESCENT IMPLEMENTATION")
print("-" * 50)

# Generate synthetic regression data
X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Apply basic gradient descent
basic_gd = BasicGradientDescent(learning_rate=0.01, max_iterations=1000)
optimal_params, cost_history = basic_gd.fit(X_reg_scaled, y_reg)

print(f"Optimal parameters found: {optimal_params}")
print(f"Number of iterations: {len(cost_history)}")
print()

# ============================================================================
# 2. PYTORCH GRADIENT DESCENT IMPLEMENTATIONS
# ============================================================================

class LinearRegressionPyTorch(nn.Module):
    """
    Linear regression model implemented in PyTorch for comparison with basic implementation.
    """
    
    def __init__(self, input_dim: int):
        super(LinearRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def train_pytorch_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                       optimizer: optim.Optimizer, criterion: nn.Module, 
                       epochs: int = 1000) -> List[float]:
    """
    Train a PyTorch model using the specified optimizer.
    
    Args:
        model: PyTorch model to train
        X: Input features
        y: Target values
        optimizer: PyTorch optimizer
        criterion: Loss function
        epochs: Number of training epochs
    
    Returns:
        List of loss values during training
    """
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions.squeeze(), y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    return loss_history

print("2. PYTORCH GRADIENT DESCENT IMPLEMENTATIONS")
print("-" * 50)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X_reg_scaled)
y_tensor = torch.FloatTensor(y_reg)

# Create model
model = LinearRegressionPyTorch(input_dim=2)
criterion = nn.MSELoss()

# Test different optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'Adam': optim.Adam(model.parameters(), lr=0.01),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.01)
}

optimizer_results = {}

for name, optimizer in optimizers.items():
    print(f"\nTraining with {name} optimizer:")
    # Reset model parameters
    model.linear.weight.data.normal_(0, 0.01)
    model.linear.bias.data.zero_()
    
    loss_history = train_pytorch_model(model, X_tensor, y_tensor, optimizer, criterion, epochs=500)
    optimizer_results[name] = loss_history
    
    print(f"Final loss with {name}: {loss_history[-1]:.6f}")

print()

# ============================================================================
# 3. STOCHASTIC AND MINI-BATCH GRADIENT DESCENT
# ============================================================================

class StochasticGradientDescent:
    """
    Implementation of Stochastic Gradient Descent (SGD) and Mini-batch Gradient Descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, batch_size: Optional[int] = None, 
                 epochs: int = 100, shuffle: bool = True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # None for full batch, 1 for SGD, >1 for mini-batch
        self.epochs = epochs
        self.shuffle = shuffle
        self.loss_history = []
    
    def create_batches(self, X: torch.Tensor, y: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create batches for training."""
        n_samples = X.shape[0]
        
        if self.batch_size is None:
            # Full batch
            return [(X, y)]
        
        # Create indices
        indices = torch.randperm(n_samples) if self.shuffle else torch.arange(n_samples)
        
        batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches
    
    def train(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
              criterion: nn.Module) -> List[float]:
        """
        Train model using SGD or mini-batch gradient descent.
        """
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = self.create_batches(X, y)
            
            for batch_X, batch_y in batches:
                # Forward pass
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(batches)
            self.loss_history.append(avg_epoch_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.6f}")
        
        return self.loss_history

print("3. STOCHASTIC AND MINI-BATCH GRADIENT DESCENT")
print("-" * 50)

# Generate larger dataset for demonstrating batch effects
X_large, y_large = make_regression(n_samples=1000, n_features=5, noise=15, random_state=42)
X_large_scaled = StandardScaler().fit_transform(X_large)
X_large_tensor = torch.FloatTensor(X_large_scaled)
y_large_tensor = torch.FloatTensor(y_large)

# Test different batch sizes
batch_sizes = [None, 1, 32, 128]  # Full batch, SGD, Mini-batch 32, Mini-batch 128
batch_results = {}

for batch_size in batch_sizes:
    batch_name = "Full Batch" if batch_size is None else f"Batch Size {batch_size}"
    print(f"\nTraining with {batch_name}:")
    
    # Create fresh model
    model = LinearRegressionPyTorch(input_dim=5)
    criterion = nn.MSELoss()
    
    # Train with specific batch size
    sgd_trainer = StochasticGradientDescent(
        learning_rate=0.01, 
        batch_size=batch_size, 
        epochs=100, 
        shuffle=True
    )
    
    loss_history = sgd_trainer.train(model, X_large_tensor, y_large_tensor, criterion)
    batch_results[batch_name] = loss_history
    
    print(f"Final loss with {batch_name}: {loss_history[-1]:.6f}")

print()

# ============================================================================
# 4. ADVANCED OPTIMIZERS IMPLEMENTATION
# ============================================================================

class AdamOptimizer:
    """
    Custom implementation of Adam optimizer for educational purposes.
    """
    
    def __init__(self, parameters, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # time step
        
        # Initialize moment estimates
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]
    
    def step(self):
        """Perform one optimization step."""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
    
    def zero_grad(self):
        """Zero out gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

print("4. ADVANCED OPTIMIZERS IMPLEMENTATION")
print("-" * 50)

# Compare custom Adam with PyTorch Adam
model_custom = LinearRegressionPyTorch(input_dim=5)
model_pytorch = LinearRegressionPyTorch(input_dim=5)

# Make sure both models start with same parameters
model_pytorch.load_state_dict(model_custom.state_dict())

criterion = nn.MSELoss()

# Custom Adam optimizer
custom_adam = AdamOptimizer(model_custom.parameters(), lr=0.01)

# PyTorch Adam optimizer
pytorch_adam = optim.Adam(model_pytorch.parameters(), lr=0.01)

print("Training with Custom Adam implementation:")
custom_losses = []
for epoch in range(200):
    predictions = model_custom(X_large_tensor)
    loss = criterion(predictions.squeeze(), y_large_tensor)
    
    custom_adam.zero_grad()
    loss.backward()
    custom_adam.step()
    
    custom_losses.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print(f"Final loss with Custom Adam: {custom_losses[-1]:.6f}")

print("\nTraining with PyTorch Adam:")
pytorch_losses = []
for epoch in range(200):
    predictions = model_pytorch(X_large_tensor)
    loss = criterion(predictions.squeeze(), y_large_tensor)
    
    pytorch_adam.zero_grad()
    loss.backward()
    pytorch_adam.step()
    
    pytorch_losses.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print(f"Final loss with PyTorch Adam: {pytorch_losses[-1]:.6f}")
print()

# ============================================================================
# 5. HEALTHCARE-SPECIFIC EXAMPLES
# ============================================================================

class HealthcareRiskPredictor(nn.Module):
    """
    Neural network for predicting healthcare risks (e.g., readmission, complications).
    This model demonstrates gradient descent in a healthcare context.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1, dropout_rate: float = 0.2):
        super(HealthcareRiskPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_dim == 1:
            layers.append(nn.Sigmoid())  # For binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def generate_healthcare_data(n_patients: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Generate synthetic healthcare data for demonstration.
    Features represent patient characteristics and medical history.
    """
    np.random.seed(42)
    
    # Feature names for interpretability
    feature_names = [
        'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'cholesterol', 'glucose', 'heart_rate', 'previous_admissions',
        'comorbidity_count', 'medication_count', 'lab_abnormalities',
        'emergency_visits', 'length_of_stay', 'icu_days'
    ]
    
    # Generate realistic healthcare features
    age = np.random.normal(65, 15, n_patients)
    bmi = np.random.normal(28, 6, n_patients)
    bp_sys = np.random.normal(140, 20, n_patients)
    bp_dia = np.random.normal(85, 15, n_patients)
    cholesterol = np.random.normal(200, 40, n_patients)
    glucose = np.random.normal(110, 30, n_patients)
    heart_rate = np.random.normal(75, 12, n_patients)
    prev_admissions = np.random.poisson(2, n_patients)
    comorbidities = np.random.poisson(3, n_patients)
    medications = np.random.poisson(5, n_patients)
    lab_abnormal = np.random.poisson(1, n_patients)
    er_visits = np.random.poisson(1, n_patients)
    los = np.random.exponential(5, n_patients)
    icu_days = np.random.exponential(1, n_patients)
    
    # Combine features
    X = np.column_stack([
        age, bmi, bp_sys, bp_dia, cholesterol, glucose, heart_rate,
        prev_admissions, comorbidities, medications, lab_abnormal,
        er_visits, los, icu_days
    ])
    
    # Generate target (readmission risk) based on realistic relationships
    risk_score = (
        0.02 * age +
        0.05 * bmi +
        0.01 * bp_sys +
        0.3 * prev_admissions +
        0.2 * comorbidities +
        0.1 * medications +
        0.4 * lab_abnormal +
        0.3 * er_visits +
        0.1 * los +
        0.5 * icu_days +
        np.random.normal(0, 2, n_patients)  # Add noise
    )
    
    # Convert to binary classification (high risk vs low risk)
    y = (risk_score > np.median(risk_score)).astype(float)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return torch.FloatTensor(X_scaled), torch.FloatTensor(y), feature_names

class HealthcareTrainer:
    """
    Specialized trainer for healthcare models with additional monitoring and validation.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()  # Binary cross-entropy for classification
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate binary classification accuracy."""
        predicted_classes = (predictions > 0.5).float()
        correct = (predicted_classes == targets).float()
        return correct.mean().item()
    
    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                   X_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, float, float, float]:
        """Train for one epoch and return metrics."""
        
        # Training phase
        self.model.train()
        train_predictions = self.model(X_train)
        train_loss = self.criterion(train_predictions.squeeze(), y_train)
        train_accuracy = self.calculate_accuracy(train_predictions.squeeze(), y_train)
        
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        
        # Validation phase
        self.model.eval()
        with torch.no_grad():
            val_predictions = self.model(X_val)
            val_loss = self.criterion(val_predictions.squeeze(), y_val)
            val_accuracy = self.calculate_accuracy(val_predictions.squeeze(), y_val)
        
        return train_loss.item(), val_loss.item(), train_accuracy, val_accuracy
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, 
              X_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 100) -> Dict:
        """
        Train the healthcare model with comprehensive monitoring.
        """
        print("Training Healthcare Risk Prediction Model")
        print("-" * 50)
        
        for epoch in range(epochs):
            train_loss, val_loss, train_acc, val_acc = self.train_epoch(X_train, y_train, X_val, y_val)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        final_results = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        return final_results

print("5. HEALTHCARE-SPECIFIC GRADIENT DESCENT EXAMPLES")
print("-" * 50)

# Generate healthcare dataset
X_health, y_health, feature_names = generate_healthcare_data(n_patients=2000)

print(f"Generated healthcare dataset:")
print(f"Number of patients: {X_health.shape[0]}")
print(f"Number of features: {X_health.shape[1]}")
print(f"Features: {feature_names}")
print(f"High-risk patients: {y_health.sum().item():.0f} ({y_health.mean().item():.1%})")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_health, y_health, test_size=0.2, random_state=42, stratify=y_health
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} patients")
print(f"Validation set: {X_val.shape[0]} patients")
print(f"Test set: {X_test.shape[0]} patients")
print()

# Create and train healthcare model
healthcare_model = HealthcareRiskPredictor(
    input_dim=X_health.shape[1],
    hidden_dims=[64, 32, 16],
    output_dim=1,
    dropout_rate=0.3
)

trainer = HealthcareTrainer(healthcare_model, learning_rate=0.001, weight_decay=1e-4)
results = trainer.train(X_train, y_train, X_val, y_val, epochs=150)

print(f"\nFinal Training Results:")
print(f"Training Accuracy: {results['final_train_accuracy']:.4f}")
print(f"Validation Accuracy: {results['final_val_accuracy']:.4f}")
print(f"Training Loss: {results['final_train_loss']:.4f}")
print(f"Validation Loss: {results['final_val_loss']:.4f}")

# Test set evaluation
healthcare_model.eval()
with torch.no_grad():
    test_predictions = healthcare_model(X_test)
    test_loss = trainer.criterion(test_predictions.squeeze(), y_test)
    test_accuracy = trainer.calculate_accuracy(test_predictions.squeeze(), y_test)

print(f"\nTest Set Performance:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss.item():.4f}")
print()

# ============================================================================
# 6. GRADIENT DESCENT DIAGNOSTICS AND VISUALIZATION
# ============================================================================

class GradientDescentDiagnostics:
    """
    Tools for diagnosing and visualizing gradient descent behavior.
    """
    
    @staticmethod
    def analyze_convergence(loss_history: List[float], window_size: int = 10) -> Dict:
        """
        Analyze convergence characteristics of gradient descent.
        """
        if len(loss_history) < window_size:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate convergence metrics
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        total_reduction = initial_loss - final_loss
        relative_reduction = total_reduction / initial_loss
        
        # Check for convergence in final window
        final_window = loss_history[-window_size:]
        final_variance = np.var(final_window)
        final_trend = np.polyfit(range(window_size), final_window, 1)[0]
        
        # Detect oscillations
        differences = np.diff(loss_history)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        oscillation_ratio = sign_changes / len(differences)
        
        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "total_reduction": total_reduction,
            "relative_reduction": relative_reduction,
            "final_variance": final_variance,
            "final_trend": final_trend,
            "oscillation_ratio": oscillation_ratio,
            "converged": final_variance < 1e-6 and abs(final_trend) < 1e-6
        }
    
    @staticmethod
    def detect_problems(loss_history: List[float]) -> List[str]:
        """
        Detect common problems in gradient descent training.
        """
        problems = []
        
        if len(loss_history) < 10:
            return ["Insufficient training data"]
        
        # Check for divergence
        if loss_history[-1] > loss_history[0]:
            problems.append("Divergence detected - loss increased overall")
        
        # Check for exploding gradients
        recent_losses = loss_history[-10:]
        if any(loss > 1e6 for loss in recent_losses):
            problems.append("Possible exploding gradients - very large loss values")
        
        # Check for vanishing gradients (no progress)
        if len(loss_history) > 50:
            recent_change = abs(loss_history[-1] - loss_history[-50])
            if recent_change < 1e-8:
                problems.append("Possible vanishing gradients - no recent progress")
        
        # Check for oscillations
        differences = np.diff(loss_history)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        if sign_changes > len(differences) * 0.7:
            problems.append("High oscillation - consider reducing learning rate")
        
        # Check for plateaus
        if len(loss_history) > 20:
            recent_losses = loss_history[-20:]
            if np.std(recent_losses) < 1e-6:
                problems.append("Training plateau - consider adjusting learning rate or architecture")
        
        return problems if problems else ["No obvious problems detected"]

print("6. GRADIENT DESCENT DIAGNOSTICS AND VISUALIZATION")
print("-" * 50)

# Analyze the healthcare model training
diagnostics = GradientDescentDiagnostics()

# Analyze training convergence
train_analysis = diagnostics.analyze_convergence(results['train_losses'])
val_analysis = diagnostics.analyze_convergence(results['val_losses'])

print("Training Loss Analysis:")
for key, value in train_analysis.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

print("\nValidation Loss Analysis:")
for key, value in val_analysis.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Detect problems
train_problems = diagnostics.detect_problems(results['train_losses'])
val_problems = diagnostics.detect_problems(results['val_losses'])

print(f"\nTraining Problems Detected:")
for problem in train_problems:
    print(f"  - {problem}")

print(f"\nValidation Problems Detected:")
for problem in val_problems:
    print(f"  - {problem}")

print()

# ============================================================================
# 7. SUMMARY AND BEST PRACTICES
# ============================================================================

print("7. SUMMARY AND BEST PRACTICES")
print("-" * 50)

best_practices = [
    "1. Always monitor both training and validation metrics to detect overfitting",
    "2. Use appropriate learning rates - start with 0.001 for Adam, 0.01 for SGD",
    "3. Implement early stopping based on validation performance",
    "4. Use regularization (weight decay, dropout) for healthcare models",
    "5. Standardize/normalize input features for stable training",
    "6. Choose batch sizes based on available memory and dataset size",
    "7. Monitor gradient norms to detect vanishing/exploding gradients",
    "8. Use learning rate scheduling for better convergence",
    "9. Implement comprehensive logging and diagnostics",
    "10. Validate model performance on held-out test sets"
]

print("Best Practices for Gradient Descent in Healthcare ML:")
for practice in best_practices:
    print(f"  {practice}")

print(f"\nImplementation completed successfully!")
print(f"All gradient descent variants and healthcare examples have been demonstrated.")
print(f"The code provides a comprehensive foundation for understanding and implementing")
print(f"gradient descent in healthcare machine learning applications.")

