"""Matrix Multiplication for Large Language Models.

This module provides comprehensive PyTorch and NumPy implementations for matrix
multiplication operations fundamental to large language model development, including
basic operations, batch processing, GPU acceleration, and healthcare applications.

The module follows object-oriented design principles with a unified base class
and Google-style docstrings for better maintainability and documentation.

Example:
    Basic usage of the matrix multiplication calculators:

    ```python
    from matrix_multiplication import MatrixMultiplicationCalculator
    
    # Initialize calculator
    calc = MatrixMultiplicationCalculator(device='cuda')
    
    # Basic matrix multiplication
    A = torch.randn(100, 200)
    B = torch.randn(200, 150)
    C = calc.multiply(A, B)
    ```

Author: LLM Learning Guide
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any
from enum import Enum


class DeviceType(Enum):
    """Enumeration for computation devices.
    
    Attributes:
        CPU: Central Processing Unit computation
        CUDA: NVIDIA GPU computation with CUDA
        MPS: Apple Silicon GPU computation (Metal Performance Shaders)
        AUTO: Automatically select best available device
    """
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'
    AUTO = 'auto'


class MatrixMultiplicationMethod(Enum):
    """Enumeration for matrix multiplication methods.
    
    Attributes:
        STANDARD: Standard @ operator (recommended)
        TORCH_MM: torch.mm() for 2D matrices only
        TORCH_MATMUL: torch.matmul() with broadcasting
        TORCH_BMM: torch.bmm() for batch operations
        NUMPY_DOT: NumPy dot product
    """
    STANDARD = '@'
    TORCH_MM = 'torch.mm'
    TORCH_MATMUL = 'torch.matmul'
    TORCH_BMM = 'torch.bmm'
    NUMPY_DOT = 'numpy.dot'


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
def get_optimal_device() -> torch.device:
    """Get the optimal computation device available.
    
    Returns:
        The best available device in order: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_optimal_device()


class BaseMatrixOperations(ABC):
    """Abstract base class for matrix operations.

    This class provides common functionality for all matrix operation
    calculators including device handling and performance monitoring.

    Attributes:
        device: Computation device (CPU, CUDA, MPS)
        dtype: Default tensor data type
        timing_enabled: Whether to track operation timing
    """

    def __init__(self, device: Union[str, DeviceType, torch.device] = DeviceType.AUTO,
                 dtype: torch.dtype = torch.float32,
                 timing_enabled: bool = False):
        """Initialize the base matrix operations calculator.

        Args:
            device: Computation device. Can be 'cpu', 'cuda', 'mps', 'auto', or torch.device
            dtype: Default tensor data type for operations
            timing_enabled: Whether to track and report operation timing

        Raises:
            ValueError: If device type is not supported
        """
        # Handle device configuration
        if isinstance(device, str):
            device = DeviceType(device)
        
        if device == DeviceType.AUTO:
            self.device = get_optimal_device()
        elif isinstance(device, DeviceType):
            self.device = torch.device(device.value)
        else:
            self.device = device
            
        self.dtype = dtype
        self.timing_enabled = timing_enabled
        self._operation_times = []

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the configured device.

        Args:
            tensor: Input tensor

        Returns:
            Tensor moved to the configured device
        """
        return tensor.to(device=self.device, dtype=self.dtype)

    def _validate_dimensions(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """Validate that matrices can be multiplied.

        Args:
            A: First matrix
            B: Second matrix

        Raises:
            ValueError: If matrices cannot be multiplied due to dimension mismatch
        """
        if A.shape[-1] != B.shape[-2]:
            raise ValueError(
                f"Cannot multiply matrices with shapes {A.shape} and {B.shape}. "
                f"Inner dimensions must match: {A.shape[-1]} != {B.shape[-2]}"
            )

    def _time_operation(self, operation_name: str, duration: float) -> None:
        """Record timing information for an operation.

        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
        """
        if self.timing_enabled:
            self._operation_times.append({
                'operation': operation_name,
                'duration': duration,
                'device': str(self.device)
            })

    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics for recorded operations.

        Returns:
            Dictionary with timing statistics including total, average, and per-operation times
        """
        if not self._operation_times:
            return {'message': 'No timing data available. Enable timing_enabled=True'}
        
        total_time = sum(op['duration'] for op in self._operation_times)
        avg_time = total_time / len(self._operation_times)
        
        return {
            'total_operations': len(self._operation_times),
            'total_time': total_time,
            'average_time': avg_time,
            'device': str(self.device),
            'operations': self._operation_times.copy()
        }

    def clear_timing_stats(self) -> None:
        """Clear all recorded timing statistics."""
        self._operation_times.clear()


# Sample healthcare data for demonstrations
HEALTHCARE_PATIENT_DATA = [
    "Patient presents with acute chest pain and elevated troponin levels.",
    "Blood pressure 140/90 mmHg, heart rate 95 bpm, temperature 98.6°F.",
    "Laboratory results show elevated white blood cell count and CRP.",
    "Patient reports chronic fatigue and joint pain for several months.",
    "Imaging studies reveal no acute abnormalities in chest X-ray.",
    "Medication adherence appears suboptimal based on patient interview.",
    "Vital signs stable, patient responding well to current treatment.",
    "Follow-up appointment scheduled for medication adjustment review."
]


class MatrixMultiplicationCalculator(BaseMatrixOperations):
    """Comprehensive matrix multiplication calculator for LLM development.

    This class provides a unified interface for all matrix multiplication
    operations commonly used in large language model development, with
    special attention to performance optimization and healthcare applications.

    The calculator supports various multiplication methods, batch operations,
    GPU acceleration, and performance monitoring for production use.

    Example:
        Basic matrix multiplication with performance monitoring:

        ```python
        calc = MatrixMultiplicationCalculator(device='cuda', timing_enabled=True)

        # Basic multiplication
        A = torch.randn(1000, 2000)
        B = torch.randn(2000, 1500)
        C = calc.multiply(A, B)

        # Batch operations
        batch_A = torch.randn(32, 512, 768)  # Batch of 32 matrices
        batch_B = torch.randn(32, 768, 1024)
        batch_C = calc.batch_multiply(batch_A, batch_B)

        # Performance analysis
        stats = calc.get_timing_stats()
        print(f"Average operation time: {stats['average_time']:.4f}s")
        ```

    Attributes:
        device: Computation device (CPU, CUDA, MPS)
        dtype: Default tensor data type
        timing_enabled: Whether to track operation timing
    """

    def multiply(self, A: torch.Tensor, B: torch.Tensor,
                method: Union[str, MatrixMultiplicationMethod] = MatrixMultiplicationMethod.STANDARD) -> torch.Tensor:
        """Perform matrix multiplication using specified method.

        Args:
            A: First matrix with shape [..., m, k]
            B: Second matrix with shape [..., k, n]
            method: Multiplication method to use

        Returns:
            Result matrix with shape [..., m, n]

        Raises:
            ValueError: If matrices cannot be multiplied or method is unsupported

        Example:
            ```python
            calc = MatrixMultiplicationCalculator()
            A = torch.randn(100, 200)
            B = torch.randn(200, 150)

            # Standard multiplication (recommended)
            C = calc.multiply(A, B)

            # Using specific method
            C = calc.multiply(A, B, method='torch.matmul')
            ```
        """
        if isinstance(method, str):
            method = MatrixMultiplicationMethod(method)

        # Move tensors to device
        A = self._to_device(A)
        B = self._to_device(B)

        # Validate dimensions
        self._validate_dimensions(A, B)

        start_time = time.time() if self.timing_enabled else None

        # Perform multiplication based on method
        if method == MatrixMultiplicationMethod.STANDARD:
            result = A @ B
        elif method == MatrixMultiplicationMethod.TORCH_MM:
            if len(A.shape) != 2 or len(B.shape) != 2:
                raise ValueError("torch.mm requires 2D matrices")
            result = torch.mm(A, B)
        elif method == MatrixMultiplicationMethod.TORCH_MATMUL:
            result = torch.matmul(A, B)
        elif method == MatrixMultiplicationMethod.TORCH_BMM:
            if len(A.shape) != 3 or len(B.shape) != 3:
                raise ValueError("torch.bmm requires 3D tensors")
            result = torch.bmm(A, B)
        elif method == MatrixMultiplicationMethod.NUMPY_DOT:
            # Convert to numpy, compute, convert back
            A_np = A.detach().cpu().numpy()
            B_np = B.detach().cpu().numpy()
            result_np = np.dot(A_np, B_np)
            result = torch.from_numpy(result_np).to(device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported multiplication method: {method}")

        if self.timing_enabled and start_time is not None:
            # Synchronize for accurate GPU timing
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()

            duration = time.time() - start_time
            self._time_operation(f"multiply_{method.value}", duration)

        return result

    def batch_multiply(self, batch_A: torch.Tensor, batch_B: torch.Tensor,
                      broadcast: bool = True) -> torch.Tensor:
        """Perform batch matrix multiplication.

        Args:
            batch_A: Batch of matrices with shape [batch_size, m, k]
            batch_B: Batch of matrices with shape [batch_size, k, n] or [k, n] for broadcasting
            broadcast: Whether to allow broadcasting for batch_B

        Returns:
            Batch of result matrices with shape [batch_size, m, n]

        Example:
            ```python
            calc = MatrixMultiplicationCalculator()

            # Batch multiplication
            batch_A = torch.randn(32, 512, 768)
            batch_B = torch.randn(32, 768, 1024)
            batch_C = calc.batch_multiply(batch_A, batch_B)

            # Broadcasting example
            single_B = torch.randn(768, 1024)  # Single matrix
            batch_C = calc.batch_multiply(batch_A, single_B, broadcast=True)
            ```
        """
        batch_A = self._to_device(batch_A)
        batch_B = self._to_device(batch_B)

        start_time = time.time() if self.timing_enabled else None

        if broadcast and len(batch_B.shape) == 2:
            # Broadcasting: single matrix multiplied with batch
            result = torch.matmul(batch_A, batch_B)
        elif len(batch_A.shape) == 3 and len(batch_B.shape) == 3:
            # Standard batch multiplication
            if batch_A.shape[0] != batch_B.shape[0]:
                raise ValueError(f"Batch sizes must match: {batch_A.shape[0]} != {batch_B.shape[0]}")
            result = torch.bmm(batch_A, batch_B)
        else:
            # Use general matmul for other cases
            result = torch.matmul(batch_A, batch_B)

        if self.timing_enabled and start_time is not None:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()

            duration = time.time() - start_time
            self._time_operation("batch_multiply", duration)

        return result

    def neural_network_layer(self, inputs: torch.Tensor, weights: torch.Tensor,
                           bias: Optional[torch.Tensor] = None,
                           activation: Optional[str] = None) -> torch.Tensor:
        """Simulate a neural network layer using matrix multiplication.

        This method demonstrates how matrix multiplication is used in neural
        networks, particularly relevant for transformer architectures in LLMs.

        Args:
            inputs: Input tensor with shape [batch_size, input_dim]
            weights: Weight matrix with shape [input_dim, output_dim]
            bias: Optional bias vector with shape [output_dim]
            activation: Optional activation function ('relu', 'gelu', 'tanh', 'sigmoid')

        Returns:
            Layer output with shape [batch_size, output_dim]

        Example:
            ```python
            calc = MatrixMultiplicationCalculator()

            # Simulate transformer feed-forward layer
            inputs = torch.randn(32, 768)  # Batch of 32, hidden dim 768
            weights = torch.randn(768, 3072)  # Expand to 3072 (4x hidden)
            bias = torch.randn(3072)

            output = calc.neural_network_layer(inputs, weights, bias, activation='gelu')
            ```
        """
        inputs = self._to_device(inputs)
        weights = self._to_device(weights)
        if bias is not None:
            bias = self._to_device(bias)

        start_time = time.time() if self.timing_enabled else None

        # Linear transformation: inputs @ weights + bias
        output = self.multiply(inputs, weights)

        if bias is not None:
            output = output + bias

        # Apply activation function
        if activation == 'relu':
            output = F.relu(output)
        elif activation == 'gelu':
            output = F.gelu(output)
        elif activation == 'tanh':
            output = torch.tanh(output)
        elif activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif activation is not None:
            raise ValueError(f"Unsupported activation function: {activation}")

        if self.timing_enabled and start_time is not None:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()

            duration = time.time() - start_time
            self._time_operation(f"neural_layer_{activation or 'linear'}", duration)

        return output

    def attention_mechanism(self, query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                          scale: bool = True) -> torch.Tensor:
        """Compute scaled dot-product attention using matrix multiplication.

        This method demonstrates the core attention mechanism used in transformers,
        which relies heavily on efficient matrix multiplication.

        Args:
            query: Query tensor with shape [batch_size, seq_len, d_model]
            key: Key tensor with shape [batch_size, seq_len, d_model]
            value: Value tensor with shape [batch_size, seq_len, d_model]
            mask: Optional attention mask with shape [batch_size, seq_len, seq_len]
            scale: Whether to scale by sqrt(d_model)

        Returns:
            Attention output with shape [batch_size, seq_len, d_model]

        Example:
            ```python
            calc = MatrixMultiplicationCalculator()

            # Transformer attention
            batch_size, seq_len, d_model = 32, 128, 768
            query = torch.randn(batch_size, seq_len, d_model)
            key = torch.randn(batch_size, seq_len, d_model)
            value = torch.randn(batch_size, seq_len, d_model)

            attention_output = calc.attention_mechanism(query, key, value)
            ```
        """
        query = self._to_device(query)
        key = self._to_device(key)
        value = self._to_device(value)
        if mask is not None:
            mask = self._to_device(mask)

        start_time = time.time() if self.timing_enabled else None

        # Compute attention scores: Q @ K^T
        scores = self.batch_multiply(query, key.transpose(-2, -1))

        # Scale by sqrt(d_model) if requested
        if scale:
            d_model = query.size(-1)
            scores = scores / (d_model ** 0.5)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values: Attention @ V
        output = self.batch_multiply(attention_weights, value)

        if self.timing_enabled and start_time is not None:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()

            duration = time.time() - start_time
            self._time_operation("attention_mechanism", duration)

        return output

    def benchmark_methods(self, matrix_sizes: List[Tuple[int, int, int]],
                         methods: Optional[List[MatrixMultiplicationMethod]] = None,
                         num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark different matrix multiplication methods.

        Args:
            matrix_sizes: List of (m, k, n) tuples representing matrix dimensions
            methods: List of methods to benchmark (default: all applicable methods)
            num_runs: Number of runs for each benchmark

        Returns:
            Dictionary with benchmark results including timing and performance metrics

        Example:
            ```python
            calc = MatrixMultiplicationCalculator(device='cuda', timing_enabled=True)

            sizes = [(100, 200, 150), (500, 1000, 750), (1000, 2000, 1500)]
            results = calc.benchmark_methods(sizes)

            for size, timings in results['results'].items():
                print(f"Size {size}: {timings}")
            ```
        """
        if methods is None:
            methods = [
                MatrixMultiplicationMethod.STANDARD,
                MatrixMultiplicationMethod.TORCH_MATMUL,
                MatrixMultiplicationMethod.TORCH_MM
            ]

        results = {
            'device': str(self.device),
            'num_runs': num_runs,
            'results': {}
        }

        for m, k, n in matrix_sizes:
            size_key = f"{m}x{k}x{n}"
            results['results'][size_key] = {}

            # Generate test matrices
            A = torch.randn(m, k, device=self.device, dtype=self.dtype)
            B = torch.randn(k, n, device=self.device, dtype=self.dtype)

            for method in methods:
                method_times = []

                # Skip incompatible methods
                if method == MatrixMultiplicationMethod.TORCH_BMM and len(A.shape) != 3:
                    continue

                try:
                    for _ in range(num_runs):
                        start_time = time.time()

                        # Perform multiplication
                        _ = self.multiply(A, B, method=method)

                        # Synchronize for accurate timing
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        elif self.device.type == 'mps':
                            torch.mps.synchronize()

                        duration = time.time() - start_time
                        method_times.append(duration)

                    # Calculate statistics
                    avg_time = sum(method_times) / len(method_times)
                    min_time = min(method_times)
                    max_time = max(method_times)

                    results['results'][size_key][method.value] = {
                        'average_time': avg_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'all_times': method_times
                    }

                except Exception as e:
                    results['results'][size_key][method.value] = {
                        'error': str(e)
                    }

        return results


class HealthcareMatrixDemo:
    """Demonstration class for healthcare applications of matrix multiplication.

    This class provides comprehensive examples of how matrix multiplication
    is used in healthcare AI and medical language model development.

    Example:
        Run healthcare demonstrations:

        ```python
        demo = HealthcareMatrixDemo()
        demo.run_all_demonstrations()
        ```
    """

    def __init__(self, device: Union[str, DeviceType] = DeviceType.AUTO):
        """Initialize the healthcare demonstration.

        Args:
            device: Computation device for demonstrations
        """
        self.calc = MatrixMultiplicationCalculator(device=device, timing_enabled=True)
        self.device = device

    def demonstrate_neural_network_layers(self) -> None:
        """Demonstrate matrix multiplication in neural network layers.

        Shows how matrix multiplication powers feed-forward layers in
        medical language models and diagnostic systems.
        """
        print("=" * 70)
        print("NEURAL NETWORK LAYERS - HEALTHCARE AI")
        print("=" * 70)

        # Simulate medical text embeddings
        batch_size = 16  # 16 patient records
        embedding_dim = 768  # BERT-like embeddings
        hidden_dim = 3072  # 4x expansion in feed-forward

        print(f"Simulating medical language model processing:")
        print(f"• Batch size: {batch_size} patient records")
        print(f"• Embedding dimension: {embedding_dim}")
        print(f"• Hidden dimension: {hidden_dim}")

        # Generate sample embeddings (representing medical text)
        medical_embeddings = torch.randn(batch_size, embedding_dim)

        # Feed-forward layer weights
        ff_weights_1 = torch.randn(embedding_dim, hidden_dim)
        ff_bias_1 = torch.randn(hidden_dim)
        ff_weights_2 = torch.randn(hidden_dim, embedding_dim)
        ff_bias_2 = torch.randn(embedding_dim)

        print(f"\nProcessing medical text through feed-forward layers...")

        # First layer: expand
        hidden = self.calc.neural_network_layer(
            medical_embeddings, ff_weights_1, ff_bias_1, activation='gelu'
        )

        # Second layer: project back
        output = self.calc.neural_network_layer(
            hidden, ff_weights_2, ff_bias_2
        )

        print(f"✅ Input shape: {medical_embeddings.shape}")
        print(f"✅ Hidden shape: {hidden.shape}")
        print(f"✅ Output shape: {output.shape}")

        # Show timing statistics
        stats = self.calc.get_timing_stats()
        print(f"\nPerformance Metrics:")
        print(f"• Total operations: {stats['total_operations']}")
        print(f"• Total time: {stats['total_time']:.4f}s")
        print(f"• Average time per operation: {stats['average_time']:.4f}s")
        print(f"• Device: {stats['device']}")

        self.calc.clear_timing_stats()

    def demonstrate_attention_mechanism(self) -> None:
        """Demonstrate attention mechanism for medical text analysis.

        Shows how attention mechanisms use matrix multiplication to focus
        on relevant parts of medical documents and patient records.
        """
        print("\n" + "=" * 70)
        print("ATTENTION MECHANISM - MEDICAL TEXT ANALYSIS")
        print("=" * 70)

        # Simulate medical document processing
        batch_size = 8  # 8 medical documents
        seq_length = 512  # Document length (tokens)
        d_model = 768  # Model dimension

        print(f"Simulating medical document attention:")
        print(f"• Batch size: {batch_size} medical documents")
        print(f"• Sequence length: {seq_length} tokens")
        print(f"• Model dimension: {d_model}")

        # Generate sample medical text representations
        query = torch.randn(batch_size, seq_length, d_model)
        key = torch.randn(batch_size, seq_length, d_model)
        value = torch.randn(batch_size, seq_length, d_model)

        # Create attention mask (simulate padding)
        mask = torch.ones(batch_size, seq_length, seq_length)
        # Mask out last 50 tokens for some sequences (padding)
        mask[:, :, -50:] = 0

        print(f"\nComputing attention for medical text analysis...")

        # Compute attention
        attention_output = self.calc.attention_mechanism(
            query, key, value, mask=mask, scale=True
        )

        print(f"✅ Query shape: {query.shape}")
        print(f"✅ Key shape: {key.shape}")
        print(f"✅ Value shape: {value.shape}")
        print(f"✅ Attention output shape: {attention_output.shape}")

        # Show timing for attention computation
        stats = self.calc.get_timing_stats()
        if stats['total_operations'] > 0:
            latest_op = stats['operations'][-1]
            print(f"\nAttention Performance:")
            print(f"• Operation: {latest_op['operation']}")
            print(f"• Duration: {latest_op['duration']:.4f}s")
            print(f"• Device: {latest_op['device']}")

        self.calc.clear_timing_stats()

    def demonstrate_batch_processing(self) -> None:
        """Demonstrate batch processing for healthcare data.

        Shows how batch matrix multiplication enables efficient processing
        of multiple patient records simultaneously.
        """
        print("\n" + "=" * 70)
        print("BATCH PROCESSING - PATIENT DATA ANALYSIS")
        print("=" * 70)

        # Simulate patient feature matrices
        num_patients = 100
        num_features = 50  # Lab values, vitals, demographics
        num_diagnoses = 20  # Possible diagnoses

        print(f"Simulating diagnostic prediction system:")
        print(f"• Number of patients: {num_patients}")
        print(f"• Features per patient: {num_features}")
        print(f"• Possible diagnoses: {num_diagnoses}")

        # Patient feature matrix
        patient_features = torch.randn(num_patients, num_features)

        # Diagnostic weight matrix (learned from training data)
        diagnostic_weights = torch.randn(num_features, num_diagnoses)
        diagnostic_bias = torch.randn(num_diagnoses)

        print(f"\nProcessing all patients simultaneously...")

        # Batch prediction for all patients
        diagnostic_scores = self.calc.neural_network_layer(
            patient_features, diagnostic_weights, diagnostic_bias, activation='sigmoid'
        )

        print(f"✅ Patient features shape: {patient_features.shape}")
        print(f"✅ Diagnostic weights shape: {diagnostic_weights.shape}")
        print(f"✅ Diagnostic scores shape: {diagnostic_scores.shape}")

        # Show top predictions for first patient
        first_patient_scores = diagnostic_scores[0]
        top_diagnoses = torch.topk(first_patient_scores, k=3)

        print(f"\nSample Results (Patient 1):")
        print(f"• Top 3 diagnostic scores: {top_diagnoses.values.tolist()}")
        print(f"• Corresponding diagnosis indices: {top_diagnoses.indices.tolist()}")

        # Performance comparison: batch vs individual
        print(f"\nBatch Processing Benefits:")
        print(f"• Processed {num_patients} patients in single operation")
        print(f"• Leverages vectorized computation and GPU parallelism")
        print(f"• Much faster than processing patients individually")

    def run_all_demonstrations(self) -> None:
        """Run all healthcare matrix multiplication demonstrations.

        Provides a comprehensive overview of matrix multiplication applications
        in healthcare AI and medical language modeling.
        """
        print("HEALTHCARE MATRIX MULTIPLICATION DEMONSTRATIONS")
        print("=" * 70)
        print("Exploring matrix multiplication in medical AI applications")
        print("=" * 70)

        self.demonstrate_neural_network_layers()
        self.demonstrate_attention_mechanism()
        self.demonstrate_batch_processing()

        print("\n" + "=" * 70)
        print("SUMMARY - MATRIX MULTIPLICATION IN HEALTHCARE AI")
        print("=" * 70)
        print("Key applications demonstrated:")
        print("• Neural network layers for medical text processing")
        print("• Attention mechanisms for document analysis")
        print("• Batch processing for efficient patient data analysis")
        print("• GPU acceleration for real-time clinical decision support")
        print("• Scalable architectures for large-scale medical AI systems")
        print("=" * 70)


def demonstrate_performance_comparison() -> None:
    """Demonstrate performance comparison between different methods and devices."""
    print("\n🚀 PERFORMANCE COMPARISON DEMONSTRATION")
    print("=" * 60)

    # Test different matrix sizes
    test_sizes = [(100, 200, 150), (500, 1000, 750), (1000, 2000, 1500)]

    for device_type in [DeviceType.CPU, DeviceType.CUDA if torch.cuda.is_available() else None]:
        if device_type is None:
            continue

        print(f"\n📊 Testing on {device_type.value.upper()}:")
        print("-" * 40)

        calc = MatrixMultiplicationCalculator(device=device_type, timing_enabled=True)

        # Benchmark different methods
        results = calc.benchmark_methods(test_sizes, num_runs=3)

        for size, method_results in results['results'].items():
            print(f"\nMatrix size {size}:")
            for method, timing in method_results.items():
                if 'error' not in timing:
                    avg_time = timing['average_time']
                    print(f"  {method:<15}: {avg_time:.4f}s")
                else:
                    print(f"  {method:<15}: {timing['error']}")


def demonstrate_unified_calculator() -> None:
    """Demonstrate the unified MatrixMultiplicationCalculator."""
    print("UNIFIED MATRIX MULTIPLICATION CALCULATOR DEMONSTRATION")
    print("=" * 60)

    # Initialize calculator with optimal device
    calc = MatrixMultiplicationCalculator(device=DeviceType.AUTO, timing_enabled=True)
    print(f"Using device: {calc.device}")

    # Basic matrix multiplication
    print(f"\n🔢 Basic Matrix Multiplication:")
    A = torch.randn(100, 200)
    B = torch.randn(200, 150)
    C = calc.multiply(A, B)
    print(f"✅ {A.shape} @ {B.shape} = {C.shape}")

    # Batch operations
    print(f"\n📦 Batch Operations:")
    batch_A = torch.randn(32, 64, 128)
    batch_B = torch.randn(32, 128, 256)
    batch_C = calc.batch_multiply(batch_A, batch_B)
    print(f"✅ Batch: {batch_A.shape} @ {batch_B.shape} = {batch_C.shape}")

    # Neural network layer
    print(f"\n🧠 Neural Network Layer:")
    inputs = torch.randn(16, 512)
    weights = torch.randn(512, 1024)
    bias = torch.randn(1024)
    output = calc.neural_network_layer(inputs, weights, bias, activation='gelu')
    print(f"✅ Layer: {inputs.shape} -> {output.shape} (with GELU activation)")

    # Performance summary
    stats = calc.get_timing_stats()
    print(f"\n📈 Performance Summary:")
    print(f"Total operations: {stats['total_operations']}")
    print(f"Total time: {stats['total_time']:.4f}s")
    print(f"Average time: {stats['average_time']:.4f}s")


if __name__ == "__main__":
    """Main execution block for demonstrations."""

    print("🔢 MATRIX MULTIPLICATION FOR LARGE LANGUAGE MODELS")
    print("=" * 70)
    print("Comprehensive demonstrations of matrix multiplication concepts")
    print("with focus on healthcare applications and LLM development")
    print("=" * 70)

    # Run unified calculator demonstration
    print("\n🔧 UNIFIED CALCULATOR DEMONSTRATION")
    demonstrate_unified_calculator()

    # Run performance comparison
    demonstrate_performance_comparison()

    # Run healthcare-specific demonstrations
    print("\n🏥 HEALTHCARE APPLICATIONS")
    healthcare_demo = HealthcareMatrixDemo(device=DeviceType.AUTO)
    healthcare_demo.run_all_demonstrations()

    print("\n✅ ALL DEMONSTRATIONS COMPLETED")
    print("=" * 70)
    print("For production use, import the classes directly:")
    print("from matrix_multiplication import MatrixMultiplicationCalculator")
    print("calc = MatrixMultiplicationCalculator(device='cuda')")
    print("=" * 70)
