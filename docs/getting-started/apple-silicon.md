# Apple Silicon Optimization

This guide covers specific optimizations for Apple Silicon processors (M1, M2, M3) to maximize performance when working with Large Language Models and deep learning workloads.

## Apple Silicon M3 Ultra Advantages

Your M3 Ultra processor provides significant advantages for LLM development:

- **🚀 Up to 20x faster training** compared to CPU-only
- **🧠 128GB unified memory** for large model loading
- **⚡ Neural Engine acceleration** for optimized operations
- **🔋 Energy efficient** training and inference
- **🎯 Native ARM64 optimizations** for all libraries

## PyTorch MPS Backend

### Configuration

Metal Performance Shaders (MPS) provides GPU acceleration on Apple Silicon:

```python
import torch
from accelerate import Accelerator

# Automatic device selection (recommended)
accelerator = Accelerator()
device = accelerator.device  # Will use 'mps' on Apple Silicon

# Manual device selection
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Verify MPS availability
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")
```

### Memory Optimization

Configure memory settings for optimal performance:

```python
import os

# Set environment variables before importing torch
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

# Enable memory efficient attention
torch.backends.mps.enable_memory_efficient_attention = True

# Memory cleanup function
def cleanup_mps_memory():
    """Clean up MPS memory cache."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc
    gc.collect()
```

## Hugging Face Transformers Optimization

### Model Loading

Optimize model loading for Apple Silicon:

```python
from transformers import AutoModel, AutoTokenizer
import torch

def load_model_optimized(model_name: str):
    """Load model optimized for Apple Silicon."""
    
    # Load with MPS device mapping
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto",          # Automatic device mapping
        low_cpu_mem_usage=True,     # Reduce CPU memory usage
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move to MPS if available
    if torch.backends.mps.is_available():
        model = model.to('mps')
    
    return model, tokenizer

# Example usage
model, tokenizer = load_model_optimized("microsoft/DialoGPT-medium")
```

### Training Optimization

Configure training for optimal Apple Silicon performance:

```python
from transformers import TrainingArguments, Trainer
import torch

training_args = TrainingArguments(
    output_dir="./results",
    
    # Apple Silicon optimizations
    use_mps_device=True,                    # Use MPS backend
    dataloader_pin_memory=False,            # Disable pin memory on Apple Silicon
    dataloader_num_workers=0,               # Single worker for MPS
    
    # Memory optimizations
    per_device_train_batch_size=4,          # Adjust based on model size
    gradient_accumulation_steps=4,          # Simulate larger batch size
    fp16=True,                              # Use half precision
    gradient_checkpointing=True,            # Save memory
    
    # Performance optimizations
    remove_unused_columns=False,            # Keep all columns for debugging
    prediction_loss_only=True,              # Reduce memory usage
    
    # Learning rate and scheduling
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
)
```

## Accelerate Library Integration

### Configuration

Set up Accelerate for optimal Apple Silicon performance:

```python
from accelerate import Accelerator, DistributedDataParallelKwargs

# Configure for Apple Silicon
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision='fp16',  # Use half precision
    kwargs_handlers=[ddp_kwargs]
)

# Prepare model and optimizer
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
```

### Training Loop

Optimized training loop for Apple Silicon:

```python
def train_model_apple_silicon(model, train_dataloader, optimizer, accelerator):
    """Optimized training loop for Apple Silicon."""
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            
            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
            
            # Memory cleanup every 100 steps
            if batch_idx % 100 == 0:
                cleanup_mps_memory()
                
            # Logging
            if batch_idx % 10 == 0:
                accelerator.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
```

## Performance Benchmarking

### System Monitoring

Monitor system performance during training:

```python
import psutil
import time
import torch

class AppleSiliconMonitor:
    """Monitor Apple Silicon performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used
    
    def log_metrics(self):
        """Log current performance metrics."""
        current_memory = psutil.virtual_memory().used
        memory_usage = (current_memory - self.initial_memory) / 1024**3  # GB
        
        print(f"Runtime: {time.time() - self.start_time:.2f}s")
        print(f"Memory Usage: {memory_usage:.2f} GB")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        
        if torch.backends.mps.is_available():
            print("MPS Backend: Active")
        
        return {
            'runtime': time.time() - self.start_time,
            'memory_gb': memory_usage,
            'cpu_percent': psutil.cpu_percent()
        }

# Usage
monitor = AppleSiliconMonitor()
# ... training code ...
metrics = monitor.log_metrics()
```

### Benchmark Script

Create a benchmark to test your setup:

```python
import torch
import time
from transformers import AutoModel, AutoTokenizer

def benchmark_apple_silicon():
    """Benchmark Apple Silicon performance."""
    
    print("🍎 Apple Silicon LLM Benchmark")
    print("=" * 40)
    
    # Test MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {'✅' if mps_available else '❌'}")
    
    if not mps_available:
        print("❌ MPS not available. Using CPU.")
        device = 'cpu'
    else:
        device = 'mps'
    
    # Load a small model for testing
    model_name = "distilbert-base-uncased"
    print(f"Loading model: {model_name}")
    
    start_time = time.time()
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    load_time = time.time() - start_time
    
    print(f"Model load time: {load_time:.2f}s")
    
    # Test inference
    text = "Apple Silicon provides excellent performance for machine learning workloads."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(**inputs)
    
    # Benchmark inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs = model(**inputs)
    inference_time = time.time() - start_time
    
    print(f"100 inference runs: {inference_time:.2f}s")
    print(f"Average per inference: {inference_time/100*1000:.2f}ms")
    
    # Memory usage
    if device == 'mps':
        print(f"MPS Memory Usage: Available")
    
    print("✅ Benchmark complete!")

if __name__ == "__main__":
    benchmark_apple_silicon()
```

## Best Practices

### 1. Memory Management

```python
# Use context managers for memory cleanup
class MPSMemoryManager:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()

# Usage
with MPSMemoryManager():
    # Your training/inference code here
    outputs = model(inputs)
```

### 2. Batch Size Optimization

```python
def find_optimal_batch_size(model, tokenizer, sample_text):
    """Find optimal batch size for your hardware."""
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    for batch_size in [1, 2, 4, 8, 16, 32]:
        try:
            # Create batch
            inputs = tokenizer([sample_text] * batch_size, 
                             return_tensors="pt", 
                             padding=True, 
                             truncation=True).to(device)
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"✅ Batch size {batch_size}: Success")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Batch size {batch_size}: Out of memory")
                break
            else:
                raise e
        
        # Cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

### 3. Model Size Recommendations

| Model Size | Recommended Batch Size | Memory Usage |
|------------|----------------------|--------------|
| Small (< 100M params) | 16-32 | < 4GB |
| Medium (100M-1B params) | 4-8 | 4-8GB |
| Large (1B-7B params) | 1-2 | 8-16GB |
| Very Large (7B+ params) | 1 | 16GB+ |

## Troubleshooting

### Common Issues

#### 1. MPS Not Available

```bash
# Check macOS version (requires 12.3+)
sw_vers

# Update PyTorch
uv pip install --upgrade torch torchvision torchaudio
```

#### 2. Out of Memory Errors

```python
# Reduce batch size
training_args.per_device_train_batch_size = 1

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use gradient accumulation
training_args.gradient_accumulation_steps = 8
```

#### 3. Slow Performance

```python
# Ensure MPS is being used
print(f"Model device: {next(model.parameters()).device}")

# Check for CPU fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

## Performance Tips

1. **Use half precision** (`fp16=True`) when possible
2. **Enable gradient checkpointing** for large models
3. **Use gradient accumulation** instead of large batch sizes
4. **Clean up memory** regularly during training
5. **Monitor system resources** to avoid bottlenecks
6. **Use the latest PyTorch** for best MPS support

## Next Steps

- **[Study Guide](../study-guide/index.md)** - Begin your learning journey
- **[Mathematical Foundations](../materials/math/index.md)** - Mathematical foundations
- **[Code Examples](../code-examples/index.md)** - Hands-on implementations
