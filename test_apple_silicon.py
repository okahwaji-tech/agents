#!/usr/bin/env python3
"""
Test script to verify Apple Silicon M3 Ultra optimization setup.
This script tests PyTorch MPS support, Hugging Face Transformers, and other key libraries.
"""

import sys
from typing import Dict, Any


def test_pytorch_mps() -> Dict[str, Any]:
    """Test PyTorch MPS (Metal Performance Shaders) support."""
    try:
        import torch
        
        results = {
            "pytorch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        }
        
        if results["mps_available"]:
            # Test tensor creation on MPS
            device = torch.device("mps")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)  # Matrix multiplication on GPU
            results["mps_test_passed"] = z.device.type == "mps"
            results["device"] = str(device)
        else:
            results["mps_test_passed"] = False
            results["device"] = "cpu"
            
        return results
    except Exception as e:
        return {"error": str(e)}


def test_transformers() -> Dict[str, Any]:
    """Test Hugging Face Transformers library."""
    try:
        import transformers
        from transformers import AutoTokenizer
        
        results = {
            "transformers_version": transformers.__version__,
        }
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        test_text = "Hello, Apple Silicon M3 Ultra!"
        tokens = tokenizer.encode(test_text)
        results["tokenizer_test_passed"] = len(tokens) > 0
        results["test_tokens"] = len(tokens)
        
        return results
    except Exception as e:
        return {"error": str(e)}


def test_accelerate() -> Dict[str, Any]:
    """Test Hugging Face Accelerate library."""
    try:
        import accelerate
        
        results = {
            "accelerate_version": accelerate.__version__,
        }
        
        # Test device detection
        from accelerate import Accelerator
        accelerator = Accelerator()
        results["accelerate_device"] = str(accelerator.device)
        results["accelerate_test_passed"] = True
        
        return results
    except Exception as e:
        return {"error": str(e)}


def test_data_science_libs() -> Dict[str, Any]:
    """Test data science libraries."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn as sns
        
        results = {
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "matplotlib_version": matplotlib.__version__,
            "seaborn_version": sns.__version__,
        }
        
        # Test basic operations
        arr = np.random.randn(1000, 1000)
        df = pd.DataFrame(arr[:100, :10])
        results["numpy_test_passed"] = arr.shape == (1000, 1000)
        results["pandas_test_passed"] = df.shape == (100, 10)
        
        return results
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run all tests and display results."""
    print("🚀 Testing Apple Silicon M3 Ultra Setup")
    print("=" * 50)
    
    # Test PyTorch MPS
    print("\n📱 Testing PyTorch MPS Support...")
    pytorch_results = test_pytorch_mps()
    if "error" not in pytorch_results:
        print(f"✅ PyTorch Version: {pytorch_results['pytorch_version']}")
        print(f"✅ MPS Available: {pytorch_results['mps_available']}")
        print(f"✅ MPS Built: {pytorch_results['mps_built']}")
        print(f"✅ Device: {pytorch_results['device']}")
        if pytorch_results['mps_test_passed']:
            print("🎉 MPS GPU acceleration is working!")
        else:
            print("⚠️  MPS test failed, using CPU")
    else:
        print(f"❌ PyTorch test failed: {pytorch_results['error']}")
    
    # Test Transformers
    print("\n🤖 Testing Hugging Face Transformers...")
    transformers_results = test_transformers()
    if "error" not in transformers_results:
        print(f"✅ Transformers Version: {transformers_results['transformers_version']}")
        print(f"✅ Tokenizer Test: {transformers_results['tokenizer_test_passed']}")
        print(f"✅ Test Tokens: {transformers_results['test_tokens']}")
    else:
        print(f"❌ Transformers test failed: {transformers_results['error']}")
    
    # Test Accelerate
    print("\n⚡ Testing Hugging Face Accelerate...")
    accelerate_results = test_accelerate()
    if "error" not in accelerate_results:
        print(f"✅ Accelerate Version: {accelerate_results['accelerate_version']}")
        print(f"✅ Accelerate Device: {accelerate_results['accelerate_device']}")
    else:
        print(f"❌ Accelerate test failed: {accelerate_results['error']}")
    
    # Test Data Science Libraries
    print("\n📊 Testing Data Science Libraries...")
    ds_results = test_data_science_libs()
    if "error" not in ds_results:
        print(f"✅ NumPy Version: {ds_results['numpy_version']}")
        print(f"✅ Pandas Version: {ds_results['pandas_version']}")
        print(f"✅ Matplotlib Version: {ds_results['matplotlib_version']}")
        print(f"✅ Seaborn Version: {ds_results['seaborn_version']}")
    else:
        print(f"❌ Data science libraries test failed: {ds_results['error']}")
    
    print("\n" + "=" * 50)
    print("🎯 Setup verification complete!")
    
    # Summary
    all_passed = (
        "error" not in pytorch_results and
        "error" not in transformers_results and
        "error" not in accelerate_results and
        "error" not in ds_results
    )
    
    if all_passed:
        print("🎉 All tests passed! Your Apple Silicon M3 Ultra setup is ready for LLM development!")
        if pytorch_results.get('mps_available', False):
            print("💪 GPU acceleration is available for maximum performance!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
