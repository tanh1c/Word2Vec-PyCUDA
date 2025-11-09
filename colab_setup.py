#!/usr/bin/env python3
# Copyright 2024 Word2Vec Implementation
# Google Colab setup script

import os
import subprocess
import sys

def install_requirements():
    """Install required packages for Google Colab."""
    print("Installing required packages for Google Colab...")
    
    packages = [
        "numba-cuda==0.4.0",
        "numpy>=1.20.0", 
        "gensim>=4.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "pynvml>=11.0.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected:")
            print(result.stdout)
            return True
        else:
            print("❌ No NVIDIA GPU detected")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def check_cuda():
    """Check CUDA availability with Numba."""
    print("\nChecking CUDA availability...")
    
    try:
        from numba import cuda
        if cuda.is_available():
            device = cuda.get_current_device()
            print(f"✓ CUDA available: {device.name}")
            
            # Try to get memory info using pynvml if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory = memory_info.total / 1024**3
                print(f"  Memory: {total_memory:.1f} GB")
            except (ImportError, Exception) as e:
                # Fallback: just show device name without memory info
                print(f"  Device: {device.name}")
                print(f"  (Memory info unavailable: {e})")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ Numba not installed")
        return False

def main():
    """Main setup function."""
    print("Word2Vec Implementation - Google Colab Setup")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return 1
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check CUDA
    cuda_available = check_cuda()
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print(f"  GPU Available: {'✓' if gpu_available else '❌'}")
    print(f"  CUDA Available: {'✓' if cuda_available else '❌'}")
    
    if gpu_available and cuda_available:
        print("\n🎉 Setup complete! Ready to run Word2Vec training.")
        print("\nTo run the full pipeline:")
        print("  !python run_all.py")
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("Training will be slower without GPU acceleration.")
        print("\nTo run anyway:")
        print("  !python run_all.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
