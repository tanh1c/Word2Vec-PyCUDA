#!/usr/bin/env python3
# Copyright 2024 Word2Vec Implementation
# Complete Google Colab setup script - gộp tất cả setup vào 1 file
# Thay thế cho các lệnh:
#   (User phải tự chạy trước: !uv pip install -q --system numba-cuda==0.4.0)
#   !python setup_numba_cuda.py
#   !python colab_setup.py

import os
import subprocess
import sys

def install_package_with_uv(package: str, quiet: bool = True) -> bool:
    """
    Install package using uv pip (tương đương với: !uv pip install -q --system package)
    
    Args:
        package: Package name with version (e.g., "numba-cuda==0.4.0")
        quiet: If True, suppress output (equivalent to -q flag)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["uv", "pip", "install"]
        if quiet:
            cmd.append("-q")
        cmd.extend(["--system", package])
        
        result = subprocess.run(
            cmd,
            capture_output=quiet,
            text=True,
            check=True
        )
        
        if not quiet:
            print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        if not quiet and e.stdout:
            print(f"  stdout: {e.stdout}")
        if not quiet and e.stderr:
            print(f"  stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        # Fallback to regular pip if uv is not available
        print(f"⚠️  uv not found, trying regular pip for {package}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if quiet:
                cmd.append("-q")
            cmd.append(package)
            
            subprocess.check_call(cmd)
            if not quiet:
                print(f"✓ {package} installed successfully (via pip)")
            return True
        except Exception as e2:
            print(f"❌ Failed to install {package} with pip: {e2}")
            return False

def check_numba_cuda_installed():
    """
    Check if numba-cuda is already installed (không install nữa).
    User phải tự install: !uv pip install -q --system numba-cuda==0.4.0
    """
    print("\n" + "=" * 60)
    print("STEP 1: Checking numba-cuda installation")
    print("=" * 60)
    print("Checking if numba-cuda is installed...")
    
    try:
        import numba
        from numba import cuda
        print("✓ numba-cuda is already installed")
        
        # Check version if possible
        try:
            import numba_cuda
            print(f"  numba version: {numba.__version__ if hasattr(numba, '__version__') else 'unknown'}")
        except:
            pass
        
        return True
    except ImportError:
        print("❌ numba-cuda is NOT installed")
        print("⚠️  Please install manually first:")
        print("   !uv pip install -q --system numba-cuda==0.4.0")
        return False

def setup_numba_cuda_config():
    """
    Setup numba-cuda configuration (from setup_numba_cuda.py).
    Tương đương với: !python setup_numba_cuda.py
    """
    print("\n" + "=" * 60)
    print("STEP 2: Configuring numba-cuda")
    print("=" * 60)
    print("🔧 Setting up numba-cuda (Official Solution)")
    print("Based on: https://github.com/googlecolab/colabtools/issues/5081")
    print()
    
    # Configure numba-cuda
    print("Configuring numba-cuda...")
    try:
        from numba import config
        config.CUDA_ENABLE_PYNVJITLINK = 1
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        print("✓ numba-cuda configuration set")
        print("  - CUDA_ENABLE_PYNVJITLINK = 1")
        print("  - CUDA_LOW_OCCUPANCY_WARNINGS = 0")
    except ImportError:
        print("❌ numba not installed - cannot configure")
        return False
    except Exception as e:
        print(f"❌ Failed to configure numba-cuda: {e}")
        return False
    
    # Test CUDA functionality
    print("\nTesting CUDA functionality...")
    try:
        from numba import cuda
        import numpy as np
        
        if cuda.is_available():
            device = cuda.get_current_device()
            print(f"✓ CUDA available: {device.name}")
            
            # Test simple kernel
            @cuda.jit
            def increment_by_one(an_array):
                pos = cuda.grid(1)
                if pos < an_array.size:
                    an_array[pos] += 1
            
            test_array = np.zeros(10, dtype=np.float32)
            increment_by_one[16, 16](test_array)
            
            expected = np.ones(10, dtype=np.float32)
            if np.allclose(test_array, expected):
                print("✓ CUDA kernel test passed!")
                return True
            else:
                print("❌ CUDA kernel test failed")
                return False
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def install_all_requirements():
    """
    Install all required packages (from colab_setup.py).
    Tương đương với một phần của: !python colab_setup.py
    """
    print("\n" + "=" * 60)
    print("STEP 3: Installing all required packages")
    print("=" * 60)
    print("Installing required packages for Google Colab...")
    
    packages = [
        "numpy>=1.20.0", 
        "gensim>=4.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "pynvml>=11.0.0"
    ]
    
    success_count = 0
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...", end=" ", flush=True)
        if install_package_with_uv(package, quiet=True):
            print("✓")
            success_count += 1
        else:
            print("❌")
            failed_packages.append(package)
    
    print(f"\nInstalled {success_count}/{len(packages)} packages successfully")
    
    if failed_packages:
        print(f"⚠️  Failed packages: {', '.join(failed_packages)}")
        return False
    
    return True

def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 60)
    print("Checking GPU availability...")
    print("=" * 60)
    
    try:
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
    print("\n" + "=" * 60)
    print("Checking CUDA availability...")
    print("=" * 60)
    
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
    """Main setup function - gộp tất cả setup."""
    print("=" * 60)
    print("  Word2Vec Implementation - Complete Google Colab Setup")
    print("=" * 60)
    print("\nThis script combines all setup steps:")
    print("  1. Check numba-cuda installation (must be installed separately)")
    print("  2. Configure numba-cuda")
    print("  3. Install all required packages")
    print("  4. Check GPU and CUDA availability")
    print()
    print("⚠️  NOTE: Please install numba-cuda manually first:")
    print("   !uv pip install -q --system numba-cuda==0.4.0")
    print()
    
    results = {
        "numba_cuda_installed": False,
        "numba_cuda_configured": False,
        "requirements_installed": False,
        "gpu_available": False,
        "cuda_available": False
    }
    
    # Step 1: Check numba-cuda installation (không install nữa)
    results["numba_cuda_installed"] = check_numba_cuda_installed()
    
    if not results["numba_cuda_installed"]:
        print("\n⚠️  Warning: numba-cuda is not installed. Please install it first:")
        print("   !uv pip install -q --system numba-cuda==0.4.0")
        print("   Continuing with other setup steps...")
    
    # Step 2: Setup numba-cuda configuration
    results["numba_cuda_configured"] = setup_numba_cuda_config()
    
    if not results["numba_cuda_configured"]:
        print("\n⚠️  Warning: Failed to configure numba-cuda. Continuing anyway...")
    
    # Step 3: Install all requirements
    results["requirements_installed"] = install_all_requirements()
    
    # Step 4: Check GPU
    results["gpu_available"] = check_gpu()
    
    # Step 5: Check CUDA
    results["cuda_available"] = check_cuda()
    
    # Summary
    print("\n" + "=" * 60)
    print("  SETUP SUMMARY")
    print("=" * 60)
    print(f"  ✓ numba-cuda installed: {'✓' if results['numba_cuda_installed'] else '❌'}")
    print(f"  ✓ numba-cuda configured: {'✓' if results['numba_cuda_configured'] else '❌'}")
    print(f"  ✓ Requirements installed: {'✓' if results['requirements_installed'] else '❌'}")
    print(f"  ✓ GPU available: {'✓' if results['gpu_available'] else '❌'}")
    print(f"  ✓ CUDA available: {'✓' if results['cuda_available'] else '❌'}")
    print("=" * 60)
    
    # Final message
    if results['gpu_available'] and results['cuda_available']:
        print("\n🎉 Setup complete! Ready to run Word2Vec training.")
        print("\nTo run the full pipeline:")
        print("  !python run_all.py")
    elif results['numba_cuda_installed'] and results['numba_cuda_configured']:
        print("\n✅ Setup completed successfully!")
        print("⚠️  Note: GPU/CUDA may not be available, but CPU training is still possible.")
        print("\nTo run the full pipeline:")
        print("  !python run_all.py")
    else:
        print("\n⚠️  Setup completed with some warnings.")
        print("Some features may not work correctly.")
        print("\nTo run anyway:")
        print("  !python run_all.py")
    
    return 0

if __name__ == "__main__":
    # Trong notebook (Colab/Jupyter), không nên dùng sys.exit()
    # vì nó sẽ gây SystemExit exception và warning
    # Chỉ gọi main() trực tiếp
    main()
    
    # Note: Nếu chạy từ command line, có thể dùng sys.exit(main())
    # nhưng trong notebook thì không cần

