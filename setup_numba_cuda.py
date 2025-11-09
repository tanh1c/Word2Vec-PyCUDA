#!/usr/bin/env python3
# Copyright 2024 Word2Vec Implementation
# Official solution for CUDA PTX compatibility in Google Colab
# Based on: https://github.com/googlecolab/colabtools/issues/5081

import subprocess
import sys

def setup_numba_cuda():
    """Setup numba-cuda with official solution from GitHub issue #5081."""
    print("🔧 Setting up numba-cuda (Official Solution)")
    print("Based on: https://github.com/googlecolab/colabtools/issues/5081")
    print()
    
    # 2. Configure numba-cuda
    print("2. Configuring numba-cuda...")
    try:
        from numba import config
        config.CUDA_ENABLE_PYNVJITLINK = 1
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        print("✓ numba-cuda configuration set")
        print("  - CUDA_ENABLE_PYNVJITLINK = 1")
        print("  - CUDA_LOW_OCCUPANCY_WARNINGS = 0")
    except Exception as e:
        print(f"❌ Failed to configure numba-cuda: {e}")
        return False
    
    # 3. Test CUDA functionality
    print("3. Testing CUDA functionality...")
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

def main():
    """Main function."""
    print("Word2Vec CUDA Setup - Official Solution")
    print("=" * 50)
    
    if setup_numba_cuda():
        print("\n🎉 numba-cuda setup completed successfully!")
        print("\nYou can now run:")
        print("  python run_all.py")
    else:
        print("\n❌ Failed to setup numba-cuda")
        print("\nTry running manually:")
        print("  !uv pip install -q --system numba-cuda==0.4.0")
        print("  from numba import config")
        print("  config.CUDA_ENABLE_PYNVJITLINK = 1")

if __name__ == "__main__":
    main()
