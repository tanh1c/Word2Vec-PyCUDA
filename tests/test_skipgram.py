# Copyright 2024 Word2Vec Implementation
# Unit tests for Skip-gram model

import os
import sys
import math
import tempfile
from typing import List

import numpy as np
import pytest

# Enable CUDA simulator for testing
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

from numba import cuda
from numba.cuda import random as c_random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w2v_skipgram import calc_skipgram, step_skipgram


def close_enough(x: float, target: float, tolerance: float = 1e-6) -> bool:
    """Check if two floats are close enough."""
    return abs(x - target) <= tolerance


def close_enough_array(x: np.ndarray, target: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if two arrays are close enough."""
    return np.allclose(x, target, atol=tolerance)


def test_step_skipgram():
    """Test Skip-gram step function with known values."""
    # Create test embeddings
    emb = np.array([
        [0, 0],      # BLANK_TOKEN
        [-0.1, 0.1], # word 1
        [0.1, -0.1], # word 2
        [0.2, 0.4],  # word 3
        [-0.3, -0.2] # word 4
    ], dtype=np.float32)
    
    emb2 = np.array([
        [0, 0],      # BLANK_TOKEN
        [0.2, -0.5], # word 1
        [-0.2, 0.3], # word 2
        [0.1, 0.7],  # word 3
        [-0.4, 0.1]  # word 4
    ], dtype=np.float32)
    
    emb_dim = emb.shape[1]
    k = 2
    thread_idx = 0
    vocab_size = emb.shape[0]
    neg_smpl_arr = [1, 1, 1, 1, 2, 2]  # Negative sampling array
    x = 3  # Center word index
    y = 4  # Context word index
    lr = 0.2
    
    # Initialize CUDA arrays
    calc_aux = cuda.to_device(np.zeros((vocab_size, emb_dim), dtype=np.float32))
    random_states = c_random.create_xoroshiro128p_states(vocab_size, seed=12345)
    neg_smpl_arr_cuda = cuda.to_device(neg_smpl_arr)
    
    w1 = cuda.to_device(emb.copy())
    w2 = cuda.to_device(emb2.copy())
    
    print(f"Initial w1[x]: {w1[x]}")
    print(f"Initial w2[y]: {w2[y]}")
    
    # Call step function
    step_skipgram(thread_idx, w1, w2, calc_aux, x, y, k, lr, neg_smpl_arr_cuda, random_states)
    
    # Copy results back
    w1_result = w1.copy_to_host()
    w2_result = w2.copy_to_host()
    calc_aux_result = calc_aux.copy_to_host()
    
    print(f"Final w1[x]: {w1_result[x]}")
    print(f"Final w2[y]: {w2_result[y]}")
    print(f"calc_aux[thread_idx]: {calc_aux_result[thread_idx]}")
    
    # Verify that weights have been updated
    assert not np.allclose(w1_result[x], emb[x])
    assert not np.allclose(w2_result[y], emb2[y])
    
    # Verify that calc_aux has been updated
    assert not np.allclose(calc_aux_result[thread_idx], 0.0)
    
    # Verify that dot product has increased (positive sample should be strengthened)
    dot_original = np.dot(emb[x], emb2[y])
    dot_final = np.dot(w1_result[x], w2_result[y])
    print(f"Dot product: {dot_original} -> {dot_final}")
    assert dot_final > dot_original


def test_calc_skipgram():
    """Test Skip-gram calculation kernel."""
    # Create test data
    sentence_count = 2
    c = 2  # window size
    k = 1  # negative samples
    learning_rate = 0.1
    embed_dim = 2
    vocab_size = 5
    
    # Create test embeddings
    w1 = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    w2 = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    w1[0, :] = 0.0  # BLANK_TOKEN
    w2[0, :] = 0.0  # BLANK_TOKEN
    
    # Test data: 2 sentences
    # Sentence 1: [1, 2, 3] -> contexts for 1: [2], for 2: [1,3], for 3: [2]
    # Sentence 2: [2, 4, 1] -> contexts for 2: [4], for 4: [2,1], for 1: [4]
    inp = np.array([1, 2, 3, 2, 4, 1], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int32)
    lengths = np.array([3, 3], dtype=np.int32)
    
    # Other arrays
    calc_aux = np.zeros((sentence_count, embed_dim), dtype=np.float32)
    subsample_weights = np.array([0.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    negsample_array = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    
    # Initialize CUDA arrays
    w1_cuda = cuda.to_device(w1)
    w2_cuda = cuda.to_device(w2)
    calc_aux_cuda = cuda.to_device(calc_aux)
    random_states = c_random.create_xoroshiro128p_states(sentence_count, seed=12345)
    subsample_weights_cuda = cuda.to_device(subsample_weights)
    negsample_array_cuda = cuda.to_device(negsample_array)
    inp_cuda = cuda.to_device(inp)
    offsets_cuda = cuda.to_device(offsets)
    lengths_cuda = cuda.to_device(lengths)
    
    # Store original values
    w1_original = w1_cuda.copy_to_host()
    w2_original = w2_cuda.copy_to_host()
    
    # Launch kernel
    blocks = 1
    threads_per_block = 2
    calc_skipgram[blocks, threads_per_block](
        sentence_count, c, k, learning_rate, w1_cuda, w2_cuda, calc_aux_cuda,
        random_states, subsample_weights_cuda, negsample_array_cuda,
        inp_cuda, offsets_cuda, lengths_cuda
    )
    
    # Get results
    w1_result = w1_cuda.copy_to_host()
    w2_result = w2_cuda.copy_to_host()
    calc_aux_result = calc_aux_cuda.copy_to_host()
    
    # Verify that weights have been updated
    assert not np.allclose(w1_result, w1_original)
    assert not np.allclose(w2_result, w2_original)
    
    # Verify that calc_aux has been used
    assert not np.allclose(calc_aux_result, 0.0)
    
    print("Skip-gram kernel test passed!")


def test_skipgram_gradient_calculation():
    """Test Skip-gram gradient calculation with known values."""
    # Simple test case
    emb = np.array([
        [0, 0],      # BLANK_TOKEN
        [0.1, 0.2],  # word 1
        [0.3, 0.4]   # word 2
    ], dtype=np.float32)
    
    emb2 = np.array([
        [0, 0],      # BLANK_TOKEN
        [0.5, 0.6],  # word 1
        [0.7, 0.8]   # word 2
    ], dtype=np.float32)
    
    # Manual calculation
    x, y = 1, 2  # Center word 1, context word 2
    dot_xy = np.dot(emb[x], emb2[y])
    sigmoid_xy = 1.0 / (1.0 + math.exp(-dot_xy))
    s_xdy_m1 = sigmoid_xy - 1
    
    # Expected gradients
    expected_grad_x = -s_xdy_m1 * emb2[y]
    expected_grad_y = -s_xdy_m1 * emb[x]
    
    print(f"Manual calculation:")
    print(f"  dot_xy: {dot_xy}")
    print(f"  sigmoid_xy: {sigmoid_xy}")
    print(f"  s_xdy_m1: {s_xdy_m1}")
    print(f"  expected_grad_x: {expected_grad_x}")
    print(f"  expected_grad_y: {expected_grad_y}")
    
    # Test with CUDA
    emb_dim = emb.shape[1]
    vocab_size = emb.shape[0]
    k = 0  # No negative sampling for this test
    lr = 1.0  # Use lr=1 for easy verification
    thread_idx = 0
    
    calc_aux = cuda.to_device(np.zeros((vocab_size, emb_dim), dtype=np.float32))
    random_states = c_random.create_xoroshiro128p_states(vocab_size, seed=12345)
    neg_smpl_arr = np.array([1, 1, 2, 2], dtype=np.int32)
    neg_smpl_arr_cuda = cuda.to_device(neg_smpl_arr)
    
    w1 = cuda.to_device(emb.copy())
    w2 = cuda.to_device(emb2.copy())
    
    # Call step function
    step_skipgram(thread_idx, w1, w2, calc_aux, x, y, k, lr, neg_smpl_arr_cuda, random_states)
    
    # Get results
    w1_result = w1.copy_to_host()
    w2_result = w2.copy_to_host()
    calc_aux_result = calc_aux.copy_to_host()
    
    # Calculate actual gradients
    actual_grad_x = w1_result[x] - emb[x]
    actual_grad_y = w2_result[y] - emb2[y]
    
    print(f"CUDA results:")
    print(f"  actual_grad_x: {actual_grad_x}")
    print(f"  actual_grad_y: {actual_grad_y}")
    
    # Verify gradients (with some tolerance for floating point)
    assert close_enough_array(actual_grad_x, expected_grad_x, 1e-5)
    assert close_enough_array(actual_grad_y, expected_grad_y, 1e-5)
    
    print("Skip-gram gradient calculation test passed!")


if __name__ == "__main__":
    pytest.main([__file__])
