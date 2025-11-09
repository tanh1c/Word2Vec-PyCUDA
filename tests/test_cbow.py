# Copyright 2024 Word2Vec Implementation
# Unit tests for CBOW model

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

from w2v_cbow import calc_cbow, step_cbow


def close_enough(x: float, target: float, tolerance: float = 1e-6) -> bool:
    """Check if two floats are close enough."""
    return abs(x - target) <= tolerance


def close_enough_array(x: np.ndarray, target: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if two arrays are close enough."""
    return np.allclose(x, target, atol=tolerance)


def test_step_cbow():
    """Test CBOW step function with known values."""
    # Create test embeddings
    emb = np.array([
        [0, 0],      # BLANK_TOKEN
        [-0.1, 0.1], # word 1
        [0.1, -0.1], # word 2
        [0.2, 0.4],  # word 3
        [-0.3, -0.2]  # word 4
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
    context_words = np.array([1, 2], dtype=np.int32)  # Context words: word 1, word 2
    context_count = 2
    center_word = 3  # Center word index
    lr = 0.2
    
    # Initialize CUDA arrays
    calc_aux = cuda.to_device(np.zeros((vocab_size, emb_dim), dtype=np.float32))
    random_states = c_random.create_xoroshiro128p_states(vocab_size, seed=12345)
    neg_smpl_arr_cuda = cuda.to_device(neg_smpl_arr)
    
    w1 = cuda.to_device(emb.copy())
    w2 = cuda.to_device(emb2.copy())
    
    print(f"Initial w1[context_words]: {w1[context_words[0]]}, {w1[context_words[1]]}")
    print(f"Initial w2[center_word]: {w2[center_word]}")
    
    # Call step function
    step_cbow(thread_idx, w1, w2, calc_aux, context_words, context_count, 
              center_word, k, lr, neg_smpl_arr_cuda, random_states)
    
    # Copy results back
    w1_result = w1.copy_to_host()
    w2_result = w2.copy_to_host()
    calc_aux_result = calc_aux.copy_to_host()
    
    print(f"Final w1[context_words]: {w1_result[context_words[0]]}, {w1_result[context_words[1]]}")
    print(f"Final w2[center_word]: {w2_result[center_word]}")
    print(f"calc_aux[thread_idx]: {calc_aux_result[thread_idx]}")
    
    # Verify that weights have been updated
    assert not np.allclose(w1_result[context_words[0]], emb[context_words[0]])
    assert not np.allclose(w1_result[context_words[1]], emb[context_words[1]])
    assert not np.allclose(w2_result[center_word], emb2[center_word])
    
    # Verify that calc_aux has been updated
    assert not np.allclose(calc_aux_result[thread_idx], 0.0)
    
    # Verify that dot product has increased (positive sample should be strengthened)
    # Calculate averaged context vector
    avg_context = (emb[context_words[0]] + emb[context_words[1]]) / 2
    avg_context_final = (w1_result[context_words[0]] + w1_result[context_words[1]]) / 2
    
    dot_original = np.dot(avg_context, emb2[center_word])
    dot_final = np.dot(avg_context_final, w2_result[center_word])
    print(f"Dot product: {dot_original} -> {dot_final}")
    assert dot_final > dot_original


def test_calc_cbow():
    """Test CBOW calculation kernel."""
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
    calc_cbow[blocks, threads_per_block](
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
    
    print("CBOW kernel test passed!")


def test_cbow_gradient_calculation():
    """Test CBOW gradient calculation with known values."""
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
    
    # Manual calculation for CBOW
    context_words = np.array([1, 2], dtype=np.int32)  # Context: word 1, word 2
    center_word = 1  # Center word: word 1
    
    # Average context vectors
    avg_context = (emb[1] + emb[2]) / 2
    dot_product = np.dot(avg_context, emb2[center_word])
    sigmoid = 1.0 / (1.0 + math.exp(-dot_product))
    s_xdy_m1 = sigmoid - 1
    
    # Expected gradients
    expected_grad_context = -s_xdy_m1 * emb2[center_word]
    expected_grad_center = -s_xdy_m1 * avg_context
    
    print(f"Manual calculation:")
    print(f"  avg_context: {avg_context}")
    print(f"  dot_product: {dot_product}")
    print(f"  sigmoid: {sigmoid}")
    print(f"  s_xdy_m1: {s_xdy_m1}")
    print(f"  expected_grad_context: {expected_grad_context}")
    print(f"  expected_grad_center: {expected_grad_center}")
    
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
    step_cbow(thread_idx, w1, w2, calc_aux, context_words, 2, 
              center_word, k, lr, neg_smpl_arr_cuda, random_states)
    
    # Get results
    w1_result = w1.copy_to_host()
    w2_result = w2.copy_to_host()
    calc_aux_result = calc_aux.copy_to_host()
    
    # Calculate actual gradients
    actual_grad_context1 = w1_result[context_words[0]] - emb[context_words[0]]
    actual_grad_context2 = w1_result[context_words[1]] - emb[context_words[1]]
    actual_grad_center = w2_result[center_word] - emb2[center_word]
    
    print(f"CUDA results:")
    print(f"  actual_grad_context1: {actual_grad_context1}")
    print(f"  actual_grad_context2: {actual_grad_context2}")
    print(f"  actual_grad_center: {actual_grad_center}")
    
    # Verify gradients (with some tolerance for floating point)
    # Note: In CBOW, both context words get the same gradient (averaged)
    assert close_enough_array(actual_grad_context1, expected_grad_context, 1e-5)
    assert close_enough_array(actual_grad_context2, expected_grad_context, 1e-5)
    assert close_enough_array(actual_grad_center, expected_grad_center, 1e-5)
    
    print("CBOW gradient calculation test passed!")


if __name__ == "__main__":
    pytest.main([__file__])
