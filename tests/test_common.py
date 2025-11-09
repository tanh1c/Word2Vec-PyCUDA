# Copyright 2024 Word2Vec Implementation
# Unit tests for common utilities

import os
import sys
import math
import tempfile
import shutil
from typing import List, Tuple

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w2v_common import (
    build_vocab, sort_vocab, prune_vocab, bias_freq_counts,
    get_subsampling_weights_and_negative_sampling_array,
    init_weight_matrices, BLANK_TOKEN
)


def create_test_data(temp_dir: str) -> str:
    """Create test data files."""
    data_dir = os.path.join(temp_dir, "test_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create test files with sample text
    test_texts = [
        "the quick brown fox jumps over the lazy dog",
        "the cat in the hat sat on the mat",
        "brown fox jumps over lazy dog again",
        "the quick brown fox is very fast",
        "lazy dog sleeps all day long"
    ]
    
    for i, text in enumerate(test_texts):
        with open(os.path.join(data_dir, f"000{i}"), "w") as f:
            f.write(text + "\n")
    
    return data_dir


def test_build_vocab():
    """Test vocabulary building."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = create_test_data(temp_dir)
        vocab = build_vocab(data_dir)
        
        # Check that we get expected words
        words = [word for word, _, _ in vocab]
        assert "the" in words
        assert "fox" in words
        assert "dog" in words
        
        # Check counts
        the_entry = next((entry for entry in vocab if entry[0] == "the"), None)
        assert the_entry is not None
        assert the_entry[1] >= 5  # "the" appears multiple times
        assert the_entry[2] >= 4  # "the" appears in multiple sentences


def test_sort_vocab():
    """Test vocabulary sorting."""
    orig = [("foo", 1, 2), ("bar", 55, 2), ("asdf", 3, 3), ("another_word", 4, 1)]
    result = sort_vocab(orig)
    
    # Check BLANK_TOKEN is first
    assert result[0][0] == BLANK_TOKEN
    assert result[0][1] == 0
    assert result[0][2] == 0
    
    # Check sorting by frequency (descending)
    assert result[1][0] == "bar"  # highest frequency
    assert result[2][0] == "another_word"  # second highest
    assert result[3][0] == "asdf"  # third
    assert result[4][0] == "foo"  # lowest


def test_prune_vocab():
    """Test vocabulary pruning."""
    orig = [("foo", 1, 2), ("bar", 55, 2), ("asdf", 3, 3), ("another_word", 4, 1)]
    
    # Test with min_occurs = 1 (no pruning)
    result1 = prune_vocab(min_occrs=1, my_vocab=orig)
    expected1 = [(word, total) for word, total, _ in orig]
    assert result1 == expected1
    
    # Test with min_occurs = 2 (prune words with < 2 sentence occurrences)
    result2 = prune_vocab(min_occrs=2, my_vocab=orig)
    expected2 = [("foo", 1), ("bar", 55), ("asdf", 3)]  # "another_word" has only 1 sentence
    assert result2 == expected2


def test_bias_freq_counts():
    """Test frequency biasing."""
    words = [BLANK_TOKEN, "pie", "apple", "mince", "beef", "delicious"]
    freqs = [0, 3, 2, 2, 1, 1]
    vocab = list(zip(words, freqs))
    
    # Test with exponent = 1.0 (no biasing)
    result1 = bias_freq_counts(vocab, exponent=1.0)
    total = sum(freqs)
    expected1 = [(word, count/total) for word, count in vocab]
    
    assert len(result1) == len(expected1)
    for i, (word, freq) in enumerate(result1):
        expected_word, expected_freq = expected1[i]
        assert word == expected_word
        assert abs(freq - expected_freq) < 1e-6
    
    # Test with exponent = 0.75
    result2 = bias_freq_counts(vocab, exponent=0.75)
    assert len(result2) == len(vocab)
    
    # Check that frequencies sum to 1
    total_freq = sum(freq for _, freq in result2)
    assert abs(total_freq - 1.0) < 1e-6


def test_subsampling_weights():
    """Test subsampling weight calculation."""
    vocab = [('pie', 3), ('apple', 2), ('mince', 2), ('beef', 1), ('delicious', 1)]
    t = 1e-5
    
    weights, neg_array = get_subsampling_weights_and_negative_sampling_array(vocab, t=t)
    
    # Check weights are between 0 and 1
    assert all(0 <= w <= 1 for w in weights)
    
    # Check negative sampling array
    assert len(neg_array) == 1000000
    assert all(0 <= idx < len(vocab) for idx in neg_array)
    
    # Check that more frequent words appear more in negative array
    word_counts = np.bincount(neg_array)
    assert word_counts[1] > word_counts[4]  # 'pie' (idx 1) should appear more than 'beef' (idx 4)


def test_weight_init():
    """Test weight matrix initialization."""
    vocab_size = 1000
    embed_dim = 100
    seed = 12345
    
    w1, w2 = init_weight_matrices(vocab_size, embed_dim, seed=seed)
    
    # Check shapes
    assert w1.shape == (vocab_size, embed_dim)
    assert w2.shape == (vocab_size, embed_dim)
    
    # Check data types
    assert w1.dtype == np.float32
    assert w2.dtype == np.float32
    
    # Check first row is zeros (BLANK_TOKEN)
    assert np.allclose(w1[0, :], 0.0)
    assert np.allclose(w2[0, :], 0.0)
    
    # Check other rows are not all zeros
    assert not np.allclose(w1[1, :], 0.0)
    assert not np.allclose(w2[1, :], 0.0)
    
    # Check distribution (should be approximately normal)
    w1_nonzero = w1[1:, :].flatten()
    w2_nonzero = w2[1:, :].flatten()
    
    # Mean should be close to 0
    assert abs(np.mean(w1_nonzero)) < 0.1
    assert abs(np.mean(w2_nonzero)) < 0.1
    
    # Variance should be close to 1/embed_dim
    expected_var = 1.0 / embed_dim
    assert abs(np.var(w1_nonzero) - expected_var) < 0.1
    assert abs(np.var(w2_nonzero) - expected_var) < 0.1


def test_blank_token_consistency():
    """Test that BLANK_TOKEN is handled consistently."""
    assert BLANK_TOKEN == "<BLANK>"
    
    # Test that BLANK_TOKEN appears in sorted vocab
    vocab = [("word1", 10, 5), ("word2", 5, 3)]
    sorted_vocab = sort_vocab(vocab)
    assert sorted_vocab[0][0] == BLANK_TOKEN
    
    # Test that BLANK_TOKEN is preserved in pruning
    pruned = prune_vocab(min_occrs=10, my_vocab=sorted_vocab)
    assert pruned[0][0] == BLANK_TOKEN


if __name__ == "__main__":
    pytest.main([__file__])
