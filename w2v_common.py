# Copyright 2024 Word2Vec Implementation
# Based on myw2v by Taneli Saastamoinen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
import math
import os
import pathlib
import re
import time
import hashlib
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

from numba import cuda
import numpy as np
from numpy import linalg, ndarray


W2V_VERSION = "1.0"
BLANK_TOKEN = "<BLANK>"

# Constants for Hierarchical Softmax and Exp Table
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
MAX_CODE_LENGTH = 40


def build_vocab(data_path: str) -> List[Tuple[str, int, int]]:
    """
    Build vocabulary from data files.
    Returns list of (word, total_count, sentence_count).
    """
    files = [fn for fn in os.listdir(data_path) if fn.startswith("0")]
    sentences_per_word = defaultdict(int)
    totals_per_word = defaultdict(int)
    
    for file in files:
        with open(os.path.join(data_path, file), encoding="utf-8") as f:
            for line in f:
                less_spacey = re.sub(r"[ ]{2,}", " ", line.strip())
                words = less_spacey.split(" ")
                if len(words) > 1:
                    uniques = set()
                    for word in words:
                        uniques.add(word)
                        totals_per_word[word] += 1
                    for deduped in uniques:
                        sentences_per_word[deduped] += 1
    
    r = []
    for word, total in totals_per_word.items():
        sent = sentences_per_word[word]
        r.append((word, total, sent))
    return r


def sort_vocab(my_vocab: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """Sort vocabulary by frequency (descending), then alphabetically."""
    vs = [(BLANK_TOKEN, 0, 0)] + sorted(my_vocab, key=lambda t: (-t[1], t[0]))
    return vs


def prune_vocab(min_occrs: int, my_vocab: List[Tuple[str, int, int]]) -> List[Tuple[str, int]]:
    """
    Prune vocabulary based on minimum sentence occurrences.
    Returns only total counts.
    """
    if min_occrs > 1:
        totals = [(wrd, total_count) for wrd, total_count, sentence_count in my_vocab 
                 if sentence_count >= min_occrs or wrd == BLANK_TOKEN]
        return totals
    else:
        return [(word, total) for word, total, _ in my_vocab]


def bias_freq_counts(vocab: List[Tuple[str, int]], exponent: float) -> List[Tuple[str, float]]:
    """Apply frequency biasing with given exponent for negative sampling."""
    totalsson = sum(count for _, count in vocab)
    plain = [(word, count / totalsson) for word, count in vocab]
    
    if exponent == 1.0:
        return plain
    
    exped = [(word, math.pow(count, exponent)) for word, count in plain]
    sum_exped = sum([q for _, q in exped])
    jooh = [(word, f/sum_exped) for word, f in exped]
    return jooh


def _get_vocab_cache_key(data_path: str, min_occurs_by_sentence: int, freq_exponent: float) -> str:
    """Generate cache key based on vocabulary parameters."""
    # Create hash from parameters that affect vocabulary
    key_string = f"{data_path}_{min_occurs_by_sentence}_{freq_exponent}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _get_vocab_cache_path(cache_key: str) -> str:
    """Get path to vocabulary cache file."""
    cache_dir = "./output/vocab_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"vocab_{cache_key}.pkl")


def _save_vocab_cache(vocab: List[Tuple[str, float]], w_to_i: Dict[str, int], 
                     word_counts: List[int], cache_path: str):
    """Save vocabulary to cache file."""
    cache_data = {
        'vocab': vocab,
        'w_to_i': w_to_i,
        'word_counts': word_counts
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)


def _load_vocab_cache(cache_path: str) -> Optional[Tuple[List[Tuple[str, float]], Dict[str, int], List[int]]]:
    """Load vocabulary from cache file. Returns None if cache doesn't exist or is invalid."""
    try:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return (cache_data['vocab'], cache_data['w_to_i'], cache_data['word_counts'])
    except Exception:
        return None


def handle_vocab(data_path: str, min_occurs_by_sentence: int, freq_exponent: float, 
                 use_cache: bool = True):
    """
    Complete vocabulary handling pipeline with optional caching.
    Returns: (biased_vocab, w_to_i, word_counts)
    - biased_vocab: List of (word, frequency) for negative sampling
    - w_to_i: Dictionary mapping word to index
    - word_counts: List of word counts (for Huffman tree construction)
    
    Args:
        use_cache: If True, try to load from cache or save to cache after building.
                   Cache is based on data_path, min_occurs_by_sentence, and freq_exponent.
                   Changing epochs or embed_dim will NOT invalidate the cache.
    """
    # Try to load from cache
    if use_cache:
        cache_key = _get_vocab_cache_key(data_path, min_occurs_by_sentence, freq_exponent)
        cache_path = _get_vocab_cache_path(cache_key)
        cached_vocab = _load_vocab_cache(cache_path)
        if cached_vocab is not None:
            return cached_vocab
    
    # Build vocabulary
    vocab: List[Tuple[str, int, int]] = build_vocab(data_path)
    sorted_vocab: List[Tuple[str, int, int]] = sort_vocab(vocab)
    pruned_vocab: List[Tuple[str, int]] = prune_vocab(min_occurs_by_sentence, sorted_vocab)
    # Store word counts before biasing
    word_counts = [count for _, count in pruned_vocab]
    biased_vocab: List[Tuple[str, float]] = bias_freq_counts(pruned_vocab, freq_exponent)
    w_to_i: Dict[str, int] = {word: idx for idx, (word, _) in enumerate(biased_vocab)}
    
    # Save to cache
    if use_cache:
        cache_key = _get_vocab_cache_key(data_path, min_occurs_by_sentence, freq_exponent)
        cache_path = _get_vocab_cache_path(cache_key)
        _save_vocab_cache(biased_vocab, w_to_i, word_counts, cache_path)
    
    return biased_vocab, w_to_i, word_counts


def get_subsampling_weights_and_negative_sampling_array(vocab: List[Tuple[str, float]], t: float) -> Tuple[ndarray, ndarray]:
    """
    Calculate subsampling weights and create negative sampling array.
    
    Negative sampling array size is dynamically adjusted based on vocabulary size:
    - For small vocabs (< 10k): uses 1M (original default)
    - For medium vocabs (10k-100k): uses 10M
    - For large vocabs (> 100k): uses 100M (same as word2vec.c original)
    
    This ensures all words appear in the array and maintains distribution accuracy.
    """
    # Subsampling weights
    tot_wgt: int = sum([c for _, c in vocab])
    freqs: List[float] = [c/tot_wgt for _, c in vocab]
    # Clamp negative probabilities to zero
    probs: List[float] = [max(0.0, 1-math.sqrt(t/freq)) if freq > 0 else 0.0 for freq in freqs]

    # Negative sampling array - precompute for efficient sampling
    vocab_size = len(vocab)
    
    # Dynamically adjust arr_len based on vocabulary size
    # Word2vec.c original uses 1e8 (100M), we scale based on vocab size
    if vocab_size < 10000:
        arr_len = 1000000  # 1M for small vocabs
    elif vocab_size < 100000:
        arr_len = 10000000  # 10M for medium vocabs
    else:
        arr_len = 100000000  # 100M for large vocabs (same as word2vec.c)
    
    print(f"Creating negative sampling array with size {arr_len:,} for vocab size {vocab_size:,}")
    
    w2 = [round(f*arr_len) for f in freqs]
    
    # Check if any words would be excluded (rounded to 0)
    excluded_count = sum(1 for scaled in w2 if scaled == 0)
    if excluded_count > 0:
        print(f"⚠️  WARNING: {excluded_count} words have frequency too low and will be excluded from negative sampling")
        print(f"   Consider increasing arr_len or reducing min_occurs threshold")
    
    neg_arr = []
    for i, scaled in enumerate(w2):
        if scaled > 0:  # Only add words that appear at least once
            neg_arr.extend([i]*scaled)
    
    actual_arr_size = len(neg_arr)
    print(f"Negative sampling array created: {actual_arr_size:,} entries ({actual_arr_size/1e6:.2f}M)")
    
    return np.asarray(probs, dtype=np.float32), np.asarray(neg_arr, dtype=np.int32)


def get_data_file_names(path: str, seed: int) -> List[str]:
    """Get shuffled list of data file names."""
    rng = np.random.default_rng(seed=seed)
    qq = [fn for fn in os.listdir(path) if fn.startswith("0")]
    # Sort first to ensure consistent shuffling
    data_files = sorted(qq)
    rng.shuffle(data_files)
    return data_files


def read_all_data_files_ever(dat_path: str, file_names: List[str], w_to_i: Dict[str, int], 
                             max_words: int = None) -> Tuple[List[int], List[int], List[int]]:
    """
    Read all data files and convert to indices.
    
    Args:
        dat_path: Path to data directory
        file_names: List of file names to read
        w_to_i: Word to index mapping
        max_words: Maximum number of words to read (None = all). If specified, 
                   will stop reading when total words reach this limit.
    
    Returns:
        Tuple of (inps, offs, lens) where:
        - inps: List of word indices
        - offs: List of offsets for each sentence
        - lens: List of sentence lengths
    """
    start = time.time()
    inps, offs, lens = [], [], []
    offset_total = 0
    stats = defaultdict(int)
    total_words_read = 0
    stopped_early = False
    
    for fn in file_names:
        fp = os.path.join(dat_path, fn)
        ok_lines = 0
        too_short_lines = 0
        with open(fp, encoding="utf-8") as f:
            for line in f:
                # Check if we've reached max_words limit
                if max_words is not None and total_words_read >= max_words:
                    stopped_early = True
                    break
                
                words = [word for word in re.split(r"[ .]+", line.strip()) if word]
                if len(words) < 2:
                    too_short_lines += 1
                    continue
                idcs = [w_to_i[w] for w in words if w in w_to_i]
                le = len(idcs)
                
                # Check if adding this sentence would exceed max_words
                if max_words is not None and total_words_read + le > max_words:
                    # Only add words up to the limit
                    remaining_words = max_words - total_words_read
                    if remaining_words > 0:
                        idcs = idcs[:remaining_words]
                        le = len(idcs)
                    else:
                        stopped_early = True
                        break
                
                ok_lines += 1
                offs.append(offset_total)
                lens.append(le)
                inps.extend(idcs)
                offset_total += le
                total_words_read += le
                
                # Break if we've reached the limit exactly
                if max_words is not None and total_words_read >= max_words:
                    stopped_early = True
                    break
        
        stats["file_read_lines_ok"] += ok_lines
        stats["one_word_sentence_lines_which_were_ignored"] += too_short_lines
        
        # Break outer loop if we've reached the limit
        if stopped_early:
            break

    print(f"read_all_data_files_ever() STATS: {stats}")
    if max_words is not None and stopped_early:
        print(f"  ⚠️  Stopped early: reached max_words limit of {max_words:,} words")
    tot_tm = time.time()-start
    print(f"read_all_data_files_ever() Total time {tot_tm} s for {len(file_names)} files (avg {tot_tm/len(file_names)} s/file)")
    return inps, offs, lens


def init_weight_matrices(vocab_size: int, embed_dim: int, seed: int) -> Tuple[ndarray, ndarray]:
    """Initialize weight matrices with Gaussian distribution."""
    rng = np.random.default_rng(seed=seed)
    rows, cols = vocab_size, embed_dim
    sigma: float = math.sqrt(1.0/cols)
    zs = rng.standard_normal(size=(rows, cols), dtype=np.float32)
    xs = sigma * zs
    # First row all zero since it represents the blank token
    xs[0, :] = 0.0
    zs2 = rng.standard_normal(size=(rows, cols), dtype=np.float32)
    xs2 = sigma * zs2
    xs2[0, :] = 0.0
    return xs, xs2


def print_norms(weights_cuda):
    """Print statistics about vector norms."""
    w = weights_cuda.copy_to_host()
    norms = [linalg.norm(v) for v in w]
    a, med, b = np.percentile(norms, [2.5, 50, 97.5])
    avg = float(sum(norms) / len(norms))
    print(f"Vector norms (count {len(norms)}) 2.5% median mean 97.5%: {a:0.4f}  {med:0.4f}  {avg:0.4f}  {b:0.4f}")


def write_vectors(weights_cuda, vocab: List[Tuple[str, float]], out_path: str):
    """Write vectors to file in word2vec format."""
    w = weights_cuda.copy_to_host()
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # len-1: skip first which is the blank token & all zero
        f.write(f"{len(w)-1} {len(w[0])}\n")
        for i, v in enumerate(w):
            # skip first which is the blank token & all zero
            if i == 0:
                continue
            v_str = " ".join([str(f) for f in v])
            word, _ = vocab[i]
            f.write(f"{word} {v_str}\n")


def write_json(to_jsonify: Dict[str, Any], json_path: str):
    """Write dictionary to JSON file."""
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonify))
        f.write("\n")
        f.flush()


def create_exp_table(exp_table_size: int = EXP_TABLE_SIZE, max_exp: float = MAX_EXP) -> ndarray:
    """
    Create precomputed exp table for fast sigmoid calculation.
    Based on word2vec.c lines 708-712.
    
    Args:
        exp_table_size: Size of the exp table (default: 1000)
        max_exp: Maximum exponent value (default: 6)
    
    Returns:
        numpy array of precomputed sigmoid values
    """
    exp_table = np.zeros(exp_table_size, dtype=np.float32)
    for i in range(exp_table_size):
        # Precompute exp((i / exp_table_size * 2 - 1) * max_exp)
        exp_value = math.exp((i / exp_table_size * 2 - 1) * max_exp)
        # Precompute sigmoid: exp(x) / (exp(x) + 1)
        exp_table[i] = exp_value / (exp_value + 1)
    return exp_table


def init_hs_weight_matrix(vocab_size: int, embed_dim: int) -> ndarray:
    """
    Initialize Hierarchical Softmax weight matrix (syn1).
    Based on word2vec.c lines 356-359.
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
    
    Returns:
        Weight matrix for internal nodes: (vocab_size - 1, embed_dim)
        Initialized with zeros
    """
    # Internal nodes: vocab_size - 1
    syn1 = np.zeros((vocab_size - 1, embed_dim), dtype=np.float32)
    return syn1


def create_huffman_tree(word_counts: List[int], max_code_length: int = MAX_CODE_LENGTH) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Create binary Huffman tree from word counts.
    Based on word2vec.c lines 205-270.
    
    Frequent words will have short unique binary codes.
    
    Args:
        word_counts: List of word counts (frequencies)
        max_code_length: Maximum code length (default: 40)
    
    Returns:
        Tuple of (codes_array, points_array, code_lengths):
        - codes_array: (vocab_size, max_code_length) binary codes, padded with -1
        - points_array: (vocab_size, max_code_length) node indices in path, padded with -1
        - code_lengths: (vocab_size,) code length for each word
    """
    vocab_size = len(word_counts)
    
    # Initialize arrays
    count = np.zeros(vocab_size * 2 + 1, dtype=np.int64)
    binary = np.zeros(vocab_size * 2 + 1, dtype=np.int32)
    parent_node = np.zeros(vocab_size * 2 + 1, dtype=np.int64)
    
    # Set initial counts
    for a in range(vocab_size):
        count[a] = word_counts[a]
    for a in range(vocab_size, vocab_size * 2):
        count[a] = int(1e15)  # Large value for internal nodes
    
    # Build Huffman tree
    pos1 = vocab_size - 1
    pos2 = vocab_size
    
    for a in range(vocab_size - 1):
        # Find two smallest nodes
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1i = pos1
                pos1 -= 1
            else:
                min1i = pos2
                pos2 += 1
        else:
            min1i = pos2
            pos2 += 1
        
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2i = pos1
                pos1 -= 1
            else:
                min2i = pos2
                pos2 += 1
        else:
            min2i = pos2
            pos2 += 1
        
        count[vocab_size + a] = count[min1i] + count[min2i]
        parent_node[min1i] = vocab_size + a
        parent_node[min2i] = vocab_size + a
        binary[min2i] = 1
    
    # Assign binary codes to each word
    codes_array = np.full((vocab_size, max_code_length), -1, dtype=np.int32)
    points_array = np.full((vocab_size, max_code_length), -1, dtype=np.int32)
    code_lengths = np.zeros(vocab_size, dtype=np.int32)
    
    for a in range(vocab_size):
        b = a
        i = 0
        code = np.zeros(max_code_length, dtype=np.int32)
        point = np.zeros(max_code_length, dtype=np.int64)
        
        # Traverse from leaf to root
        while True:
            code[i] = binary[b]
            point[i] = b
            i += 1
            b = parent_node[b]
            if b == vocab_size * 2 - 2:
                break
            if i >= max_code_length:
                break  # Safety check
        
        code_lengths[a] = i
        # Store code and point arrays (reversed)
        points_array[a, 0] = vocab_size - 2  # Root node
        for b_idx in range(i):
            codes_array[a, i - b_idx - 1] = code[b_idx]
            if b_idx < i - 1:
                points_array[a, i - b_idx] = int(point[b_idx] - vocab_size)
    
    return codes_array, points_array, code_lengths
