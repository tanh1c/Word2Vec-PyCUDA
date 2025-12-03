# Copyright 2024 Word2Vec Implementation
# Skip-gram implementation with Numba CUDA

import math
import os
import time
from typing import List, Tuple, Dict, Any

from numba import cuda
from numba.cuda import random as c_random
import numpy as np
from numpy import ndarray

from w2v_common import (
    handle_vocab, get_subsampling_weights_and_negative_sampling_array,
    get_data_file_names, read_all_data_files_ever, init_weight_matrices,
    print_norms, write_vectors, write_json, W2V_VERSION,
    create_exp_table, init_hs_weight_matrix, create_huffman_tree,
    EXP_TABLE_SIZE, MAX_EXP, MAX_CODE_LENGTH
)


@cuda.jit
def calc_skipgram(
        rows: int,
        c: int,
        k: int,
        learning_rate: float,
        w1,
        w2,
        calc_aux,
        random_states,
        subsample_weights,
        negsample_array,
        inp,
        offsets,
        lengths,
        use_hs,
        syn1,
        codes_array,
        points_array,
        code_lengths,
        exp_table,
        exp_table_size,
        max_exp):
    """
    CUDA kernel for Skip-gram training.
    Based on word2vec.c Skip-gram implementation (lines 495-543).
    Supports both Hierarchical Softmax and Negative Sampling.
    """
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= rows:
        return
    le = lengths[idx]
    off = offsets[idx]
    
    for centre in range(0, le):
        word_idx = inp[off + centre]
        prob_to_reject = subsample_weights[word_idx]
        rnd = c_random.xoroshiro128p_uniform_float32(random_states, idx)
        
        if rnd > prob_to_reject:
            r_f = c_random.xoroshiro128p_uniform_float32(random_states, idx)
            r: int = math.ceil(r_f * c)
            
            # Context before center word
            for context_pre in range(max(0, centre-r), centre):
                step_skipgram(idx, w1, w2, calc_aux, inp[off+centre], inp[off+context_pre], 
                             k, learning_rate, negsample_array, random_states,
                             use_hs, syn1, codes_array, points_array, code_lengths,
                             exp_table, exp_table_size, max_exp)
            
            # Context after center word
            for context_post in range(centre + 1, min(le, centre + 1 + r)):
                step_skipgram(idx, w1, w2, calc_aux, inp[off+centre], inp[off+context_post], 
                             k, learning_rate, negsample_array, random_states,
                             use_hs, syn1, codes_array, points_array, code_lengths,
                             exp_table, exp_table_size, max_exp)


@cuda.jit(device=True)
def fast_sigmoid(f, exp_table, exp_table_size, max_exp):
    """
    Fast sigmoid using precomputed exp table.
    Based on word2vec.c exp table lookup.
    """
    if f <= -max_exp:
        return 0.0
    elif f >= max_exp:
        return 1.0
    else:
        idx = int((f + max_exp) * (exp_table_size / max_exp / 2.0))
        if idx < 0:
            idx = 0
        if idx >= exp_table_size:
            idx = exp_table_size - 1
        return exp_table[idx]


@cuda.jit(device=True)
def step_skipgram(thread_idx, w1, w2, calc_aux, x, y, k, learning_rate, negsample_array, random_states,
                  use_hs, syn1, codes_array, points_array, code_lengths,
                  exp_table, exp_table_size, max_exp):
    """
    Device function for Skip-gram gradient calculation.
    Based on word2vec.c Skip-gram implementation (lines 505-519).
    Supports both Hierarchical Softmax and Negative Sampling.
    """
    emb_dim = w1.shape[1]
    negs_arr_len = len(negsample_array)
    
    # Initialize error accumulator
    for i in range(emb_dim):
        calc_aux[thread_idx, i] = 0.0
    
    # Hierarchical Softmax (if enabled) - traverse tree for context word y
    if use_hs:
        codelen = code_lengths[y]
        max_code_len = codes_array.shape[1]  # Get max code length from array shape
        for d in range(codelen):
            if d >= max_code_len:
                break
            node_idx = points_array[y, d]
            if node_idx < 0:
                continue
            
            # Calculate dot product: w1[x] · syn1[node]
            f = 0.0
            for i in range(emb_dim):
                f += w1[x, i] * syn1[node_idx, i]
            
            # Early skip if f is outside range (same as original code)
            # This prevents unnecessary updates when sigmoid is saturated
            if f <= -max_exp:
                continue
            if f >= max_exp:
                continue
            
            # Get sigmoid from exp table (only if in range)
            sigmoid_val = fast_sigmoid(f, exp_table, exp_table_size, max_exp)
            
            # Get code bit (0 or 1)
            code_bit = codes_array[y, d]
            if code_bit < 0:
                continue
            
            # Calculate gradient: g = (1 - code_bit - sigmoid) * learning_rate
            g = (1.0 - float(code_bit) - sigmoid_val) * learning_rate
            
            # Propagate errors output -> hidden
            for i in range(emb_dim):
                calc_aux[thread_idx, i] += g * syn1[node_idx, i]
            
            # Learn weights hidden -> output
            for i in range(emb_dim):
                syn1[node_idx, i] += g * w1[x, i]
    
    # Negative Sampling (if enabled)
    if k > 0:
        # Positive sample: predict context word y
        dot_xy = 0.0
        for i in range(emb_dim):
            dot_xy += w1[x, i] * w2[y, i]
        s_xdy_m1 = fast_sigmoid(dot_xy, exp_table, exp_table_size, max_exp) - 1.0
        
        # Positive sample gradients
        for i in range(emb_dim):
            calc_aux[thread_idx, i] += -learning_rate * s_xdy_m1 * w2[y, i]
            w2[y, i] -= learning_rate * s_xdy_m1 * w1[x, i]
        
        # Negative samples
        for neg_sample in range(0, k):
            rnd = c_random.xoroshiro128p_uniform_float32(random_states, thread_idx)
            q_idx: int = int(math.floor(negs_arr_len * rnd))
            neg = negsample_array[q_idx]
            dot_xq = 0.0
            for i in range(emb_dim):
                dot_xq += w1[x, i] * w2[neg, i]
            s_dxq = fast_sigmoid(dot_xq, exp_table, exp_table_size, max_exp)
            
            # Negative sample gradients
            for i in range(emb_dim):
                calc_aux[thread_idx, i] -= learning_rate * s_dxq * w2[neg, i]
                w2[neg, i] -= learning_rate * s_dxq * w1[x, i]
    
    # Note: Original code does NOT use gradient clipping, only early skip
    # Gradient clipping may reduce training effectiveness
    # Update center word vector (same as original code)
    for i in range(emb_dim):
        w1[x, i] += calc_aux[thread_idx, i]


def train_skipgram(
        data_path: str,
        out_file_path: str,
        epochs: int,
        embed_dim: int = 100,
        min_occurs: int = 3,
        c: int = 5,
        k: int = 5,
        t: float = 1e-5,
        vocab_freq_exponent: float = 0.75,
        lr_max: float = 0.025,
        lr_min: float = 0.0025,
        cuda_threads_per_block: int = 32,
        hs: int = 0):
    """
    Train Skip-gram model.
    Based on word2vec.c Skip-gram implementation.
    
    Args:
        hs: Hierarchical Softmax flag (0=NS only, 1=HS, can combine with k>0)
    """
    params = {
        "model_type": "skipgram",
        "w2v_version": W2V_VERSION,
        "data_path": data_path,
        "out_file_path": out_file_path,
        "epochs": epochs,
        "embed_dim": embed_dim,
        "min_occurs": min_occurs,
        "c": c,
        "k": k,
        "t": t,
        "vocab_freq_exponent": vocab_freq_exponent,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "cuda_threads_per_block": cuda_threads_per_block,
        "hs": hs
    }
    stats = {}
    params_path = out_file_path + "_params.json"
    stats_path = out_file_path + "_stats.json"

    seed = 12345
    
    # Adjust learning rate based on training method
    original_lr_max = lr_max
    original_lr_min = lr_min
    
    if hs == 1 and k == 0:
        # HS only: Use same learning rate as NS (as per word2vec.c original)
        # No learning rate reduction needed - HS and NS use same LR schedule
        pass
        
    elif hs == 1 and k > 0:
        # HS + NS: Reduce learning rate to prevent gradient explosion
        print(f"⚠️  WARNING: Using both HS and NS together may cause issues.")
        print(f"   Consider using only one (hs=1, k=0) or (hs=0, k>0) for better results.")
        print(f"   Learning rate reduced by 50% to prevent gradient explosion.")
        lr_max = lr_max * 0.5
        lr_min = lr_min * 0.5
    
    if epochs > 1:
        lr_step = (lr_max - lr_min) / (epochs - 1)
    else:
        lr_step = (lr_max - lr_min)

    print(f"Skip-gram Training Parameters:")
    print(f"Seed: {seed}")
    print(f"Window size: {c}")
    if hs == 1:
        print(f"Hierarchical Softmax: Enabled")
    if k > 0:
        print(f"Negative samples: {k}")
    if original_lr_max != lr_max:
        print(f"Learning rate adjusted: {original_lr_max} → {lr_max} (reduced for stability)")
    print(f"Learning rate: {lr_max} → {lr_min} (step: {lr_step:.6f})")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Min word count: {min_occurs}")

    print(f"\nBuilding vocabulary from: {data_path}")
    start = time.time()

    vocab, w_to_i, word_counts = handle_vocab(data_path, min_occurs, freq_exponent=vocab_freq_exponent)
    ssw, negs = get_subsampling_weights_and_negative_sampling_array(vocab, t=t)
    vocab_size = len(vocab)
    print(f"Vocabulary built in {time.time() - start:.2f}s. Vocab size: {vocab_size:,}")
    
    # Create exp table
    print("Creating exp table for fast sigmoid...")
    exp_table = create_exp_table(EXP_TABLE_SIZE, MAX_EXP)
    
    # Setup Hierarchical Softmax if enabled
    use_hs = (hs == 1)
    syn1_cuda = None
    codes_array_cuda = None
    points_array_cuda = None
    code_lengths_cuda = None
    
    if use_hs:
        print("Creating Huffman tree for Hierarchical Softmax...")
        hs_start = time.time()
        codes_array, points_array, code_lengths = create_huffman_tree(word_counts, MAX_CODE_LENGTH)
        syn1 = init_hs_weight_matrix(vocab_size, embed_dim)
        print(f"Huffman tree created in {time.time() - hs_start:.2f}s")
        print(f"  Codes array shape: {codes_array.shape}")
        print(f"  Points array shape: {points_array.shape}")
        print(f"  Syn1 matrix shape: {syn1.shape}")

    data_files = get_data_file_names(data_path, seed=seed)
    print(f"Processing {len(data_files)} data files...")
    inps_, offs_, lens_ = read_all_data_files_ever(data_path, data_files, w_to_i)
    inps, offs, lens = (np.asarray(inps_, dtype=np.int32), 
                       np.asarray(offs_, dtype=np.int32), 
                       np.asarray(lens_, dtype=np.int32))
    sentence_count = len(lens)
    blocks: int = math.ceil(sentence_count / cuda_threads_per_block)
    
    print(f"Data loaded: {sentence_count:,} sentences, {len(inps):,} total words")
    print(f"CUDA config: {cuda_threads_per_block} threads/block, {blocks} blocks")

    # Initialize weight matrices
    data_init_start = time.time()
    w1, w2 = init_weight_matrices(vocab_size, embed_dim, seed=seed)
    calc_aux = np.zeros((sentence_count, embed_dim), dtype=np.float32)
    data_size_weights = 4 * (w1.size + w2.size)
    data_size_inputs = 4 * (inps.size + offs.size + lens.size + ssw.size + negs.size)
    data_size_aux = 4 * calc_aux.size
    print(f"Weight matrices initialized in {time.time()-data_init_start:.2f}s")
    print(f"Memory usage: {data_size_weights:,} weights + {data_size_inputs:,} inputs + {data_size_aux:,} aux = {data_size_weights+data_size_inputs+data_size_aux:,} bytes")

    # Transfer to GPU
    print("Transferring data to GPU...")
    data_transfer_start = time.time()
    inps_cuda, offs_cuda, lens_cuda = cuda.to_device(inps), cuda.to_device(offs), cuda.to_device(lens)
    ssw_cuda, negs_cuda = cuda.to_device(ssw), cuda.to_device(negs)
    w1_cuda, w2_cuda = cuda.to_device(w1), cuda.to_device(w2)
    calc_aux_cuda = cuda.to_device(calc_aux)
    exp_table_cuda = cuda.to_device(exp_table)
    
    if use_hs:
        syn1_cuda = cuda.to_device(syn1)
        codes_array_cuda = cuda.to_device(codes_array)
        points_array_cuda = cuda.to_device(points_array)
        code_lengths_cuda = cuda.to_device(code_lengths)
    
    print(f"Data transfer completed in {time.time()-data_transfer_start:.2f}s")

    stats["sentence_count"] = len(lens)
    stats["word_count"] = len(inps)
    stats["vocab_size"] = vocab_size
    stats["approx_data_size_weights"] = data_size_weights
    stats["approx_data_size_inputs"] = data_size_inputs
    stats["approx_data_size_aux"] = data_size_aux
    stats["approx_data_size_total"] = data_size_weights + data_size_inputs + data_size_aux

    # Initialize CUDA random states
    print(f"Initializing CUDA random states for {sentence_count:,} threads...")
    random_init_start = time.time()
    random_states_cuda = c_random.create_xoroshiro128p_states(sentence_count, seed=seed)
    print(f"CUDA random states initialized in {time.time()-random_init_start:.2f}s")

    # Prepare HS parameters (use dummy arrays if HS disabled)
    if not use_hs:
        # Create dummy arrays for HS (will not be used, but needed for kernel signature)
        dummy_syn1 = cuda.device_array((1, embed_dim), dtype=np.float32)
        dummy_codes = cuda.device_array((vocab_size, MAX_CODE_LENGTH), dtype=np.int32)
        dummy_points = cuda.device_array((vocab_size, MAX_CODE_LENGTH), dtype=np.int32)
        dummy_lengths = cuda.device_array(vocab_size, dtype=np.int32)
        syn1_param = dummy_syn1
        codes_param = dummy_codes
        points_param = dummy_points
        lengths_param = dummy_lengths
    else:
        syn1_param = syn1_cuda
        codes_param = codes_array_cuda
        points_param = points_array_cuda
        lengths_param = code_lengths_cuda
    
    print_norms(w1_cuda)
    print(f"\nStarting Skip-gram training - {epochs} epochs...")
    epoch_times = []
    calc_start = time.time()
    
    for epoch in range(0, epochs):
        lr = lr_max - (epoch * lr_step)
        epoch_start = time.time()
        
        # Launch CUDA kernel
        calc_skipgram[blocks, cuda_threads_per_block](
            sentence_count, c, k, lr, w1_cuda, w2_cuda, calc_aux_cuda, 
            random_states_cuda, ssw_cuda, negs_cuda, inps_cuda, offs_cuda, lens_cuda,
            use_hs, syn1_param, codes_param, points_param, lengths_param,
            exp_table_cuda, EXP_TABLE_SIZE, MAX_EXP)
        
        print(f"  Epoch {epoch+1} kernel launched in {time.time()-epoch_start:.2f}s (LR: {lr:.6f})")
        
        # Synchronize
        sync_start = time.time()
        cuda.synchronize()
        epoch_times.append(time.time()-epoch_start)
        print(f"  Synchronized in {time.time()-sync_start:.2f}s")
        print(f"  → Epoch {epoch+1} completed in {epoch_times[-1]:.2f}s")
    
    print(f"\nSkip-gram training completed!")
    print(f"Epoch times - Min: {min(epoch_times):.2f}s, Avg: {np.mean(epoch_times):.2f}s, Max: {max(epoch_times):.2f}s")
    print(f"Total training time: {time.time()-calc_start:.2f}s")
    print(f"Total time: {time.time()-start:.2f}s")
    
    print_norms(w1_cuda)
    
    # Save results
    stats["epoch_time_min_seconds"] = min(epoch_times)
    stats["epoch_time_avg_seconds"] = np.mean(epoch_times)
    stats["epoch_time_max_seconds"] = max(epoch_times)
    stats["epoch_time_total_seconds"] = sum(epoch_times)
    stats["epoch_times_all_seconds"] = epoch_times
    
    print(f"Saving Skip-gram vectors to: {out_file_path}")
    write_vectors(w1_cuda, vocab, out_file_path)
    
    print(f"Saving parameters to: {params_path}")
    write_json(params, params_path)
    
    print(f"Saving statistics to: {stats_path}")
    write_json(stats, stats_path)
    
    print("Skip-gram training completed successfully!")
