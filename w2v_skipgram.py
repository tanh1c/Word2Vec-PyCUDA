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
        hs: int = 0,
        max_memory_gb: float = 70.0,
        max_words: int = None,
        vocab: list = None,
        w_to_i: dict = None,
        word_counts: list = None,
        ssw: np.ndarray = None,
        negs: np.ndarray = None):
    """
    Train Skip-gram model.
    Based on word2vec.c Skip-gram implementation.
    
    Args:
        hs: Hierarchical Softmax flag (0=NS only, 1=HS, can combine with k>0)
        max_memory_gb: Maximum GPU memory usage in GB. If estimated memory exceeds this,
                       the dataset will be automatically split into batches for processing.
                       Default: 70.0 GB (safe for A100 80GB GPU)
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
    
    # Learning rate schedule
    # For multiple epochs: decrease between epochs
    # For all epochs: decrease LINEARLY within epoch (as per word2vec.c)
    if epochs > 1:
        lr_step = (lr_max - lr_min) / (epochs - 1)
    else:
        lr_step = 0.0  # Not used for single epoch (LR decays within epoch)

    print(f"Skip-gram Training Parameters:")
    print(f"Seed: {seed}")
    print(f"Window size: {c}")
    if hs == 1:
        print(f"Hierarchical Softmax: Enabled")
    if k > 0:
        print(f"Negative samples: {k}")
    if original_lr_max != lr_max:
        print(f"Learning rate adjusted: {original_lr_max} → {lr_max} (reduced for stability)")
    if epochs == 1:
        print(f"Learning rate: {lr_max} → ~0 (will decrease LINEARLY within epoch, as per word2vec.c)")
    else:
        print(f"Learning rate: {lr_max} → {lr_min} (step: {lr_step:.6f} between epochs, also decreases LINEARLY within each epoch)")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Min word count: {min_occurs}")

    # Start timing for total execution
    start = time.time()

    # Build vocabulary if not provided (for reuse when training both models)
    if vocab is None or w_to_i is None or word_counts is None:
        print(f"\nBuilding vocabulary from: {data_path}")
        vocab_start = time.time()
        vocab, w_to_i, word_counts = handle_vocab(data_path, min_occurs, freq_exponent=vocab_freq_exponent, use_cache=True)
        vocab_size = len(vocab)
        build_time = time.time() - vocab_start
        print(f"Vocabulary {'loaded from cache' if build_time < 1.0 else 'built'} in {build_time:.2f}s. Vocab size: {vocab_size:,}")
    else:
        vocab_size = len(vocab)
        print(f"\nUsing pre-built vocabulary. Vocab size: {vocab_size:,}")
    
    # Build subsampling weights and negative sampling array if not provided
    if ssw is None or negs is None:
        ssw, negs = get_subsampling_weights_and_negative_sampling_array(vocab, t=t)
    
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
    if max_words is not None:
        print(f"  ⚠️  Limiting to {max_words:,} total words (will stop early if reached)")
    inps_, offs_, lens_ = read_all_data_files_ever(data_path, data_files, w_to_i, max_words=max_words)
    inps, offs, lens = (np.asarray(inps_, dtype=np.int32), 
                       np.asarray(offs_, dtype=np.int32), 
                       np.asarray(lens_, dtype=np.int32))
    sentence_count = len(lens)
    total_words = len(inps)  # Total words for LR decay calculation
    
    print(f"Data loaded: {sentence_count:,} sentences, {total_words:,} total words")

    # Initialize weight matrices
    data_init_start = time.time()
    w1, w2 = init_weight_matrices(vocab_size, embed_dim, seed=seed)
    data_size_weights = 4 * (w1.size + w2.size)
    data_size_inputs = 4 * (inps.size + offs.size + lens.size + ssw.size + negs.size)
    
    # Calculate memory usage and determine batch size
    weights_gb = data_size_weights / (1024**3)
    inputs_gb = data_size_inputs / (1024**3)
    
    # Estimate calc_aux memory for full dataset
    calc_aux_size_full = sentence_count * embed_dim * 4
    calc_aux_gb_full = calc_aux_size_full / (1024**3)
    total_memory_gb = weights_gb + inputs_gb + calc_aux_gb_full
    
    # Determine if batch processing is needed
    use_batch_processing = (total_memory_gb > max_memory_gb)
    
    if use_batch_processing:
        # Calculate batch size based on available memory
        available_memory_gb = max_memory_gb - weights_gb - inputs_gb
        # Reserve 5GB for overhead
        available_memory_gb = max(1.0, available_memory_gb - 5.0)
        
        # Calculate max sentences per batch
        bytes_per_sentence = embed_dim * 4  # float32
        max_batch_sentences = int((available_memory_gb * 1024**3) / bytes_per_sentence)
        
        # Round down to nice numbers for better performance
        if max_batch_sentences >= 10_000_000:
            batch_size = 10_000_000
        elif max_batch_sentences >= 5_000_000:
            batch_size = 5_000_000
        elif max_batch_sentences >= 2_000_000:
            batch_size = 2_000_000
        elif max_batch_sentences >= 1_000_000:
            batch_size = 1_000_000
        else:
            batch_size = max(100_000, max_batch_sentences)
        
        num_batches = math.ceil(sentence_count / batch_size)
        batch_aux_gb = (batch_size * embed_dim * 4) / (1024**3)
        batch_total_gb = weights_gb + inputs_gb + batch_aux_gb
        
        print(f"\n⚠️  Memory usage would be {total_memory_gb:.1f} GB (exceeds {max_memory_gb} GB limit)")
        print(f"📦 Using batch processing: {num_batches} batches, {batch_size:,} sentences/batch")
        print(f"   Memory per batch: {batch_total_gb:.1f} GB (calc_aux: {batch_aux_gb:.1f} GB)")
    else:
        batch_size = sentence_count
        num_batches = 1
        print(f"\n✅ Memory usage: {total_memory_gb:.1f} GB (within {max_memory_gb} GB limit)")
        print(f"   Processing all {sentence_count:,} sentences in one batch")
    
    blocks: int = math.ceil(batch_size / cuda_threads_per_block)
    print(f"CUDA config: {cuda_threads_per_block} threads/block, {blocks} blocks per batch")

    # Transfer to GPU - Transfer weights and vocab arrays (these are shared across batches)
    print("Transferring data to GPU...")
    data_transfer_start = time.time()
    ssw_cuda, negs_cuda = cuda.to_device(ssw), cuda.to_device(negs)
    w1_cuda, w2_cuda = cuda.to_device(w1), cuda.to_device(w2)
    exp_table_cuda = cuda.to_device(exp_table)
    
    # Keep input arrays on CPU - will slice and transfer per batch
    # This saves GPU memory
    
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
    stats["use_batch_processing"] = use_batch_processing
    if use_batch_processing:
        stats["batch_size"] = batch_size
        stats["num_batches"] = num_batches
        batch_aux_size = batch_size * embed_dim * 4
        stats["approx_data_size_aux_per_batch"] = batch_aux_size
        stats["approx_data_size_total"] = data_size_weights + data_size_inputs + batch_aux_size
    else:
        data_size_aux = 4 * (sentence_count * embed_dim)
        stats["approx_data_size_aux"] = data_size_aux
        stats["approx_data_size_total"] = data_size_weights + data_size_inputs + data_size_aux

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
    
    # Track total words processed across all epochs (as per word2vec.c)
    # Learning rate decays based on total words processed, not per epoch
    # Use int64 to avoid overflow with large datasets and multiple epochs
    words_processed_total = np.int64(0)
    total_words_for_training = np.int64(epochs) * np.int64(total_words)
    
    for epoch in range(0, epochs):
        epoch_start = time.time()
        
        # Process each batch
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, sentence_count)
            batch_sentence_count = batch_end - batch_start
            
            if num_batches > 1:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}: sentences {batch_start:,}-{batch_end:,}")
            
            # Calculate word offset for this batch (offsets are cumulative)
            batch_word_start = offs[batch_start] if batch_start < len(offs) else 0
            batch_word_end = offs[batch_end] if batch_end < len(offs) else len(inps)
            batch_word_count = batch_word_end - batch_word_start
            
            # Calculate learning rate for this batch (linear decay as per word2vec.c)
            # Formula from word2vec.c line 397: alpha = starting_alpha * (1 - word_count_actual / (iter * train_words + 1))
            # word_count_actual is total words processed across all epochs
            # This ensures LR decreases linearly from lr_max to ~0 over entire training
            denominator = total_words_for_training + 1
            current_lr = lr_max * (1.0 - words_processed_total / denominator) if denominator > 0 else lr_max
            
            # Apply minimum threshold (as per word2vec.c line 398: min = starting_alpha * 0.0001)
            min_lr_threshold = lr_max * 0.0001
            current_lr = max(current_lr, min_lr_threshold)
            
            # Also apply lr_min as additional constraint (for multi-epoch training)
            if epochs > 1:
                current_lr = max(current_lr, lr_min)
            
            if num_batches > 1 and batch_idx == 0:
                print(f"    Learning rate: {current_lr:.6f} (decaying linearly, progress: {words_processed_total/total_words_for_training*100:.1f}%)")
            
            # Create batch arrays (slicing from CPU arrays)
            batch_lens = lens[batch_start:batch_end]
            batch_offs_local = offs[batch_start:batch_end] - batch_word_start  # Adjust offsets to start from 0
            batch_inps_local = inps[batch_word_start:batch_word_end]
            
            # Transfer batch arrays to GPU
            batch_lens_cuda = cuda.to_device(batch_lens)
            batch_offs_cuda = cuda.to_device(batch_offs_local)
            batch_inps_cuda = cuda.to_device(batch_inps_local)
            
            # Create calc_aux for this batch
            batch_calc_aux = np.zeros((batch_sentence_count, embed_dim), dtype=np.float32)
            batch_calc_aux_cuda = cuda.to_device(batch_calc_aux)
            
            # Create random states for this batch
            batch_random_states_cuda = c_random.create_xoroshiro128p_states(
                batch_sentence_count, seed=seed + epoch * 10000 + batch_idx * 100
            )
            
            # Launch CUDA kernel for this batch with current learning rate
            batch_blocks = math.ceil(batch_sentence_count / cuda_threads_per_block)
            calc_skipgram[batch_blocks, cuda_threads_per_block](
                batch_sentence_count, c, k, current_lr, w1_cuda, w2_cuda, batch_calc_aux_cuda, 
                batch_random_states_cuda, ssw_cuda, negs_cuda, batch_inps_cuda, 
                batch_offs_cuda, batch_lens_cuda,
                use_hs, syn1_param, codes_param, points_param, lengths_param,
                exp_table_cuda, EXP_TABLE_SIZE, MAX_EXP)
            
            # Update total words processed counter (as per word2vec.c)
            # Note: Actual words processed may vary due to subsampling, but this is an approximation
            # Use int64 to avoid overflow with large datasets and multiple epochs
            words_processed_total = np.int64(words_processed_total) + np.int64(batch_word_count)
            
            # Free batch arrays from GPU memory
            del batch_lens_cuda, batch_offs_cuda, batch_inps_cuda, batch_calc_aux_cuda, batch_random_states_cuda
        
        # Synchronize after all batches
        sync_start = time.time()
        cuda.synchronize()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Final LR after epoch (using same formula as word2vec.c)
        denominator = total_words_for_training + 1
        final_lr = lr_max * (1.0 - words_processed_total / denominator) if denominator > 0 else lr_max
        final_lr = max(final_lr, lr_max * 0.0001)
        if epochs > 1:
            final_lr = max(final_lr, lr_min)
        
        progress_percent = (words_processed_total / total_words_for_training * 100) if total_words_for_training > 0 else 0.0
        print(f"  Epoch {epoch+1} completed in {epoch_time:.2f}s (LR: {final_lr:.6f}, Progress: {progress_percent:.1f}%)")
    
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
