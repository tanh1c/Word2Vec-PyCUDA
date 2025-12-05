#!/usr/bin/env python3
# Copyright 2024 Word2Vec Implementation
# Main script to run complete Word2Vec pipeline - NOTEBOOK VERSION
# Modified to run in notebook (no interactive menu, use config variables)

# ============================================
# CONFIGURATION - Change these values
# ============================================
# Dataset selection
use_wmt14 = False      # True to use WMT14 News
dataset_name = "Text8" # "Text8" or "WMT14 News"

# Dataset size (only for WMT14)
max_sentences = None   # None = full dataset, or number like 100000
max_files = None       # None = all files, or number like 10
max_words = None       # None = no limit, or number like 700000000 for 700M words

# Training method
use_hs_only = False    # True = HS only (HS=1, k=0)

# Model selection
should_train_skipgram = True  # True to train Skip-gram
should_train_cbow = True      # True to train CBOW

# Phrase detection
use_phrases = False    # True to enable phrase detection

# ============================================
# IMPORTS
# ============================================
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: Numba-cuda configuration and GPU check should be done by running colab_setup_all.py first
# This file assumes colab_setup_all.py has already been executed

from data_handler import download_text8, preprocess_text8, download_wmt14_news, preprocess_wmt14_news
from w2v_skipgram import train_skipgram
from w2v_cbow import train_cbow
from evaluation import word_analogy_test, similarity_test, save_evaluation_results, compare_models

def print_section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_summary(sg_acc: float, cbow_acc: float, sg_stats: dict, cbow_stats: dict,
                 sg_sem: float = None, sg_syn: float = None, 
                 cbow_sem: float = None, cbow_syn: float = None):
    """Print final summary of results."""
    print_section_header("FINAL SUMMARY")
    
    print(f"Model Performance:")
    if sg_acc is not None:
    print(f"  Skip-gram accuracy: {sg_acc:.4f} ({sg_acc*100:.2f}%)")
        if sg_sem is not None and sg_syn is not None:
            print(f"    - Semantic:  {sg_sem:.4f} ({sg_sem*100:.2f}%)")
            print(f"    - Syntactic: {sg_syn:.4f} ({sg_syn*100:.2f}%)")
    if cbow_acc is not None:
    print(f"  CBOW accuracy: {cbow_acc:.4f} ({cbow_acc*100:.2f}%)")
        if cbow_sem is not None and cbow_syn is not None:
            print(f"    - Semantic:  {cbow_sem:.4f} ({cbow_sem*100:.2f}%)")
            print(f"    - Syntactic: {cbow_syn:.4f} ({cbow_syn*100:.2f}%)")
    if sg_acc is not None and cbow_acc is not None:
    print(f"  Difference: {sg_acc - cbow_acc:.4f} ({(sg_acc - cbow_acc)*100:+.2f}%)")
    
    has_stats = sg_stats or cbow_stats
    if has_stats:
        print(f"\nTraining Times:")
        if sg_stats:
            sg_time = sg_stats.get('epoch_time_total_seconds', 0)
            print(f"  Skip-gram: {sg_time:.2f}s")
        if cbow_stats:
            cbow_time = cbow_stats.get('epoch_time_total_seconds', 0)
            print(f"  CBOW: {cbow_time:.2f}s")
    if sg_stats and cbow_stats:
        sg_time = sg_stats.get('epoch_time_total_seconds', 0)
        cbow_time = cbow_stats.get('epoch_time_total_seconds', 0)
        print(f"  Difference: {sg_time - cbow_time:.2f}s")
        
        print(f"\nData Processed:")
        stats = sg_stats if sg_stats else cbow_stats
        if stats:
            words = stats.get('word_count', 0)
            print(f"  Words: {words:,}")
            print(f"  Sentences: {stats.get('sentence_count', 0):,}")
            print(f"  Vocabulary: {stats.get('vocab_size', 0):,}")
    
    print(f"\nOutput Files:")
    if sg_acc is not None:
    print(f"  Skip-gram vectors: ./output/vectors_skipgram")
        print(f"  Skip-gram evaluation: ./output/skipgram_eval.json")
        print(f"  Skip-gram statistics: ./output/vectors_skipgram_stats.json")
    if cbow_acc is not None:
    print(f"  CBOW vectors: ./output/vectors_cbow")
        print(f"  CBOW evaluation: ./output/cbow_eval.json")
        print(f"  CBOW statistics: ./output/vectors_cbow_stats.json")


def main():
    """Main pipeline execution - NOTEBOOK VERSION."""
    # Config is set at the top of the file (no interactive menu needed)
    
    # Print configuration
    print(f"\nDataset: {dataset_name}")
    if use_wmt14:
        print("  - WMT14/WMT15 News Crawl (combines WMT14 2012-2013 + WMT15 2014)")
        print("  - Higher quality news articles")
        if max_words:
            print(f"  - Limited to {max_words:,} words ({max_words/1e6:.1f}M words)")
        elif max_sentences:
            print(f"  - Limited to {max_sentences:,} sentences")
    else:
        print("  - Text8 (17M words, ~100MB)")
        print("  - Smaller, faster to download and process")
    
    if use_hs_only:
        print("  🎯 Training: Hierarchical Softmax ONLY (HS=1, k=0)")
    else:
        print("  🎯 Training: Negative Sampling ONLY (HS=0, k=5)")
    
    if not should_train_skipgram:
        print("  ⏭️  Skip-gram training: Disabled")
    if not should_train_cbow:
        print("  ⏭️  CBOW training: Disabled")
    
    if use_phrases:
        print("  🔗 Phrase detection: Enabled")
    
    # NOTE: GPU check is skipped here because it should already be done in colab_setup_all.py
    # If you run this file independently, make sure to run colab_setup_all.py first
    
    # 1. Download & Preprocess Data
    print_section_header(f"STEP 1: DOWNLOADING & PREPROCESSING {dataset_name.upper()}")
    data_dir = "./data"
    
    if use_phrases:
        print("  🔗 Phrase detection: Enabled (will combine frequent bigrams)")
    
    if use_wmt14:
        news_file = download_wmt14_news(data_dir)
        processed_dir = preprocess_wmt14_news(news_file, "./data/wmt14_processed", 
                                            max_sentences=max_sentences, max_files=max_files,
                                            use_phrases=use_phrases)
    else:
        text8_file = download_text8(data_dir)
        processed_dir = preprocess_text8(text8_file, "./data/text8_processed",
                                        use_phrases=use_phrases)
    
    # Prepare training parameters (used by both models)
    epochs_value = 1  # Set epochs here for consistency
    base_params = {
        "epochs": epochs_value,
        "embed_dim": 600,
        "min_occurs": 5,
        "c": 5,
        "k": 0 if use_hs_only else 5,
        "t": 1e-5,
        "vocab_freq_exponent": 0.75,
        "lr_max": 0.025,
        # For 1 epoch with large dataset, keep learning rate high (as in paper)
        "lr_min": 0.025 if epochs_value == 1 else 0.0001,
        "cuda_threads_per_block": 512,  # Optimized for A100 GPU
        "hs": 1 if use_hs_only else 0,
        "max_words": max_words  # Limit total words for training (None = no limit)
    }
    
    # Build vocabulary once if training both models (to save time)
    shared_vocab = None
    shared_w_to_i = None
    shared_word_counts = None
    shared_ssw = None
    shared_negs = None
    
    if should_train_skipgram and should_train_cbow:
        print_section_header("STEP 2: BUILDING SHARED VOCABULARY")
        print("  ℹ️  Building vocabulary once for both Skip-gram and CBOW models")
        print("  ℹ️  Vocabulary will be cached for future runs (even with different epochs/dim)")
        from w2v_common import handle_vocab, get_subsampling_weights_and_negative_sampling_array
        import time
        start = time.time()
        shared_vocab, shared_w_to_i, shared_word_counts = handle_vocab(
            processed_dir, base_params["min_occurs"], freq_exponent=base_params["vocab_freq_exponent"], use_cache=True
        )
        shared_ssw, shared_negs = get_subsampling_weights_and_negative_sampling_array(shared_vocab, t=base_params["t"])
        vocab_size = len(shared_vocab)
        build_time = time.time() - start
        print(f"  ✓ Vocabulary {'loaded from cache' if build_time < 1.0 else 'built'} in {build_time:.2f}s. Vocab size: {vocab_size:,}")
        print(f"  ✓ Vocabulary will be reused for both models\n")
    
    # 2/3. Train Skip-gram (if selected)
    if should_train_skipgram:
        step_num = 3 if (should_train_skipgram and should_train_cbow) else 2
        print_section_header(f"STEP {step_num}: TRAINING SKIP-GRAM MODEL")
        skipgram_params = base_params.copy()
        
        if epochs_value == 1:
            print("  ℹ️  Using 1 epoch: Learning rate will be kept constant at 0.025 (as per paper)")
    
    print("Skip-gram parameters:")
    for key, value in skipgram_params.items():
        print(f"  {key}: {value}")
    
        # Validate: HS and NS cannot be used together
    if skipgram_params["hs"] == 1 and skipgram_params["k"] > 0:
            raise ValueError("Error: Cannot use HS (hs=1) and Negative Sampling (k>0) together. Please choose either HS only (hs=1, k=0) or NS only (hs=0, k>0).")
        
        # Pass shared vocabulary if available
        if shared_vocab is not None:
            train_skipgram(processed_dir, "./output/vectors_skipgram", 
                          vocab=shared_vocab, w_to_i=shared_w_to_i, word_counts=shared_word_counts,
                          ssw=shared_ssw, negs=shared_negs, **skipgram_params)
        else:
    train_skipgram(processed_dir, "./output/vectors_skipgram", **skipgram_params)
    else:
        step_num = 3 if (should_train_skipgram and should_train_cbow) else 2
        print_section_header(f"STEP {step_num}: SKIPPING SKIP-GRAM TRAINING")
        print("  ⏭️  Skip-gram training skipped as requested")
        skipgram_params = base_params.copy()  # Still need params for CBOW if training both
    
    # 4. Train CBOW (if selected)
    if should_train_cbow:
    print_section_header("STEP 4: TRAINING CBOW MODEL")
        cbow_params = base_params.copy()
        # CBOW uses same learning rate as Skip-gram (0.025) to prevent gradient explosion
        cbow_params["lr_max"] = 0.025
        # For 1 epoch with large dataset, keep learning rate high (same as Skip-gram)
        cbow_params["lr_min"] = 0.025 if epochs_value == 1 else 0.0001
        
        if epochs_value == 1:
            print("  ℹ️  Using 1 epoch: Learning rate will be kept constant at 0.025 (same as Skip-gram)")
    
    print("CBOW parameters:")
    for key, value in cbow_params.items():
        print(f"  {key}: {value}")
    
        # Validate: HS and NS cannot be used together
    if cbow_params["hs"] == 1 and cbow_params["k"] > 0:
            raise ValueError("Error: Cannot use HS (hs=1) and Negative Sampling (k>0) together. Please choose either HS only (hs=1, k=0) or NS only (hs=0, k>0).")
        
        # Pass shared vocabulary if available
        if shared_vocab is not None:
            train_cbow(processed_dir, "./output/vectors_cbow",
                      vocab=shared_vocab, w_to_i=shared_w_to_i, word_counts=shared_word_counts,
                      ssw=shared_ssw, negs=shared_negs, **cbow_params)
        else:
    train_cbow(processed_dir, "./output/vectors_cbow", **cbow_params)
    else:
        print_section_header("STEP 4: SKIPPING CBOW TRAINING")
        print("  ⏭️  CBOW training skipped as requested")
    
    # 5. Evaluate Skip-gram (if trained)
    sg_result = None
    sg_details = None
    sg_sem = None
    sg_syn = None
    sg_total = None
    sg_acc = None
    sg_sim = None
    
    if should_train_skipgram:
    print_section_header("STEP 5: EVALUATING SKIP-GRAM MODEL")
        sg_result, sg_details = word_analogy_test("./output/vectors_skipgram")

        sg_sem   = sg_result["semantic_accuracy"]
        sg_syn   = sg_result["syntactic_accuracy"]
        sg_total = sg_result["total_accuracy"]
        sg_acc   = sg_total  # Total accuracy for comparison functions

    sg_sim = similarity_test("./output/vectors_skipgram")

    save_evaluation_results({
            "semantic_accuracy": sg_sem,
            "syntactic_accuracy": sg_syn,
            "total_accuracy": sg_total,
        "details": sg_details,
        "similarity_test": sg_sim
    }, "./output/skipgram_eval.json")
    else:
        print_section_header("STEP 5: SKIPPING SKIP-GRAM EVALUATION")
        print("  ⏭️  Skip-gram evaluation skipped (model not trained)")

    # 6. Evaluate CBOW (if trained)
    cbow_result = None
    cbow_details = None
    cbow_sem = None
    cbow_syn = None
    cbow_total = None
    cbow_acc = None
    cbow_sim = None
    
    if should_train_cbow:
    print_section_header("STEP 6: EVALUATING CBOW MODEL")
        cbow_result, cbow_details = word_analogy_test("./output/vectors_cbow")

        cbow_sem   = cbow_result["semantic_accuracy"]
        cbow_syn   = cbow_result["syntactic_accuracy"]
        cbow_total = cbow_result["total_accuracy"]
        cbow_acc   = cbow_total  # Total accuracy for comparison functions

    cbow_sim = similarity_test("./output/vectors_cbow")

    save_evaluation_results({
            "semantic_accuracy": cbow_sem,
            "syntactic_accuracy": cbow_syn,
            "total_accuracy": cbow_total,
        "details": cbow_details,
        "similarity_test": cbow_sim
    }, "./output/cbow_eval.json")
    else:
        print_section_header("STEP 6: SKIPPING CBOW EVALUATION")
        print("  ⏭️  CBOW evaluation skipped (model not trained)")
        
    # 7. Model Comparison (Custom Skip-gram vs CBOW) - only if both trained
    if should_train_skipgram and should_train_cbow:
        print_section_header("STEP 7: COMPARING CUSTOM MODELS (Skip-gram vs CBOW)")
    # Pass pre-computed accuracy values to avoid re-evaluating
    comparison = compare_models("./output/vectors_skipgram", "./output/vectors_cbow",
                                sg_acc=sg_acc, sg_details=sg_details,
                                cbow_acc=cbow_acc, cbow_details=cbow_details)
    else:
        print_section_header("STEP 7: SKIPPING MODEL COMPARISON")
        if should_train_skipgram:
            print("  ⏭️  Model comparison skipped (CBOW not trained)")
        elif should_train_cbow:
            print("  ⏭️  Model comparison skipped (Skip-gram not trained)")
    
    # Load statistics for summary
    sg_stats = {}
    cbow_stats = {}
    
    try:
        import json
        if should_train_skipgram:
            try:
        with open("./output/vectors_skipgram_stats.json", "r") as f:
            sg_stats = json.load(f)
            except FileNotFoundError:
                pass
        if should_train_cbow:
            try:
        with open("./output/vectors_cbow_stats.json", "r") as f:
            cbow_stats = json.load(f)
    except FileNotFoundError:
                pass
    except Exception:
        print("Warning: Could not load statistics files")
    
    # Final Summary
    print_summary(sg_acc, cbow_acc, sg_stats, cbow_stats,
                 sg_sem=sg_sem, sg_syn=sg_syn,
                 cbow_sem=cbow_sem, cbow_syn=cbow_syn)
    
    print(f"\n🎉 Word2Vec training and evaluation completed successfully!")
    print(f"📁 Check the ./output/ directory for all results.")
    
    print(f"\nDataset used: {dataset_name}")


if __name__ == "__main__":
    # In notebook, don't use sys.exit() as it will cause SystemExit exception
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        raise  # Re-raise to show error in notebook
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Don't use sys.exit() in notebook - re-raise exception
        raise  # Re-raise to show error in notebook
