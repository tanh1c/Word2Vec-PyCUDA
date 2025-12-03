#!/usr/bin/env python3
# Copyright 2024 Word2Vec Implementation
# Main script to run complete Word2Vec pipeline - NOTEBOOK VERSION
# Đã chỉnh sửa để chạy trong notebook (bỏ interactive menu, thay bằng config variables)

# ============================================
# CONFIGURATION - Thay đổi các giá trị này
# ============================================
# Dataset selection
use_wikipedia = False  # True để dùng Wikipedia
use_wmt14 = False      # True để dùng WMT14 News
dataset_name = "Text8" # "Text8", "Wikipedia", hoặc "WMT14 News"

# Dataset size (chỉ cho WMT14)
max_sentences = None   # None = full dataset, hoặc số như 100000
max_files = None       # None = all files, hoặc số như 10

# Training method
use_hs_only = False    # True = HS only (HS=1, k=0)
use_hs = False         # True = HS + NS (HS=1, k=5) hoặc HS only
# use_hs_only và use_hs không thể cùng True (use_hs_only sẽ override)

# Phrase detection
use_phrases = False    # True để enable phrase detection

# Gensim training
use_gensim = False     # True để train Gensim models

# Stop after evaluation
stop_after_eval = False # True để skip visualization

# ============================================
# IMPORTS
# ============================================
import os
import sys
from pathlib import Path

# Configure numba-cuda for CUDA PTX compatibility (Official Solution)
# Based on: https://github.com/googlecolab/colabtools/issues/5081
try:
    from numba import config
    config.CUDA_ENABLE_PYNVJITLINK = 1
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    print("✓ numba-cuda configuration set for CUDA PTX compatibility")
except ImportError:
    print("⚠️  numba not available - CUDA configuration skipped")
except Exception as e:
    print(f"⚠️  Could not configure numba: {e}")

# Import các modules (đã được load trong các cell trước)
from data_handler import download_text8, preprocess_text8, download_wmt14_news, preprocess_wmt14_news, download_wikipedia, preprocess_wikipedia
from w2v_skipgram import train_skipgram
from w2v_cbow import train_cbow
from evaluation import word_analogy_test, similarity_test, save_evaluation_results, compare_models, train_gensim_models, evaluate_gensim_models, compare_with_gensim
from visualization import plot_tsne, plot_similarity_heatmap, plot_training_comparison, plot_accuracy_comparison


def check_gpu_availability():
    """Check if CUDA GPU is available."""
    try:
        from numba import cuda
        if cuda.is_available():
            device = cuda.get_current_device()
            print(f"✓ CUDA GPU available: {device.name}")
            
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
            print("✗ CUDA GPU not available")
            return False
    except ImportError:
        print("✗ Numba CUDA not available")
        return False


def print_section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_summary(sg_acc: float, cbow_acc: float, sg_stats: dict, cbow_stats: dict):
    """Print final summary of results."""
    print_section_header("FINAL SUMMARY")
    
    print(f"Model Performance:")
    print(f"  Skip-gram accuracy: {sg_acc:.4f} ({sg_acc*100:.2f}%)")
    print(f"  CBOW accuracy: {cbow_acc:.4f} ({cbow_acc*100:.2f}%)")
    print(f"  Difference: {sg_acc - cbow_acc:.4f} ({(sg_acc - cbow_acc)*100:+.2f}%)")
    
    if sg_stats and cbow_stats:
        sg_time = sg_stats.get('epoch_time_total_seconds', 0)
        cbow_time = cbow_stats.get('epoch_time_total_seconds', 0)
        print(f"\nTraining Times:")
        print(f"  Skip-gram: {sg_time:.2f}s")
        print(f"  CBOW: {cbow_time:.2f}s")
        print(f"  Difference: {sg_time - cbow_time:.2f}s")
        
        sg_words = sg_stats.get('word_count', 0)
        cbow_words = cbow_stats.get('word_count', 0)
        print(f"\nData Processed:")
        print(f"  Words: {sg_words:,}")
        print(f"  Sentences: {sg_stats.get('sentence_count', 0):,}")
        print(f"  Vocabulary: {sg_stats.get('vocab_size', 0):,}")
    
    print(f"\nOutput Files:")
    print(f"  Skip-gram vectors: ./output/vectors_skipgram")
    print(f"  CBOW vectors: ./output/vectors_cbow")
    print(f"  Visualizations: ./output/*.png")
    print(f"  Evaluation results: ./output/*.json")


def main():
    """Main pipeline execution - NOTEBOOK VERSION."""
    # Config được set ở đầu file (không cần interactive menu)
    
    # Print configuration
    print(f"\nDataset: {dataset_name}")
    if use_wikipedia:
        print("  - Wikipedia Dump (English, ~500MB download)")
        print("  - High quality encyclopedia articles")
    elif use_wmt14:
        print("  - WMT14 News Crawl (850M words, ~3.2GB)")
        print("  - Higher quality news articles")
        if max_sentences:
            print(f"  - Limited to {max_sentences:,} sentences")
    else:
        print("  - Text8 Wikipedia (17M words, ~100MB)")
        print("  - Smaller, faster to download and process")
    
    if use_hs_only:
        print("  🎯 Training: Hierarchical Softmax ONLY (HS=1, k=0)")
    elif use_hs:
        print("  🎯 Training: Hierarchical Softmax + Negative Sampling (HS=1, k>0)")
    else:
        print("  🎯 Training: Negative Sampling ONLY (HS=0, k>0)")
    
    if use_phrases:
        print("  🔗 Phrase detection: Enabled")
    
    if use_gensim:
        print("  📚 Gensim training: Enabled")
    
    if stop_after_eval:
        print("  ⏹️  Will stop after Step 6 (Evaluation)")
    
    # 1. Setup & GPU Check
    print_section_header("STEP 1: GPU AVAILABILITY CHECK")
    if not check_gpu_availability():
        print("⚠️  Warning: No GPU detected. Training will be slow on CPU.")
        print("Continuing anyway...")
        # Bỏ input() - không hoạt động tốt trong notebook
    
    # 2. Download & Preprocess Data
    print_section_header(f"STEP 2: DOWNLOADING & PREPROCESSING {dataset_name.upper()}")
    data_dir = "./data"
    
    if use_phrases:
        print("  🔗 Phrase detection: Enabled (will combine frequent bigrams)")
    
    if use_wikipedia:
        wiki_dir = download_wikipedia(data_dir)
        processed_dir = preprocess_wikipedia(wiki_dir, "./data/wikipedia_processed")
    elif use_wmt14:
        news_file = download_wmt14_news(data_dir)
        processed_dir = preprocess_wmt14_news(news_file, "./data/wmt14_processed", 
                                            max_sentences=max_sentences, max_files=max_files,
                                            use_phrases=use_phrases)
    else:
        text8_file = download_text8(data_dir)
        processed_dir = preprocess_text8(text8_file, "./data/text8_processed",
                                        use_phrases=use_phrases)
    
    # 3. Train Skip-gram
    print_section_header("STEP 3: TRAINING SKIP-GRAM MODEL")
    skipgram_params = {
        "epochs": 10,
        "embed_dim": 100,
        "min_occurs": 5,
        "c": 5,
        "k": 0 if use_hs_only else 5,
        "t": 1e-5,
        "vocab_freq_exponent": 0.75,
        "lr_max": 0.025,
        "lr_min": 0.0001,
        "cuda_threads_per_block": 32,
        "hs": 1 if use_hs else 0
    }
    
    print("Skip-gram parameters:")
    for key, value in skipgram_params.items():
        print(f"  {key}: {value}")
    
    if skipgram_params["hs"] == 1 and skipgram_params["k"] > 0:
        print("  ⚠️  Note: Learning rate will be automatically reduced by 50% to prevent gradient explosion")
    
    train_skipgram(processed_dir, "./output/vectors_skipgram", **skipgram_params)
    
    # 4. Train CBOW
    print_section_header("STEP 4: TRAINING CBOW MODEL")
    cbow_params = skipgram_params.copy()
    cbow_params["lr_max"] = 0.05
    cbow_params["lr_min"] = 0.0001
    
    print("CBOW parameters:")
    for key, value in cbow_params.items():
        print(f"  {key}: {value}")
    
    if cbow_params["hs"] == 1 and cbow_params["k"] > 0:
        print("  ⚠️  Note: Learning rate will be automatically reduced by 50% to prevent gradient explosion")
    
    train_cbow(processed_dir, "./output/vectors_cbow", **cbow_params)
    
    # 5. Evaluate Skip-gram
    print_section_header("STEP 5: EVALUATING SKIP-GRAM MODEL")
    sg_acc, sg_details = word_analogy_test("./output/vectors_skipgram")
    sg_sim = similarity_test("./output/vectors_skipgram")
    save_evaluation_results({
        "accuracy": sg_acc,
        "details": sg_details,
        "similarity_test": sg_sim
    }, "./output/skipgram_eval.json")
    
    # 6. Evaluate CBOW
    print_section_header("STEP 6: EVALUATING CBOW MODEL")
    cbow_acc, cbow_details = word_analogy_test("./output/vectors_cbow")
    cbow_sim = similarity_test("./output/vectors_cbow")
    save_evaluation_results({
        "accuracy": cbow_acc,
        "details": cbow_details,
        "similarity_test": cbow_sim
    }, "./output/cbow_eval.json")
    
    # Check if should stop after evaluation (only if not using Gensim)
    if stop_after_eval and not use_gensim:
        print_section_header("STOPPING AFTER STEP 6")
        print("Skipping visualization and comparison steps as requested.")
        print(f"\n✅ Training and evaluation completed!")
        print(f"📁 Check the ./output/ directory for:")
        print(f"  - Vectors: ./output/vectors_skipgram, ./output/vectors_cbow")
        print(f"  - Evaluation results: ./output/skipgram_eval.json, ./output/cbow_eval.json")
        print(f"  - Statistics: ./output/vectors_skipgram_stats.json, ./output/vectors_cbow_stats.json")
        print(f"\n📊 Evaluation Results:")
        print(f"  Skip-gram accuracy: {sg_acc:.4f} ({sg_acc*100:.2f}%)")
        print(f"  CBOW accuracy: {cbow_acc:.4f} ({cbow_acc*100:.2f}%)")
        return
    
    # 7. Train Gensim Models (if enabled)
    gensim_sg_path = None
    gensim_cbow_path = None
    gensim_sg_time = None
    gensim_cbow_time = None
    gensim_sg_acc = None
    gensim_sg_details = None
    gensim_cbow_acc = None
    gensim_cbow_details = None
    
    if use_gensim:
        print_section_header("STEP 7: TRAINING GENSIM MODELS")
        gensim_sg_path, gensim_cbow_path, gensim_sg_time, gensim_cbow_time = train_gensim_models(
            processed_dir,
            output_dir="./output/gensim",
            epochs=skipgram_params["epochs"],
            embed_dim=skipgram_params["embed_dim"],
            min_count=skipgram_params["min_occurs"],
            window=skipgram_params["c"],
            negative=skipgram_params["k"],
            hs=skipgram_params["hs"],
            alpha=skipgram_params["lr_max"],
            min_alpha=skipgram_params["lr_min"]
        )
        
        # 8. Evaluate Gensim Models
        print_section_header("STEP 8: EVALUATING GENSIM MODELS")
        gensim_sg_acc, gensim_sg_details, gensim_cbow_acc, gensim_cbow_details = evaluate_gensim_models(
            gensim_sg_path,
            gensim_cbow_path,
            output_dir="./output/gensim"
        )
        
        # 9. Compare with Gensim
        print_section_header("STEP 9: COMPARING WITH GENSIM")
        # Load custom model statistics
        custom_sg_time = None
        custom_cbow_time = None
        try:
            import json
            with open("./output/vectors_skipgram_stats.json", "r") as f:
                custom_sg_stats = json.load(f)
                custom_sg_time = custom_sg_stats.get("epoch_time_total_seconds", None)
            with open("./output/vectors_cbow_stats.json", "r") as f:
                custom_cbow_stats = json.load(f)
                custom_cbow_time = custom_cbow_stats.get("epoch_time_total_seconds", None)
        except FileNotFoundError:
            pass
        
        gensim_comparison = compare_with_gensim(
            "./output/vectors_skipgram",
            "./output/vectors_cbow",
            gensim_sg_path,
            gensim_cbow_path,
            gensim_sg_time,
            gensim_cbow_time,
            gensim_sg_acc,
            gensim_sg_details,
            gensim_cbow_acc,
            gensim_cbow_details,
            custom_sg_time,
            custom_cbow_time
        )
        
        if stop_after_eval:
            print_section_header("STOPPING AFTER GENSIM COMPARISON")
            print("Skipping visualization steps as requested.")
            print(f"\n✅ Training, evaluation and comparison completed!")
            print(f"📁 Check the ./output/ directory for:")
            print(f"  - Custom vectors: ./output/vectors_skipgram, ./output/vectors_cbow")
            print(f"  - Gensim vectors: ./output/gensim/vectors_skipgram_gensim, ./output/gensim/vectors_cbow_gensim")
            print(f"  - Evaluation results: ./output/skipgram_eval.json, ./output/cbow_eval.json")
            print(f"  - Gensim evaluation: ./output/gensim/skipgram_eval.json, ./output/gensim/cbow_eval.json")
            print(f"  - Comparison: ./output/gensim_comparison.json")
            print(f"\n📊 Evaluation Results:")
            print(f"  Custom Skip-gram: {sg_acc:.4f} ({sg_acc*100:.2f}%)")
            print(f"  Custom CBOW: {cbow_acc:.4f} ({cbow_acc*100:.2f}%)")
            print(f"  Gensim Skip-gram: {gensim_sg_acc:.4f} ({gensim_sg_acc*100:.2f}%)")
            print(f"  Gensim CBOW: {gensim_cbow_acc:.4f} ({gensim_cbow_acc*100:.2f}%)")
            return
    
    # 10. Model Comparison (Custom Skip-gram vs CBOW)
    step_num = 7 if not use_gensim else 10
    print_section_header(f"STEP {step_num}: COMPARING CUSTOM MODELS (Skip-gram vs CBOW)")
    # Pass pre-computed accuracy values to avoid re-evaluating
    comparison = compare_models("./output/vectors_skipgram", "./output/vectors_cbow",
                                sg_acc=sg_acc, sg_details=sg_details,
                                cbow_acc=cbow_acc, cbow_details=cbow_details)
    
    # 11. Visualizations
    step_num = 8 if not use_gensim else 11
    print_section_header(f"STEP {step_num}: CREATING VISUALIZATIONS")
    
    # t-SNE plots
    print("Creating t-SNE visualizations...")
    plot_tsne("./output/vectors_skipgram", "./output/skipgram_tsne.png")
    plot_tsne("./output/vectors_cbow", "./output/cbow_tsne.png")
    
    # Similarity heatmaps
    print("Creating similarity heatmaps...")
    test_words = ["king", "queen", "man", "woman", "computer", "science", "university", "student"]
    plot_similarity_heatmap("./output/vectors_skipgram", test_words, "./output/skipgram_heatmap.png")
    plot_similarity_heatmap("./output/vectors_cbow", test_words, "./output/cbow_heatmap.png")
    
    # Training comparison
    print("Creating training comparison plots...")
    plot_training_comparison(
        "./output/vectors_skipgram_stats.json",
        "./output/vectors_cbow_stats.json",
        "./output/training_comparison.png"
    )
    
    # Accuracy comparison
    plot_accuracy_comparison(sg_acc, cbow_acc, "./output/accuracy_comparison.png")
    
    # 12. Load statistics for summary
    sg_stats = {}
    cbow_stats = {}
    
    try:
        import json
        with open("./output/vectors_skipgram_stats.json", "r") as f:
            sg_stats = json.load(f)
        with open("./output/vectors_cbow_stats.json", "r") as f:
            cbow_stats = json.load(f)
    except FileNotFoundError:
        print("Warning: Could not load statistics files")
    
    # 13. Final Summary
    print_summary(sg_acc, cbow_acc, sg_stats, cbow_stats)
    
    print(f"\n🎉 Word2Vec training and evaluation completed successfully!")
    print(f"📁 Check the ./output/ directory for all results and visualizations.")
    
    print(f"\nDataset used: {dataset_name}")


if __name__ == "__main__":
    # Trong notebook, không dùng sys.exit() vì sẽ gây SystemExit exception
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        # Không dùng sys.exit() trong notebook
        raise  # Re-raise để notebook hiển thị error
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Không dùng sys.exit() trong notebook - re-raise exception
        raise  # Re-raise để notebook hiển thị error

