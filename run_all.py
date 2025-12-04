#!/usr/bin/env python3
# Copyright 2024 Word2Vec Implementation
# Main script to run complete Word2Vec pipeline

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def get_user_choice(prompt: str, options: list, default: int = 0) -> int:
    """Get user choice from a menu."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        marker = " [default]" if i == default + 1 else ""
        print(f"  {i}. {option}{marker}")
    
    while True:
        try:
            choice = input(f"\nSelect option (1-{len(options)}, default={default+1}): ").strip()
            if not choice:
                return default
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return choice_idx
            else:
                print(f"⚠️  Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("⚠️  Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by user.")
            sys.exit(0)


def interactive_menu():
    """Interactive menu for selecting options."""
    print("="*60)
    print("  Word2Vec CBOW & Skip-gram Implementation")
    print("  Based on Mikolov's original paper")
    print("  Using Numba CUDA for GPU acceleration")
    print("="*60)
    
    # STEP 1: Dataset selection
    dataset_options = [
        "Text8 (17M words, ~100MB) - Fast, good for testing",
        "Wikipedia (~500MB) - High quality articles",
        "WMT14 News (850M words, ~3.2GB) - Large news corpus"
    ]
    dataset_choice = get_user_choice("📁 STEP 1: Select Dataset", dataset_options, default=0)
    
    dataset_name = ["Text8", "Wikipedia", "WMT14 News"][dataset_choice]
    use_wikipedia = (dataset_choice == 1)
    use_wmt14 = (dataset_choice == 2)
    
    # STEP 2: Training method selection
    training_options = [
        "Negative Sampling ONLY (HS=0, k=5) - Default, fast",
        "Hierarchical Softmax ONLY (HS=1, k=0) - Accurate but slower",
        "HS + Negative Sampling (HS=1, k=5) - Best accuracy"
    ]
    training_choice = get_user_choice("🎯 STEP 2: Select Training Method", training_options, default=0)
    
    use_hs_only = (training_choice == 1)
    use_hs = (training_choice == 1 or training_choice == 2)
    
    # STEP 3: Size selection (only for WMT14)
    max_sentences = None
    max_files = None
    if use_wmt14:
        size_options = [
            "Full dataset (all sentences) - ~850M words",
            "Tiny (100K sentences, ~100MB) - Quick test",
            "Small (1M sentences, ~1GB) - Good balance",
            "Medium (5M sentences, ~5GB) - Larger dataset",
            "Custom size (enter number of sentences)"
        ]
        size_choice = get_user_choice("📦 STEP 3: Select Dataset Size (WMT14 only)", size_options, default=0)
        
        if size_choice == 0:
            max_sentences = None
            max_files = None
        elif size_choice == 1:
            max_sentences = 100000
            max_files = 1
        elif size_choice == 2:
            max_sentences = 1000000
            max_files = 10
        elif size_choice == 3:
            max_sentences = 5000000
            max_files = 50
        else:  # Custom
            while True:
                try:
                    custom_input = input("Enter number of sentences: ").strip()
                    max_sentences = int(custom_input)
                    max_files = max(1, max_sentences // 100000)  # Estimate files
                    break
                except ValueError:
                    print("⚠️  Please enter a valid number")
    else:
        size_choice = None
    
    # STEP 4: Phrase detection
    phrase_options = [
        "No phrase detection - Default, faster preprocessing",
        "Enable phrase detection - Combine frequent bigrams (e.g., 'new york' -> 'new_york')"
    ]
    phrase_choice = get_user_choice("🔗 STEP 4: Phrase Detection", phrase_options, default=0)
    use_phrases = (phrase_choice == 1)
    
    # STEP 5: Gensim training
    gensim_options = [
        "Skip Gensim training - Default",
        "Train and evaluate Gensim models - Compare with custom implementation"
    ]
    gensim_choice = get_user_choice("📚 STEP 5: Gensim Training", gensim_options, default=0)
    use_gensim = (gensim_choice == 1)
    
    # STEP 6: Stop after evaluation
    stop_options = [
        "Full pipeline (training + evaluation + visualization) - Default",
        "Stop after evaluation (skip visualization) - Faster"
    ]
    stop_choice = get_user_choice("⏹️  STEP 6: Stop After Evaluation", stop_options, default=0)
    stop_after_eval = (stop_choice == 1)
    
    # Summary
    print("\n" + "="*60)
    print("  CONFIGURATION SUMMARY")
    print("="*60)
    print(f"  📁 Dataset: {dataset_name}")
    if use_wmt14 and max_sentences:
        print(f"  📦 Size: {max_sentences:,} sentences (~{max_files} files)")
    elif use_wmt14:
        print(f"  📦 Size: Full dataset")
    print(f"  🎯 Training: {training_options[training_choice]}")
    print(f"  🔗 Phrases: {'Enabled' if use_phrases else 'Disabled'}")
    print(f"  📚 Gensim: {'Enabled' if use_gensim else 'Disabled'}")
    print(f"  ⏹️  Stop after eval: {'Yes' if stop_after_eval else 'No'}")
    print("="*60)
    
    confirm = input("\nProceed with this configuration? (y/n, default=y): ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("⚠️  Cancelled by user.")
        sys.exit(0)
    
    return {
        'use_wikipedia': use_wikipedia,
        'use_wmt14': use_wmt14,
        'dataset_name': dataset_name,
        'use_hs_only': use_hs_only,
        'use_hs': use_hs,
        'max_sentences': max_sentences,
        'max_files': max_files,
        'use_phrases': use_phrases,
        'use_gensim': use_gensim,
        'stop_after_eval': stop_after_eval
    }


def main():
    """Main pipeline execution."""
    # Check if running in non-interactive mode (command line arguments)
    if len(sys.argv) > 1 and sys.argv[1] not in ['--help', '-h']:
        # Legacy command-line mode
        print("⚠️  Command-line arguments detected. Using legacy mode.")
        print("💡 Tip: Run without arguments for interactive menu.\n")
        
        # Parse command line arguments for dataset selection
        use_wmt14 = "--wmt14" in sys.argv or "--news" in sys.argv
        use_wikipedia = "--wikipedia" in sys.argv or "--wiki" in sys.argv
        
        if use_wikipedia:
            dataset_name = "Wikipedia"
        elif use_wmt14:
            dataset_name = "WMT14 News"
        else:
            dataset_name = "Text8"
        
        # Parse size reduction options
        max_sentences = None
        max_files = None
        
        if "--small" in sys.argv:
            max_sentences = 1000000
            max_files = 10
        elif "--medium" in sys.argv:
            max_sentences = 5000000
            max_files = 50
        elif "--tiny" in sys.argv:
            max_sentences = 100000
            max_files = 1
        elif "--custom" in sys.argv:
            try:
                custom_idx = sys.argv.index("--custom")
                if custom_idx + 1 < len(sys.argv):
                    max_sentences = int(sys.argv[custom_idx + 1])
            except (ValueError, IndexError):
                max_sentences = None
        
        # Parse training method options
        use_hs_only = "--hs-only" in sys.argv
        use_hs = use_hs_only or "--hs-ns" in sys.argv or "--hs+ns" in sys.argv or "--hs" in sys.argv
        
        # Parse phrase detection option
        use_phrases = "--phrases" in sys.argv or "--phrase" in sys.argv
        
        # Parse Gensim option
        use_gensim = "--gensim" in sys.argv
        
        # Parse stop option
        stop_after_eval = "--stop-after-eval" in sys.argv
    else:
        # Interactive menu mode (default)
        config = interactive_menu()
        use_wikipedia = config['use_wikipedia']
        use_wmt14 = config['use_wmt14']
        dataset_name = config['dataset_name']
        use_hs_only = config['use_hs_only']
        use_hs = config['use_hs']
        max_sentences = config['max_sentences']
        max_files = config['max_files']
        use_phrases = config['use_phrases']
        use_gensim = config['use_gensim']
        stop_after_eval = config['stop_after_eval']
    
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
        print("Warning: No GPU detected. Training will be slow on CPU.")
        input("Press Enter to continue anyway, or Ctrl+C to exit...")
    
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
    epochs_value = 1  # Set epochs here for consistency
    skipgram_params = {
        "epochs": epochs_value,
        "embed_dim": 600,
        "min_occurs": 5,
        "c": 5,
        "k": 0 if use_hs_only else 5,
        "t": 1e-5,
        "vocab_freq_exponent": 0.75,
        "lr_max": 0.025,
        # For 1 epoch with large dataset, keep learning rate high (as in paper)
        # With 1 epoch, we want to use a constant high learning rate
        "lr_min": 0.025 if epochs_value == 1 else 0.0001,
        "cuda_threads_per_block": 512,  # Optimized for A100 GPU (was 32, too low)
        "hs": 1 if use_hs else 0
    }
    
    if epochs_value == 1:
        print("  ℹ️  Using 1 epoch: Learning rate will be kept constant at 0.025 (as per paper)")
    
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


    # 6. Evaluate CBOW
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
        print(f"  Skip-gram semantic:  {sg_sem:.4f} ({sg_sem*100:.2f}%)")
        print(f"  Skip-gram syntactic: {sg_syn:.4f} ({sg_syn*100:.2f}%)")
        print(f"  Skip-gram total:     {sg_total:.4f} ({sg_total*100:.2f}%)")

        print(f"  CBOW semantic:        {cbow_sem:.4f} ({cbow_sem*100:.2f}%)")
        print(f"  CBOW syntactic:       {cbow_syn:.4f} ({cbow_syn*100:.2f}%)")
        print(f"  CBOW total:           {cbow_total:.4f} ({cbow_total*100:.2f}%)")
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
    print("\n💡 TIP: Run 'python run_all.py' (without arguments) for interactive menu!")
    print("\nLegacy command-line usage (still supported):")
    print("  python run_all.py --wikipedia --hs-only")
    print("  python run_all.py --wmt14 --small --hs-ns --phrases")
    print("  python run_all.py --hs --stop-after-eval")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
