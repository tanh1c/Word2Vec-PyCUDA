# Copyright 2024 Word2Vec Implementation
# Evaluation utilities for word2vec models

import json
import os
import time
import re
from typing import List, Tuple, Dict, Any
import requests
from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath


def download_questions_words(output_path: str = "./data/questions-words.txt") -> str:
    """Download questions-words.txt for word analogy test."""
    if os.path.isfile(output_path):
        print(f"Questions-words.txt already exists at: {output_path}")
        return output_path
    
    url = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
    print(f"Downloading questions-words.txt from {url}...")
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    print(f"Questions-words.txt downloaded to: {output_path}")
    return output_path


def word_analogy_test(vectors_path: str, questions_path: str = None) -> Tuple[dict, List[Dict]]:
    """
    Run word analogy test on trained vectors.
    Returns dictionary containing: 
        - semantic_accuracy
        - syntactic_accuracy
        - total_accuracy
    And details_by_category (list).
    """

    if questions_path is None:
        questions_path = download_questions_words()

    print(f"Loading vectors from: {vectors_path}")
    start = time.time()
    vecs = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
    print(f"Vectors loaded in {time.time() - start:.2f}s")

    print(f"Running word analogy test with: {questions_path}")
    eval_start = time.time()
    overall_acc, details = vecs.evaluate_word_analogies(questions_path, case_insensitive=True)
    print(f"Word analogy test completed in {time.time() - eval_start:.2f}s")

    # =====================================================
    #   NEW: Thống kê semantic & syntactic
    # =====================================================
    semantic_correct = 0
    semantic_total = 0

    syntactic_correct = 0
    syntactic_total = 0

    for cat in details:
        correct = len(cat["correct"])
        total = correct + len(cat["incorrect"])

        # Phân loại semantic vs syntactic dựa trên categories trong questions-words.txt
        # Semantic (5 categories): capital-common-countries, capital-world, currency, city-in-state, family
        # Syntactic (9 categories): gram1-9 (adjective-to-adverb, opposite, comparative, superlative, etc.)
        section = cat["section"].lower()

        # Semantic categories keywords (5 categories từ questions-words.txt)
        # 1. capital-common-countries, capital-world → "capital"
        # 2. currency → "currency"
        # 3. city-in-state → "city-in-state"
        # 4. family → "family"
        semantic_keywords = ["capital", "currency", "family", "city-in-state"]
        is_semantic = any(keyword in section for keyword in semantic_keywords)
        
        if is_semantic:
            semantic_correct += correct
            semantic_total += total
        else:
            # Syntactic categories (tất cả các categories còn lại, thường bắt đầu bằng "gram")
            syntactic_correct += correct
            syntactic_total += total

    semantic_acc = semantic_correct / semantic_total if semantic_total > 0 else 0
    syntactic_acc = syntactic_correct / syntactic_total if syntactic_total > 0 else 0

    # Tổng toàn bộ
    total_acc = (
        (semantic_correct + syntactic_correct) /
        (semantic_total + syntactic_total)
        if (semantic_total + syntactic_total) > 0 else 0
    )
    return (
        {
            "semantic_accuracy": semantic_acc,
            "syntactic_accuracy": syntactic_acc,
            "total_accuracy": total_acc
        },
        details
    )

def similarity_test(vectors_path: str, test_words: List[str] = None) -> Dict[str, Any]:
    """
    Test word similarity and find most similar words.
    """
    if test_words is None:
        test_words = ["king", "queen", "man", "woman", "computer", "science", "university", "student"]
    
    print(f"Loading vectors for similarity test: {vectors_path}")
    vecs = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
    
    results = {}
    
    print("\nMost similar words:")
    for word in test_words:
        if word in vecs:
            similar = vecs.most_similar(word, topn=5)
            results[word] = similar
            print(f"\n{word}:")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")
        else:
            print(f"Word '{word}' not found in vocabulary")
            results[word] = []
    
    # Test some word pairs for similarity
    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("computer", "science"),
        ("university", "student"),
        ("good", "bad"),
        ("big", "small")
    ]
    
    print("\nWord pair similarities:")
    pair_similarities = {}
    for word1, word2 in word_pairs:
        if word1 in vecs and word2 in vecs:
            similarity = vecs.similarity(word1, word2)
            pair_similarities[f"{word1}-{word2}"] = similarity
            print(f"  {word1} - {word2}: {similarity:.4f}")
        else:
            print(f"  {word1} - {word2}: One or both words not found")
            pair_similarities[f"{word1}-{word2}"] = None
    
    results["pair_similarities"] = pair_similarities
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    import numpy as np
    
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert numpy types to Python native types
    results_converted = convert_numpy_types(results)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to: {output_path}")


def compare_models(skipgram_path: str, cbow_path: str, output_path: str = "./output/model_comparison.json",
                   sg_acc: float = None, sg_details: List[Dict] = None,
                   cbow_acc: float = None, cbow_details: List[Dict] = None):
    """
    Compare Skip-gram and CBOW models.
    
    Args:
        skipgram_path: Path to Skip-gram vectors
        cbow_path: Path to CBOW vectors
        output_path: Output path for comparison JSON
        sg_acc: Pre-computed Skip-gram accuracy (optional, will compute if None)
        sg_details: Pre-computed Skip-gram evaluation details (optional)
        cbow_acc: Pre-computed CBOW accuracy (optional, will compute if None)
        cbow_details: Pre-computed CBOW evaluation details (optional)
    """
    print("Comparing Skip-gram vs CBOW models...")
    
    # Evaluate both models only if not provided
    if sg_acc is None or sg_details is None:
        print("Evaluating Skip-gram model...")
        sg_result, sg_details = word_analogy_test(skipgram_path)
        sg_acc = sg_result["total_accuracy"]  # Extract total accuracy from dict
    else:
        print("Using pre-computed Skip-gram accuracy...")
    
    if cbow_acc is None or cbow_details is None:
        print("Evaluating CBOW model...")
        cbow_result, cbow_details = word_analogy_test(cbow_path)
        cbow_acc = cbow_result["total_accuracy"]  # Extract total accuracy from dict
    else:
        print("Using pre-computed CBOW accuracy...")
    
    # Load statistics
    sg_stats_path = skipgram_path + "_stats.json"
    cbow_stats_path = cbow_path + "_stats.json"
    
    sg_stats = {}
    cbow_stats = {}
    
    if os.path.isfile(sg_stats_path):
        with open(sg_stats_path, "r") as f:
            sg_stats = json.load(f)
    
    if os.path.isfile(cbow_stats_path):
        with open(cbow_stats_path, "r") as f:
            cbow_stats = json.load(f)
    
    comparison = {
        "models": {
            "skipgram": {
                "accuracy": sg_acc,
                "details": sg_details,
                "stats": sg_stats
            },
            "cbow": {
                "accuracy": cbow_acc,
                "details": cbow_details,
                "stats": cbow_stats
            }
        },
        "summary": {
            "skipgram_accuracy": sg_acc,
            "cbow_accuracy": cbow_acc,
            "accuracy_difference": sg_acc - cbow_acc,
            "skipgram_training_time": sg_stats.get("epoch_time_total_seconds", 0),
            "cbow_training_time": cbow_stats.get("epoch_time_total_seconds", 0),
            "time_difference": sg_stats.get("epoch_time_total_seconds", 0) - cbow_stats.get("epoch_time_total_seconds", 0)
        }
    }
    
    save_evaluation_results(comparison, output_path)
    
    print(f"\nModel Comparison Summary:")
    print(f"Skip-gram accuracy: {sg_acc:.4f} ({sg_acc*100:.2f}%)")
    print(f"CBOW accuracy: {cbow_acc:.4f} ({cbow_acc*100:.2f}%)")
    print(f"Difference: {sg_acc - cbow_acc:.4f} ({(sg_acc - cbow_acc)*100:.2f}%)")
    
    if sg_stats and cbow_stats:
        sg_time = sg_stats.get("epoch_time_total_seconds", 0)
        cbow_time = cbow_stats.get("epoch_time_total_seconds", 0)
        print(f"Skip-gram training time: {sg_time:.2f}s")
        print(f"CBOW training time: {cbow_time:.2f}s")
        print(f"Time difference: {sg_time - cbow_time:.2f}s")
    
    return comparison


def train_gensim_models(data_path: str, output_dir: str = "./output/gensim",
                        epochs: int = 10, embed_dim: int = 100, min_count: int = 5,
                        window: int = 5, negative: int = 5, hs: int = 0,
                        alpha: float = 0.025, min_alpha: float = 0.0001,
                        sample: float = 1e-5, workers: int = 4) -> Tuple[str, str, float, float]:
    """
    Train Skip-gram and CBOW models using Gensim library.
    
    Args:
        data_path: Path to preprocessed data directory
        output_dir: Output directory for Gensim models
        epochs: Number of training epochs
        embed_dim: Embedding dimension
        min_count: Minimum word count
        window: Window size
        negative: Number of negative samples (if hs=0)
        hs: Hierarchical Softmax (1=HS, 0=NS)
        alpha: Initial learning rate
        min_alpha: Minimum learning rate
        sample: Subsampling threshold
        workers: Number of worker threads
    
    Returns:
        Tuple of (skipgram_path, cbow_path, skipgram_time, cbow_time)
    """
    import glob
    from data_handler import get_data_file_names
    
    os.makedirs(output_dir, exist_ok=True)
    
    skipgram_path = os.path.join(output_dir, "vectors_skipgram_gensim")
    cbow_path = os.path.join(output_dir, "vectors_cbow_gensim")
    
    # Read all sentences from data files
    print("Reading data files...")
    data_files = get_data_file_names(data_path, seed=12345)
    sentences = []
    
    start = time.time()
    for filename in data_files:
        filepath = os.path.join(data_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words = [word for word in re.split(r"[ .]+", line) if word]
                    if len(words) >= 2:
                        sentences.append(words)
    
    print(f"Read {len(sentences):,} sentences in {time.time() - start:.2f}s")
    
    # Common parameters (Gensim 4.0+ uses vector_size instead of size)
    common_params = {
        "vector_size": embed_dim,  # Changed from "size" in Gensim 4.0+
        "window": window,
        "min_count": min_count,
        "hs": hs,
        "negative": 0 if hs == 1 else negative,
        "alpha": alpha,
        "min_alpha": min_alpha,
        "sample": sample,
        "workers": workers,
        "sg": 0  # Will be set per model
    }
    
    # Train Skip-gram model
    print("\n" + "="*60)
    print("  TRAINING GENSIM SKIP-GRAM MODEL")
    print("="*60)
    skipgram_params = common_params.copy()
    skipgram_params["sg"] = 1  # Skip-gram
    
    print(f"Parameters: {skipgram_params}")
    print("Building vocabulary...")
    sg_model = Word2Vec(**skipgram_params)
    sg_model.build_vocab(sentences)
    vocab_size = len(sg_model.wv.vocab) if hasattr(sg_model.wv, 'vocab') else len(sg_model.wv.key_to_index)
    print(f"Vocabulary size: {vocab_size:,}")
    
    print(f"Training Skip-gram model for {epochs} epochs...")
    sg_start = time.time()
    sg_model.train(sentences, total_examples=sg_model.corpus_count, epochs=epochs)
    sg_time = time.time() - sg_start
    
    print(f"Skip-gram training completed in {sg_time:.2f}s")
    print(f"Saving Skip-gram vectors to: {skipgram_path}")
    sg_model.wv.save_word2vec_format(skipgram_path, binary=False)
    
    # Train CBOW model
    print("\n" + "="*60)
    print("  TRAINING GENSIM CBOW MODEL")
    print("="*60)
    cbow_params = common_params.copy()
    cbow_params["sg"] = 0  # CBOW
    cbow_params["alpha"] = alpha * 2  # CBOW typically uses higher LR
    
    print(f"Parameters: {cbow_params}")
    print("Building vocabulary...")
    cbow_model = Word2Vec(**cbow_params)
    cbow_model.build_vocab(sentences)
    vocab_size = len(cbow_model.wv.vocab) if hasattr(cbow_model.wv, 'vocab') else len(cbow_model.wv.key_to_index)
    print(f"Vocabulary size: {vocab_size:,}")
    
    print(f"Training CBOW model for {epochs} epochs...")
    cbow_start = time.time()
    cbow_model.train(sentences, total_examples=cbow_model.corpus_count, epochs=epochs)
    cbow_time = time.time() - cbow_start
    
    print(f"CBOW training completed in {cbow_time:.2f}s")
    print(f"Saving CBOW vectors to: {cbow_path}")
    cbow_model.wv.save_word2vec_format(cbow_path, binary=False)
    
    return skipgram_path, cbow_path, sg_time, cbow_time


def evaluate_gensim_models(skipgram_path: str, cbow_path: str, 
                           output_dir: str = "./output/gensim",
                           questions_path: str = None) -> Tuple[float, List[Dict], float, List[Dict]]:
    """
    Evaluate Gensim-trained Skip-gram and CBOW models.
    
    Args:
        skipgram_path: Path to Skip-gram vectors
        cbow_path: Path to CBOW vectors
        output_dir: Output directory for evaluation results
        questions_path: Path to questions-words.txt file
    
    Returns:
        Tuple of (skipgram_acc, skipgram_details, cbow_acc, cbow_details)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if questions_path is None:
        questions_path = download_questions_words()
    
    # Evaluate Skip-gram
    print("\n" + "="*60)
    print("  EVALUATING GENSIM SKIP-GRAM MODEL")
    print("="*60)
    sg_acc, sg_details = word_analogy_test(skipgram_path, questions_path)
    sg_sim = similarity_test(skipgram_path)
    
    sg_eval_path = os.path.join(output_dir, "skipgram_eval.json")
    save_evaluation_results({
        "accuracy": sg_acc,
        "details": sg_details,
        "similarity_test": sg_sim
    }, sg_eval_path)
    
    # Evaluate CBOW
    print("\n" + "="*60)
    print("  EVALUATING GENSIM CBOW MODEL")
    print("="*60)
    cbow_acc, cbow_details = word_analogy_test(cbow_path, questions_path)
    cbow_sim = similarity_test(cbow_path)
    
    cbow_eval_path = os.path.join(output_dir, "cbow_eval.json")
    save_evaluation_results({
        "accuracy": cbow_acc,
        "details": cbow_details,
        "similarity_test": cbow_sim
    }, cbow_eval_path)
    
    return sg_acc, sg_details, cbow_acc, cbow_details


def compare_with_gensim(custom_sg_path: str, custom_cbow_path: str,
                        gensim_sg_path: str, gensim_cbow_path: str,
                        gensim_sg_time: float, gensim_cbow_time: float,
                        gensim_sg_acc: float, gensim_sg_details: List[Dict],
                        gensim_cbow_acc: float, gensim_cbow_details: List[Dict],
                        custom_sg_time: float = None, custom_cbow_time: float = None,
                        output_path: str = "./output/gensim_comparison.json") -> Dict[str, Any]:
    """
    Compare custom implementation with Gensim models.
    
    Args:
        custom_sg_path: Path to custom Skip-gram vectors
        custom_cbow_path: Path to custom CBOW vectors
        gensim_sg_path: Path to Gensim Skip-gram vectors
        gensim_cbow_path: Path to Gensim CBOW vectors
        gensim_sg_time: Gensim Skip-gram training time
        gensim_cbow_time: Gensim CBOW training time
        gensim_sg_acc: Gensim Skip-gram accuracy
        gensim_sg_details: Gensim Skip-gram evaluation details
        gensim_cbow_acc: Gensim CBOW accuracy
        gensim_cbow_details: Gensim CBOW evaluation details
        custom_sg_time: Custom Skip-gram training time (optional)
        custom_cbow_time: Custom CBOW training time (optional)
        output_path: Output path for comparison JSON
    
    Returns:
        Comparison dictionary
    """
    print("\n" + "="*60)
    print("  COMPARING CUSTOM vs GENSIM MODELS")
    print("="*60)
    
    # Evaluate custom models
    print("Evaluating custom models...")
    custom_sg_acc, custom_sg_details = word_analogy_test(custom_sg_path)
    custom_cbow_acc, custom_cbow_details = word_analogy_test(custom_cbow_path)
    
    # Load custom model statistics if available
    custom_sg_stats = {}
    custom_cbow_stats = {}
    
    custom_sg_stats_path = custom_sg_path + "_stats.json"
    custom_cbow_stats_path = custom_cbow_path + "_stats.json"
    
    if os.path.isfile(custom_sg_stats_path):
        with open(custom_sg_stats_path, "r") as f:
            custom_sg_stats = json.load(f)
            if custom_sg_time is None:
                custom_sg_time = custom_sg_stats.get("epoch_time_total_seconds", 0)
    
    if os.path.isfile(custom_cbow_stats_path):
        with open(custom_cbow_stats_path, "r") as f:
            custom_cbow_stats = json.load(f)
            if custom_cbow_time is None:
                custom_cbow_time = custom_cbow_stats.get("epoch_time_total_seconds", 0)
    
    # Create comparison
    comparison = {
        "skipgram": {
            "custom": {
                "accuracy": custom_sg_acc,
                "training_time": custom_sg_time,
                "details": custom_sg_details
            },
            "gensim": {
                "accuracy": gensim_sg_acc,
                "training_time": gensim_sg_time,
                "details": gensim_sg_details
            },
            "accuracy_difference": custom_sg_acc - gensim_sg_acc,
            "time_difference": (custom_sg_time or 0) - gensim_sg_time,
            "speedup": gensim_sg_time / (custom_sg_time or 1) if custom_sg_time else None
        },
        "cbow": {
            "custom": {
                "accuracy": custom_cbow_acc,
                "training_time": custom_cbow_time,
                "details": custom_cbow_details
            },
            "gensim": {
                "accuracy": gensim_cbow_acc,
                "training_time": gensim_cbow_time,
                "details": gensim_cbow_details
            },
            "accuracy_difference": custom_cbow_acc - gensim_cbow_acc,
            "time_difference": (custom_cbow_time or 0) - gensim_cbow_time,
            "speedup": gensim_cbow_time / (custom_cbow_time or 1) if custom_cbow_time else None
        },
        "summary": {
            "custom_skipgram_accuracy": custom_sg_acc,
            "gensim_skipgram_accuracy": gensim_sg_acc,
            "custom_cbow_accuracy": custom_cbow_acc,
            "gensim_cbow_accuracy": gensim_cbow_acc,
            "skipgram_accuracy_diff": custom_sg_acc - gensim_sg_acc,
            "cbow_accuracy_diff": custom_cbow_acc - gensim_cbow_acc
        }
    }
    
    save_evaluation_results(comparison, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print("\nSkip-gram:")
    print(f"  Custom:  {custom_sg_acc:.4f} ({custom_sg_acc*100:.2f}%) - {custom_sg_time:.2f}s" if custom_sg_time else f"  Custom:  {custom_sg_acc:.4f} ({custom_sg_acc*100:.2f}%)")
    print(f"  Gensim:  {gensim_sg_acc:.4f} ({gensim_sg_acc*100:.2f}%) - {gensim_sg_time:.2f}s")
    print(f"  Diff:    {custom_sg_acc - gensim_sg_acc:+.4f} ({(custom_sg_acc - gensim_sg_acc)*100:+.2f}%)")
    if custom_sg_time:
        print(f"  Speedup: {gensim_sg_time / custom_sg_time:.2f}x {'(Gensim faster)' if gensim_sg_time < custom_sg_time else '(Custom faster)'}")
    
    print("\nCBOW:")
    print(f"  Custom:  {custom_cbow_acc:.4f} ({custom_cbow_acc*100:.2f}%) - {custom_cbow_time:.2f}s" if custom_cbow_time else f"  Custom:  {custom_cbow_acc:.4f} ({custom_cbow_acc*100:.2f}%)")
    print(f"  Gensim:  {gensim_cbow_acc:.4f} ({gensim_cbow_acc*100:.2f}%) - {gensim_cbow_time:.2f}s")
    print(f"  Diff:    {custom_cbow_acc - gensim_cbow_acc:+.4f} ({(custom_cbow_acc - gensim_cbow_acc)*100:+.2f}%)")
    if custom_cbow_time:
        print(f"  Speedup: {gensim_cbow_time / custom_cbow_time:.2f}x {'(Gensim faster)' if gensim_cbow_time < custom_cbow_time else '(Custom faster)'}")
    
    return comparison
