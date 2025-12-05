# Copyright 2024 Word2Vec Implementation
# Evaluation utilities for word2vec models

import json
import os
import time
import re
from typing import List, Tuple, Dict, Any
import requests
from gensim.models import KeyedVectors


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
    #   NEW: Semantic & syntactic statistics
    # =====================================================
    semantic_correct = 0
    semantic_total = 0

    syntactic_correct = 0
    syntactic_total = 0

    for cat in details:
        correct = len(cat["correct"])
        total = correct + len(cat["incorrect"])
    
        # Classify semantic vs syntactic based on categories in questions-words.txt
        # Semantic (5 categories): capital-common-countries, capital-world, currency, city-in-state, family
        # Syntactic (9 categories): gram1-9 (adjective-to-adverb, opposite, comparative, superlative, etc.)
        section = cat["section"].lower()

        # Semantic categories keywords (5 categories from questions-words.txt)
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
            # Syntactic categories (all remaining categories, usually starting with "gram")
            syntactic_correct += correct
            syntactic_total += total

    semantic_acc = semantic_correct / semantic_total if semantic_total > 0 else 0
    syntactic_acc = syntactic_correct / syntactic_total if syntactic_total > 0 else 0

    # Total overall accuracy
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


