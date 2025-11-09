# Copyright 2024 Word2Vec Implementation
# Visualization utilities for word2vec models

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from collections import Counter


def plot_tsne(vectors_path: str, output_path: str, n_words: int = 500, perplexity: int = 30, 
              random_state: int = 42, figsize: tuple = (12, 10)):
    """
    Create t-SNE visualization of word embeddings.
    """
    print(f"Creating t-SNE visualization for: {vectors_path}")
    print(f"Parameters: n_words={n_words}, perplexity={perplexity}")
    
    # Load vectors
    vecs = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
    print(f"Loaded {len(vecs)} word vectors")
    
    # Select most frequent words for visualization
    if n_words < len(vecs):
        # Get most frequent words (assuming they are ordered by frequency in the file)
        words = list(vecs.key_to_index.keys())[:n_words]
    else:
        words = list(vecs.key_to_index.keys())
        n_words = len(words)
    
    print(f"Selected {n_words} words for visualization")
    
    # Get vectors for selected words
    vectors = np.array([vecs[word] for word in words])
    
    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                n_iter=1000, verbose=1)
    vectors_2d = tsne.fit_transform(vectors)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=20)
    
    # Annotate some interesting words
    interesting_words = ["king", "queen", "man", "woman", "computer", "science", 
                        "university", "student", "good", "bad", "big", "small",
                        "the", "and", "of", "to", "in", "is", "it", "you"]
    
    for i, word in enumerate(words):
        if word in interesting_words:
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title(f"t-SNE Visualization of Word Embeddings\n({n_words} words, perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to: {output_path}")


def plot_similarity_heatmap(vectors_path: str, words: List[str], output_path: str, 
                           figsize: tuple = (10, 8)):
    """
    Create similarity heatmap for selected words.
    """
    print(f"Creating similarity heatmap for: {vectors_path}")
    print(f"Words: {words}")
    
    # Load vectors
    vecs = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
    
    # Filter words that exist in vocabulary
    existing_words = [word for word in words if word in vecs]
    if len(existing_words) != len(words):
        missing = set(words) - set(existing_words)
        print(f"Warning: Missing words: {missing}")
    
    if len(existing_words) < 2:
        print("Error: Need at least 2 words for heatmap")
        return
    
    # Calculate similarity matrix
    n = len(existing_words)
    similarity_matrix = np.zeros((n, n))
    
    for i, word1 in enumerate(existing_words):
        for j, word2 in enumerate(existing_words):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = vecs.similarity(word1, word2)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, 
                xticklabels=existing_words, 
                yticklabels=existing_words,
                annot=True, 
                fmt='.3f', 
                cmap='coolwarm', 
                center=0,
                square=True)
    
    plt.title("Word Similarity Heatmap")
    plt.xlabel("Words")
    plt.ylabel("Words")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Similarity heatmap saved to: {output_path}")


def plot_training_comparison(skipgram_stats_path: str, cbow_stats_path: str, 
                           output_path: str, figsize: tuple = (15, 5)):
    """
    Create comparison plots for Skip-gram vs CBOW training.
    """
    print(f"Creating training comparison plots...")
    
    # Load statistics
    with open(skipgram_stats_path, 'r') as f:
        sg_stats = json.load(f)
    with open(cbow_stats_path, 'r') as f:
        cbow_stats = json.load(f)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Training time comparison
    models = ['Skip-gram', 'CBOW']
    times = [sg_stats.get('epoch_time_total_seconds', 0), 
             cbow_stats.get('epoch_time_total_seconds', 0)]
    
    axes[0].bar(models, times, color=['skyblue', 'lightcoral'])
    axes[0].set_title('Training Time Comparison')
    axes[0].set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        axes[0].text(i, v + max(times)*0.01, f'{v:.1f}s', ha='center', va='bottom')
    
    # 2. Epoch times comparison
    sg_epochs = sg_stats.get('epoch_times_all_seconds', [])
    cbow_epochs = cbow_stats.get('epoch_times_all_seconds', [])
    
    if sg_epochs and cbow_epochs:
        epochs = range(1, len(sg_epochs) + 1)
        axes[1].plot(epochs, sg_epochs, 'o-', label='Skip-gram', color='blue')
        axes[1].plot(epochs, cbow_epochs, 's-', label='CBOW', color='red')
        axes[1].set_title('Epoch Time Progression')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # 3. Data statistics
    sg_sentences = sg_stats.get('sentence_count', 0)
    sg_words = sg_stats.get('word_count', 0)
    cbow_sentences = cbow_stats.get('sentence_count', 0)
    cbow_words = cbow_stats.get('word_count', 0)
    
    x = np.arange(2)
    width = 0.35
    
    axes[2].bar(x - width/2, [sg_sentences, cbow_sentences], width, 
                label='Sentences', color='lightgreen')
    axes[2].bar(x + width/2, [sg_words, cbow_words], width, 
                label='Words', color='orange')
    
    axes[2].set_title('Data Statistics')
    axes[2].set_ylabel('Count')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models)
    axes[2].legend()
    
    # Add value labels on bars
    for i, (sent, words) in enumerate([(sg_sentences, sg_words), (cbow_sentences, cbow_words)]):
        axes[2].text(i - width/2, sent + max(sg_sentences, cbow_sentences)*0.01, 
                    f'{sent:,}', ha='center', va='bottom', fontsize=8)
        axes[2].text(i + width/2, words + max(sg_words, cbow_words)*0.01, 
                    f'{words:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training comparison plot saved to: {output_path}")


def plot_accuracy_comparison(skipgram_acc: float, cbow_acc: float, output_path: str, 
                            figsize: tuple = (8, 6)):
    """
    Create accuracy comparison bar chart.
    """
    print(f"Creating accuracy comparison plot...")
    
    models = ['Skip-gram', 'CBOW']
    accuracies = [skipgram_acc, cbow_acc]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.4f}\n({acc*100:.2f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Word Analogy Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, max(accuracies) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotation
    diff = skipgram_acc - cbow_acc
    plt.text(0.5, max(accuracies) * 0.9, 
             f'Difference: {diff:.4f} ({(diff*100):+.2f}%)', 
             ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy comparison plot saved to: {output_path}")
