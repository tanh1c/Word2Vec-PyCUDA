#!/usr/bin/env python3
"""
Compare word similarity results between HS and NS methods
Generate simplified similarity comparison graph with clear heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# File paths
base_dir = Path(__file__).parent
hs_long_path = base_dir / "HS" / "HS_similarity_long.csv"
ns_long_path = base_dir / "NS" / "NS_similarity_long.csv"

# Load data
print("Loading similarity data...")
hs_long = pd.read_csv(hs_long_path)
ns_long = pd.read_csv(ns_long_path)

# Add method column
hs_long['method'] = 'HS'
ns_long['method'] = 'NS'

# Combine data
combined_long = pd.concat([hs_long, ns_long], ignore_index=True)

# Get unique test words
test_words = sorted(combined_long['test_word'].unique())
print(f"Loaded {len(combined_long)} similarity rows")
print(f"Test words: {test_words}")


# ============================================
# GRAPH: Simplified Similarity Comparison with Clear Heatmap
# ============================================
def create_similarity_graph():
    """
    Create simplified similarity comparison graph with:
    1. Large, clear heatmap comparing HS vs NS for top word pairs
    2. Average similarity scores comparison
    """
    print("\n[Graph] Creating Similarity Comparison with Heatmap...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    
    # ============================================
    # Subplot 1: Main Heatmap - HS vs NS for Top Word Pairs
    # ============================================
    ax1 = axes[0]
    
    # Get top similar words for each test word
    heatmap_data = []
    
    for test_word in test_words:
        word_data = combined_long[combined_long['test_word'] == test_word].copy()
        
        # Get top 3 most common similar words (by frequency and average score) - limit to avoid overcrowding
        top_similar = word_data.groupby('similar_word').agg({
            'similarity_score': ['mean', 'count']
        }).reset_index()
        top_similar.columns = ['similar_word', 'avg_score', 'count']
        top_similar = top_similar.sort_values(['count', 'avg_score'], ascending=[False, False]).head(3)
        
        for _, row in top_similar.iterrows():
            similar_word = row['similar_word']
            
            hs_data = word_data[(word_data['similar_word'] == similar_word) & 
                               (word_data['method'] == 'HS')]['similarity_score']
            ns_data = word_data[(word_data['similar_word'] == similar_word) & 
                               (word_data['method'] == 'NS')]['similarity_score']
            
            hs_avg = hs_data.mean() if len(hs_data) > 0 else np.nan
            ns_avg = ns_data.mean() if len(ns_data) > 0 else np.nan
            
            if not (np.isnan(hs_avg) and np.isnan(ns_avg)):
                heatmap_data.append({
                    'pair': f"{test_word} ↔ {similar_word}",
                    'HS': hs_avg if not np.isnan(hs_avg) else 0,
                    'NS': ns_avg if not np.isnan(ns_avg) else 0
                })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    if len(heatmap_df) == 0:
        print("Warning: No data for heatmap")
        return
    
    # Sort by average of HS and NS to show best pairs first
    heatmap_df['avg_score'] = (heatmap_df['HS'] + heatmap_df['NS']) / 2
    heatmap_df = heatmap_df.sort_values('avg_score', ascending=False)
    
    # Limit to top 20 pairs to avoid overcrowding
    heatmap_df = heatmap_df.head(20)
    
    # Create pivot for heatmap
    heatmap_pivot = heatmap_df.set_index('pair')[['HS', 'NS']].T
    
    # Create heatmap
    sns.heatmap(heatmap_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
               vmin=0, vmax=1, cbar_kws={'label': 'Similarity Score', 'shrink': 0.8},
               linewidths=1, linecolor='white', ax=ax1,
               annot_kws={'fontsize': 9, 'fontweight': 'bold'},
               square=False)
    
    ax1.set_title('Word Similarity Heatmap: HS vs NS Comparison\n(Top 20 Word Pairs by Average Similarity Score)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax1.set_xlabel('Word Pairs (Test Word ↔ Similar Word)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Training Method', fontsize=13, fontweight='bold')
    ax1.set_yticklabels(['Hierarchical Softmax', 'Negative Sampling'], 
                       rotation=0, fontsize=12, fontweight='bold')
    ax1.set_xticklabels(heatmap_pivot.columns, rotation=45, ha='right', fontsize=10)
    
    # ============================================
    # Subplot 2: Average Similarity Scores by Test Word
    # ============================================
    ax2 = axes[1]
    
    # Calculate average scores per test word
    avg_scores = combined_long.groupby(['method', 'test_word'])['similarity_score'].mean().reset_index()
    
    x_pos = np.arange(len(test_words))
    width = 0.4
    
    hs_avg = []
    ns_avg = []
    
    for word in test_words:
        hs_val = avg_scores[(avg_scores['test_word'] == word) & 
                           (avg_scores['method'] == 'HS')]['similarity_score'].values
        ns_val = avg_scores[(avg_scores['test_word'] == word) & 
                           (avg_scores['method'] == 'NS')]['similarity_score'].values
        
        hs_avg.append(hs_val[0] if len(hs_val) > 0 else 0)
        ns_avg.append(ns_val[0] if len(ns_val) > 0 else 0)
    
    bars1 = ax2.bar(x_pos - width/2, hs_avg, width, 
                   label='Hierarchical Softmax', alpha=0.85, color='#2E86AB', 
                   edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, ns_avg, width, 
                   label='Negative Sampling', alpha=0.85, color='#A23B72', 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (hs_val, ns_val) in enumerate(zip(hs_avg, ns_avg)):
        if hs_val > 0:
            ax2.text(i - width/2, hs_val + 0.015, f'{hs_val:.3f}', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        if ns_val > 0:
            ax2.text(i + width/2, ns_val + 0.015, f'{ns_val:.3f}', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Test Word', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Similarity Score', fontsize=13, fontweight='bold')
    ax2.set_title('Average Similarity Scores by Test Word\n(HS vs NS Comparison)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(test_words, fontsize=11, fontweight='bold', rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True, 
              edgecolor='black', framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    max_val = max(max(hs_avg) if hs_avg else [0], max(ns_avg) if ns_avg else [0])
    ax2.set_ylim(bottom=0, top=max_val * 1.25)
    
    # Add horizontal reference lines
    for y_val in [0.3, 0.5, 0.7]:
        if y_val < max_val * 1.25:
            ax2.axhline(y=y_val, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    plt.suptitle('Word Similarity Analysis: Hierarchical Softmax vs Negative Sampling', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'Graph_Similarity_Comparison.png', bbox_inches='tight', dpi=300)
    print("  ✓ Saved: Graph_Similarity_Comparison.png")
    plt.close()


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Generate similarity comparison graph"""
    print("="*60)
    print("Word Similarity Comparison Graph Generator")
    print("(HS vs NS with Clear Heatmap)")
    print("="*60)
    
    try:
        create_similarity_graph()
        
        print("\n" + "="*60)
        print("✓ Graph generated successfully!")
        print("="*60)
        print("\nGenerated file:")
        print("  Graph_Similarity_Comparison.png - Similarity comparison with heatmap")
        print(f"\nFile saved to: {base_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
