#!/usr/bin/env python3
"""
Compare HS (Hierarchical Softmax) vs NS (Negative Sampling) results
Generate fair comparison graphs considering vocab_size and total_words
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# File paths
base_dir = Path(__file__).parent
hs_long_path = base_dir / "HS" / "HS_results_long.csv"
ns_long_path = base_dir / "NS" / "NS_results_long.csv"

# Load data
print("Loading data...")
hs_long = pd.read_csv(hs_long_path)
ns_long = pd.read_csv(ns_long_path)

# Add method column to distinguish HS and NS
hs_long['method'] = 'HS'
ns_long['method'] = 'NS'

# Combine data for comparison
combined_df = pd.concat([hs_long, ns_long], ignore_index=True)

# Create configuration label
combined_df['config'] = combined_df.apply(
    lambda x: f"{x['epochs']}ep-{x['embed_dim']}d", axis=1
)

print(f"Loaded {len(hs_long)} HS rows and {len(ns_long)} NS rows")
print(f"Total combined rows: {len(combined_df)}")

# Calculate efficiency metrics
combined_df['efficiency'] = combined_df['total_accuracy'] / (combined_df['training_time_seconds'] / 60)  # accuracy per minute
combined_df['words_per_second'] = combined_df['total_words'] / combined_df['training_time_seconds']  # words processed per second


# ============================================
# GRAPH 1: Fair Comparison - Same Data Size
# ============================================
def graph1_fair_comparison():
    """
    Compare HS vs NS for configurations with similar vocab_size and total_words
    This ensures fair comparison
    """
    print("\n[Graph 1] Creating Fair Comparison (Same Data Size)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Group 1 - vocab_size == 520233 and total_words == 783000000
    ax1 = axes[0]
    
    # Filter for Group 1: Large vocabulary and large dataset
    similar_data = combined_df[
        (combined_df['vocab_size'] == 520233) &
        (combined_df['total_words'] == 783000000)
    ].copy()
    
    if len(similar_data) > 0:
        # Get values for title
        vocab_val = 520233
        words_val = 783
        
        similar_data['config_label'] = similar_data.apply(
            lambda x: f"{x['epochs']}ep-{x['embed_dim']}d-{x['model'][0]}", axis=1
        )
        
        pivot_similar = similar_data.pivot_table(
            index='config_label',
            columns='method',
            values='total_accuracy',
            aggfunc='mean'
        )
        
        x_pos = range(len(pivot_similar))
        width = 0.35
        
        hs_vals = pivot_similar['HS'].values if 'HS' in pivot_similar.columns else []
        ns_vals = pivot_similar['NS'].values if 'NS' in pivot_similar.columns else []
        
        if len(hs_vals) > 0:
            ax1.bar([x - width/2 for x in x_pos], hs_vals, width, 
                   label='Hierarchical Softmax', alpha=0.8, color='#2E86AB')
            for i, val in enumerate(hs_vals):
                if pd.notna(val):
                    ax1.text(i - width/2, val + 0.01, f'{val:.3f}', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if len(ns_vals) > 0:
            ax1.bar([x + width/2 for x in x_pos], ns_vals, width, 
                   label='Negative Sampling', alpha=0.8, color='#A23B72')
            for i, val in enumerate(ns_vals):
                if pd.notna(val):
                    ax1.text(i + width/2, val + 0.01, f'{val:.3f}', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(pivot_similar.index, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel('Total Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(f'Group 1: Large Dataset Comparison\nVocab Size: {vocab_val/1000:.0f}K tokens | Dataset: {words_val}M words', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
    
    # Subplot 2: Group 2 - vocab_size = 260796 and total_words = 291539433 (exact match)
    ax2 = axes[1]
    
    similar_data2 = combined_df[
        (combined_df['vocab_size'] == 260796) &
        (combined_df['total_words'] == 291539433)
    ].copy()
    
    if len(similar_data2) > 0:
        vocab_val = int(similar_data2['vocab_size'].iloc[0])
        words_val = int(similar_data2['total_words'].iloc[0] / 1e6)
        
        similar_data2['config_label'] = similar_data2.apply(
            lambda x: f"{x['epochs']}ep-{x['embed_dim']}d-{x['model'][0]}", axis=1
        )
        
        pivot_similar2 = similar_data2.pivot_table(
            index='config_label',
            columns='method',
            values='total_accuracy',
            aggfunc='mean'
        )
        
        x_pos2 = range(len(pivot_similar2))
        
        hs_vals2 = pivot_similar2['HS'].values if 'HS' in pivot_similar2.columns else []
        ns_vals2 = pivot_similar2['NS'].values if 'NS' in pivot_similar2.columns else []
        
        if len(hs_vals2) > 0:
            ax2.bar([x - width/2 for x in x_pos2], hs_vals2, width, 
                   label='Hierarchical Softmax', alpha=0.8, color='#2E86AB')
            for i, val in enumerate(hs_vals2):
                if pd.notna(val):
                    ax2.text(i - width/2, val + 0.01, f'{val:.3f}', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if len(ns_vals2) > 0:
            ax2.bar([x + width/2 for x in x_pos2], ns_vals2, width, 
                   label='Negative Sampling', alpha=0.8, color='#A23B72')
            for i, val in enumerate(ns_vals2):
                if pd.notna(val):
                    ax2.text(i + width/2, val + 0.01, f'{val:.3f}', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(pivot_similar2.index, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Total Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Group 2: Medium Dataset Comparison\nVocab Size: {vocab_val/1000:.0f}K tokens | Dataset: {words_val}M words', 
                     fontsize=13, fontweight='bold', pad=15)
        ax2.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'Graph1_Fair_Comparison.png', bbox_inches='tight')
    print("  ✓ Saved: Graph1_Fair_Comparison.png")
    plt.close()


# ============================================
# GRAPH 2: Accuracy vs Training Time (with Data Size)
# ============================================
def graph2_accuracy_vs_time():
    """
    Improved scatter plot: Accuracy vs Training Time with better visualization
    Shows the trade-off between accuracy and training time with clearer presentation
    """
    print("\n[Graph 2] Creating Accuracy vs Training Time...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Skip-gram - Large Dataset (Group 1)
    ax1 = axes[0, 0]
    
    skipgram_data = combined_df[combined_df['model'] == 'Skip-gram'].copy()
    large_dataset = skipgram_data[
        (skipgram_data['vocab_size'] == 520233) &
        (skipgram_data['total_words'] == 783000000)
    ].copy()
    
    for method in ['HS', 'NS']:
        subset = large_dataset[large_dataset['method'] == method]
        if len(subset) == 0:
            continue
            
        marker = 'o' if method == 'HS' else 's'
        color = '#2E86AB' if method == 'HS' else '#A23B72'
        label_name = 'Hierarchical Softmax' if method == 'HS' else 'Negative Sampling'
        
        # Larger markers for better visibility
        sizes = 300 + (subset['vocab_size'] / 1000) * 2
        
        scatter = ax1.scatter(subset['training_time_seconds'] / 60, 
                             subset['total_accuracy'],
                             s=sizes, alpha=0.8, label=label_name, 
                             marker=marker, color=color, edgecolors='white', linewidth=2.0,
                             zorder=5)
        
        # Add clear labels with config info
        for idx, row in subset.iterrows():
            ax1.annotate(f"{row['epochs']}ep-{int(row['embed_dim'])}d", 
                        (row['training_time_seconds'] / 60, row['total_accuracy']),
                        fontsize=9, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                        ha='center', va='center')
    
    ax1.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Skip-gram: Large Dataset (520K vocab, 783M words)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax1.set_ylim(bottom=0, top=max(skipgram_data['total_accuracy']) * 1.1 if len(skipgram_data) > 0 else 0.7)
    
    # Subplot 2: Skip-gram - Medium Dataset (Group 2)
    ax2 = axes[0, 1]
    
    medium_dataset = skipgram_data[
        (skipgram_data['vocab_size'] == 260796) &
        (skipgram_data['total_words'] == 291539433)
    ].copy()
    
    for method in ['HS', 'NS']:
        subset = medium_dataset[medium_dataset['method'] == method]
        if len(subset) == 0:
            continue
            
        marker = 'o' if method == 'HS' else 's'
        color = '#2E86AB' if method == 'HS' else '#A23B72'
        label_name = 'Hierarchical Softmax' if method == 'HS' else 'Negative Sampling'
        
        sizes = 300 + (subset['vocab_size'] / 1000) * 2
        
        scatter = ax2.scatter(subset['training_time_seconds'] / 60, 
                             subset['total_accuracy'],
                             s=sizes, alpha=0.8, label=label_name, 
                             marker=marker, color=color, edgecolors='white', linewidth=2.0,
                             zorder=5)
        
        for idx, row in subset.iterrows():
            ax2.annotate(f"{row['epochs']}ep-{int(row['embed_dim'])}d", 
                        (row['training_time_seconds'] / 60, row['total_accuracy']),
                        fontsize=9, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                        ha='center', va='center')
    
    ax2.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Skip-gram: Medium Dataset (260K vocab, 292M words)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax2.set_ylim(bottom=0, top=max(skipgram_data['total_accuracy']) * 1.1 if len(skipgram_data) > 0 else 0.7)
    
    # Subplot 3: CBOW - Large Dataset (Group 1)
    ax3 = axes[1, 0]
    
    cbow_data = combined_df[combined_df['model'] == 'CBOW'].copy()
    large_dataset_cb = cbow_data[
        (cbow_data['vocab_size'] == 520233) &
        (cbow_data['total_words'] == 783000000)
    ].copy()
    
    for method in ['HS', 'NS']:
        subset = large_dataset_cb[large_dataset_cb['method'] == method]
        if len(subset) == 0:
            continue
            
        marker = 'o' if method == 'HS' else 's'
        color = '#2E86AB' if method == 'HS' else '#A23B72'
        label_name = 'Hierarchical Softmax' if method == 'HS' else 'Negative Sampling'
        
        sizes = 300 + (subset['vocab_size'] / 1000) * 2
        
        scatter = ax3.scatter(subset['training_time_seconds'] / 60, 
                             subset['total_accuracy'],
                             s=sizes, alpha=0.8, label=label_name, 
                             marker=marker, color=color, edgecolors='white', linewidth=2.0,
                             zorder=5)
        
        for idx, row in subset.iterrows():
            ax3.annotate(f"{row['epochs']}ep-{int(row['embed_dim'])}d", 
                        (row['training_time_seconds'] / 60, row['total_accuracy']),
                        fontsize=9, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                        ha='center', va='center')
    
    ax3.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('CBOW: Large Dataset (520K vocab, 783M words)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax3.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax3.set_ylim(bottom=0, top=max(cbow_data['total_accuracy']) * 1.1 if len(cbow_data) > 0 else 0.7)
    
    # Subplot 4: CBOW - Medium Dataset (Group 2)
    ax4 = axes[1, 1]
    
    medium_dataset_cb = cbow_data[
        (cbow_data['vocab_size'] == 260796) &
        (cbow_data['total_words'] == 291539433)
    ].copy()
    
    for method in ['HS', 'NS']:
        subset = medium_dataset_cb[medium_dataset_cb['method'] == method]
        if len(subset) == 0:
            continue
            
        marker = 'o' if method == 'HS' else 's'
        color = '#2E86AB' if method == 'HS' else '#A23B72'
        label_name = 'Hierarchical Softmax' if method == 'HS' else 'Negative Sampling'
        
        sizes = 300 + (subset['vocab_size'] / 1000) * 2
        
        scatter = ax4.scatter(subset['training_time_seconds'] / 60, 
                             subset['total_accuracy'],
                             s=sizes, alpha=0.8, label=label_name, 
                             marker=marker, color=color, edgecolors='white', linewidth=2.0,
                             zorder=5)
        
        for idx, row in subset.iterrows():
            ax4.annotate(f"{row['epochs']}ep-{int(row['embed_dim'])}d", 
                        (row['training_time_seconds'] / 60, row['total_accuracy']),
                        fontsize=9, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                        ha='center', va='center')
    
    ax4.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('CBOW: Medium Dataset (260K vocab, 292M words)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax4.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax4.set_ylim(bottom=0, top=max(cbow_data['total_accuracy']) * 1.1 if len(cbow_data) > 0 else 0.7)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'Graph2_Accuracy_vs_Time.png', bbox_inches='tight')
    print("  ✓ Saved: Graph2_Accuracy_vs_Time.png")
    plt.close()


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Generate all comparison graphs with fair comparison"""
    print("="*60)
    print("HS vs NS Fair Comparison Graphs Generator")
    print("(Considering vocab_size and total_words)")
    print("="*60)
    
    try:
        graph1_fair_comparison()
        graph2_accuracy_vs_time()
        
        print("\n" + "="*60)
        print("✓ All graphs generated successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  1. Graph1_Fair_Comparison.png - Fair comparison with similar data sizes")
        print("  2. Graph2_Accuracy_vs_Time.png - Accuracy vs Training Time trade-off (by dataset size)")
        print(f"\nAll files saved to: {base_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
