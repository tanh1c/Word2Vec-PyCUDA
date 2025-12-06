# HS Results Statistics

## Overview

This directory contains parsed results from Hierarchical Softmax (HS) training experiments.

## CSV Files

### 1. `HS_results_wide.csv`
**Format**: One row per configuration (6 rows total)
- **Columns**: 
  - `epochs`: Number of training epochs (1, 3, 10)
  - `embed_dim`: Embedding dimension (300, 600)
  - `vocab_size`: Vocabulary size
  - `total_words`: Total number of words used for training
  - `sentences`: Number of sentences
  - `skipgram_semantic_accuracy`: Skip-gram semantic accuracy
  - `skipgram_syntactic_accuracy`: Skip-gram syntactic accuracy
  - `skipgram_total_accuracy`: Skip-gram total accuracy
  - `skipgram_training_time_seconds`: Skip-gram total training time
  - `skipgram_time_per_epoch_seconds`: Skip-gram average time per epoch
  - `cbow_semantic_accuracy`: CBOW semantic accuracy
  - `cbow_syntactic_accuracy`: CBOW syntactic accuracy
  - `cbow_total_accuracy`: CBOW total accuracy
  - `cbow_training_time_seconds`: CBOW total training time
  - `cbow_time_per_epoch_seconds`: CBOW average time per epoch

**Use case**: Compare both models side-by-side for the same configuration

### 2. `HS_results_long.csv` (Recommended for Visualization)
**Format**: One row per model per configuration (12 rows total)
- **Columns**:
  - `epochs`: Number of training epochs (1, 3, 10)
  - `embed_dim`: Embedding dimension (300, 600)
  - `model`: Model type ("Skip-gram" or "CBOW")
  - `semantic_accuracy`: Semantic accuracy
  - `syntactic_accuracy`: Syntactic accuracy
  - `total_accuracy`: Total accuracy
  - `training_time_seconds`: Total training time in seconds
  - `time_per_epoch_seconds`: Average time per epoch in seconds
  - `vocab_size`: Vocabulary size
  - `total_words`: Total number of words used for training
  - `sentences`: Number of sentences

**Use case**: Easy to filter by model type, create grouped bar charts, line plots, etc.

## Configuration Matrix

| Epochs | Dimension | Configurations |
|--------|-----------|----------------|
| 1      | 300       | ✓              |
| 1      | 600       | ✓              |
| 3      | 300       | ✓              |
| 3      | 600       | ✓              |
| 10     | 300       | ✓              |
| 10     | 600       | ✓              |

**Total**: 6 configurations × 2 models = 12 data points

## Visualization Suggestions

### Using `HS_results_long.csv` (Recommended)

1. **Accuracy Comparison**
   - Grouped bar chart: epochs × accuracy (grouped by model)
   - Line chart: epochs vs accuracy (separate lines for Skip-gram and CBOW)
   - Faceted by `embed_dim` (300 vs 600)

2. **Training Time Analysis**
   - Bar chart: training time by model and epochs
   - Compare `time_per_epoch_seconds` across configurations
   - Efficiency analysis: accuracy vs training time

3. **Semantic vs Syntactic**
   - Stacked bar chart: semantic + syntactic accuracy
   - Scatter plot: semantic vs syntactic accuracy

4. **Model Comparison**
   - Filter by `model` column
   - Compare Skip-gram vs CBOW performance
   - Analyze which model performs better for semantic vs syntactic tasks

### Example Python/Matplotlib Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('HS_results_long.csv')

# Accuracy by epochs and model
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='epochs', y='total_accuracy', hue='model')
plt.title('Total Accuracy by Epochs and Model')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(title='Model')
plt.show()

# Training time comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='epochs', y='training_time_seconds', hue='model')
plt.title('Training Time by Epochs and Model')
plt.ylabel('Time (seconds)')
plt.xlabel('Epochs')
plt.legend(title='Model')
plt.show()

# Accuracy vs Dimension
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='embed_dim', y='total_accuracy', hue='model')
plt.title('Accuracy by Embedding Dimension')
plt.ylabel('Accuracy')
plt.xlabel('Embedding Dimension')
plt.legend(title='Model')
plt.show()
```

## Data Summary

The results show:
- **Best Accuracy**: 3 epochs, dim 600, Skip-gram (52.88% total)
- **Fastest Training**: 1 epoch, dim 300, CBOW (442.2 seconds)
- **Most Efficient**: Analyze accuracy/time ratio

## Regenerating CSV Files

To regenerate the CSV files after updating result files:

```bash
python parse_results.py
```

