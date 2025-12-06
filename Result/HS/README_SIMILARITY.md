# Word Similarity Data for Heatmap Visualization

## Overview

This directory contains parsed word similarity data from "Most similar words" sections in HS result files. The data is extracted from evaluation outputs showing the top 5 most similar words for 8 test words across different configurations.

## CSV Files

### 1. `HS_similarity_long.csv` (Recommended for Heatmap)
**Format**: One row per similarity pair (480 rows total)

**Columns**:
- `epochs`: Number of training epochs (1, 3, 10)
- `embed_dim`: Embedding dimension (300, 600)
- `model`: Model type ("Skip-gram" or "CBOW")
- `test_word`: Test word (king, queen, man, woman, computer, science, university, student)
- `similar_word`: Similar word found by the model
- `similarity_score`: Cosine similarity score (0.0 to 1.0)

**Example**:
```csv
epochs,embed_dim,model,test_word,similar_word,similarity_score
1,300,Skip-gram,king,runcie,0.3867
1,300,Skip-gram,king,coretta,0.3853
1,300,Skip-gram,king,knighting,0.3794
```

**Use case**: 
- Create heatmaps comparing similarity scores across configurations
- Filter by test_word to analyze specific words
- Compare Skip-gram vs CBOW performance
- Analyze how epochs/dimensions affect similarity

### 2. `HS_similarity_by_word.csv` (Word-Focused Format)
**Format**: One row per test word per configuration (48 rows total)

**Columns**:
- `epochs`: Number of training epochs
- `embed_dim`: Embedding dimension
- `test_word`: Test word
- `sg_word_1` to `sg_word_5`: Top 5 similar words from Skip-gram
- `sg_score_1` to `sg_score_5`: Corresponding similarity scores
- `cbow_word_1` to `cbow_word_5`: Top 5 similar words from CBOW
- `cbow_score_1` to `cbow_score_5`: Corresponding similarity scores

**Use case**:
- Quick comparison of top similar words between models
- Analyze which words appear in top 5 across configurations
- Export for table presentations

### 3. `HS_similarity_matrix.csv` (Wide Matrix Format)
**Format**: One row per configuration (12 rows total)

**Columns**:
- `epochs`, `embed_dim`, `model`: Configuration identifiers
- Columns for each unique `test_word_similar_word` pair (148 pairs total)

**Use case**:
- Direct matrix for heatmap creation
- Compare all pairs across configurations at once

## Test Words

All configurations are evaluated on these 8 test words:
1. **king**
2. **queen**
3. **man**
4. **woman**
5. **computer**
6. **science**
7. **university**
8. **student**

Each test word has 5 most similar words extracted (top 5), resulting in:
- **40 similarity pairs** per model per configuration
- **80 pairs total** per configuration (40 Skip-gram + 40 CBOW)
- **480 pairs total** across all configurations

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

## Heatmap Visualization Suggestions

### Using `HS_similarity_long.csv` (Recommended)

#### Example 1: Similarity Score Heatmap by Test Word

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('HS_similarity_long.csv')

# Filter for specific test word
df_king = df[df['test_word'] == 'king']

# Create pivot table: epochs × embed_dim, values = similarity_score
# Group by model and similar_word
for model in ['Skip-gram', 'CBOW']:
    df_model = df_king[df_king['model'] == model]
    
    # Create heatmap: configuration vs similar_word
    pivot = df_model.pivot_table(
        index=['epochs', 'embed_dim'],
        columns='similar_word',
        values='similarity_score'
    )
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title(f'Similarity Scores for "king" - {model}')
    plt.ylabel('Configuration (Epochs, Dim)')
    plt.xlabel('Similar Words')
    plt.tight_layout()
    plt.show()
```

#### Example 2: Compare Models Side-by-Side

```python
# Create comparison heatmap
df_king = df[df['test_word'] == 'king']

pivot = df_king.pivot_table(
    index=['epochs', 'embed_dim', 'model'],
    columns='similar_word',
    values='similarity_score'
)

plt.figure(figsize=(15, 12))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Similarity Scores for "king" - All Configurations')
plt.ylabel('Configuration (Epochs, Dim, Model)')
plt.xlabel('Similar Words')
plt.tight_layout()
plt.show()
```

#### Example 3: Heatmap of Average Similarity by Configuration

```python
# Calculate average similarity score for each configuration
avg_similarity = df.groupby(['epochs', 'embed_dim', 'model', 'test_word'])['similarity_score'].mean().reset_index()

pivot = avg_similarity.pivot_table(
    index=['epochs', 'embed_dim', 'model'],
    columns='test_word',
    values='similarity_score'
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('Average Similarity Scores by Test Word')
plt.ylabel('Configuration (Epochs, Dim, Model)')
plt.xlabel('Test Words')
plt.tight_layout()
plt.show()
```

#### Example 4: Top Similar Word Comparison Across Configurations

```python
# Get top similar word for each test word in each configuration
top_similar = df.loc[df.groupby(['epochs', 'embed_dim', 'model', 'test_word'])['similarity_score'].idxmax()]

# Create heatmap showing top similarity scores
pivot = top_similar.pivot_table(
    index=['epochs', 'embed_dim', 'model'],
    columns='test_word',
    values='similarity_score'
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Top Similarity Score for Each Test Word')
plt.ylabel('Configuration (Epochs, Dim, Model)')
plt.xlabel('Test Words')
plt.tight_layout()
plt.show()
```

#### Example 5: Word-Specific Heatmap (All Configurations)

```python
# Create detailed heatmap for one word showing all configurations
test_word = 'king'
df_word = df[df['test_word'] == test_word]

# Create configuration label
df_word['config'] = df_word.apply(
    lambda x: f"{x['epochs']}ep-{x['embed_dim']}d-{x['model']}", axis=1
)

pivot = df_word.pivot_table(
    index='config',
    columns='similar_word',
    values='similarity_score'
)

plt.figure(figsize=(15, 10))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
            cbar_kws={'label': 'Similarity Score'})
plt.title(f'Similarity Heatmap for "{test_word}" - All Configurations')
plt.ylabel('Configuration')
plt.xlabel('Similar Words')
plt.tight_layout()
plt.show()
```

## Data Summary

- **Total similarity pairs**: 480 (6 configs × 2 models × 8 test words × 5 similar words)
- **Test words**: 8 (king, queen, man, woman, computer, science, university, student)
- **Similar words per test word**: 5 (top 5)
- **Models**: Skip-gram, CBOW
- **Epochs**: 1, 3, 10
- **Dimensions**: 300, 600

## Regenerating CSV Files

To regenerate the similarity CSV files after updating result files:

```bash
python parse_similarity.py
```

## Notes

- Similarity scores are cosine similarity (0.0 to 1.0, higher = more similar)
- Top 5 similar words are extracted from evaluation output
- Missing values in matrix format indicate the word pair doesn't appear in that configuration's top 5

