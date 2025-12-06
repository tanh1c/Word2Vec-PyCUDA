# HS vs NS Fair Comparison Graphs

## Overview

This script generates **3 comprehensive comparison graphs** between Hierarchical Softmax (HS) and Negative Sampling (NS) training methods, with **fair comparison** that considers `vocab_size` and `total_words` - two critical factors that significantly affect training time and accuracy.

## Why Fair Comparison?

Different configurations use different vocabulary sizes and total word counts:
- **HS**: vocab_size ranges from 260K to 520K, total_words from 290M to 800M
- **NS**: vocab_size ranges from 260K to 520K, total_words from 290M to 833M

Comparing configurations with vastly different data sizes is unfair. This script addresses this by:
1. **Grouping similar data sizes** for direct comparison
2. **Normalizing metrics** by vocab_size and total_words
3. **Showing efficiency metrics** (accuracy per time, words per second)
4. **Visualizing trade-offs** between accuracy and training time

## Requirements

```bash
pip install pandas matplotlib seaborn numpy
```

## Usage

```bash
cd w2v_implementation/Result
python compare_HS_vs_NS.py
```

## Generated Graphs

### Graph 1: Fair Comparison (Same Data Size)
**File**: `Graph1_Fair_Comparison.png`

Four subplots showing fair comparisons:

1. **~520K vocab, ~800M words**: Compares HS vs NS for configurations with similar large vocabulary and word count
   - Shows: 1ep-300d, 1ep-600d, 3ep-300d, 3ep-600d configurations
   - Fair comparison: Same vocab_size and total_words

2. **~260K vocab, ~290M words**: Compares HS vs NS for configurations with similar smaller vocabulary and word count
   - Shows: 10ep-300d, 10ep-600d configurations
   - Fair comparison: Same vocab_size and total_words

3. **Training Efficiency**: Accuracy per training minute
   - Higher = Better accuracy achieved per minute of training
   - Shows efficiency across all configurations

4. **Processing Speed**: Words processed per second
   - Higher = Faster data processing
   - Shows which method processes data faster

**Insights**:
- Direct comparison of HS vs NS with **same data size** (fair comparison)
- Efficiency metrics show which method is more time-efficient
- Processing speed shows which method handles data faster

### Graph 2: Accuracy vs Training Time (with Data Size)
**File**: `Graph2_Accuracy_vs_Time.png`

Two subplots (Skip-gram and CBOW) showing scatter plots:
- **X-axis**: Training Time (minutes)
- **Y-axis**: Total Accuracy
- **Point size**: Vocab Size (in K) - larger points = larger vocabulary
- **Color**: Method (HS = red, NS = blue)
- **Labels**: Configuration (epochs-dimension)

**Insights**:
- Trade-off between accuracy and training time
- Visualize how vocab_size affects both accuracy and time
- Identify configurations with best accuracy/time ratio
- See which method achieves better accuracy in similar time

### Graph 3: Normalized Comparison
**File**: `Graph3_Normalized_Comparison.png`

Two subplots showing normalized accuracy metrics:

1. **Accuracy per K Vocabulary**: Accuracy divided by vocabulary size (in thousands)
   - Shows efficiency per vocabulary unit
   - Higher = Better performance per vocabulary size
   - Normalizes for different vocab sizes

2. **Accuracy per Million Words**: Accuracy divided by total words (in millions)
   - Shows efficiency per data unit
   - Higher = Better performance per million words processed
   - Normalizes for different dataset sizes

**Insights**:
- Fair comparison across different data sizes
- Shows which method/model is more efficient per unit of data
- Identifies configurations with best data efficiency
- Normalizes for vocab_size and total_words differences

## Data Sources

The script reads from:
- `HS/HS_results_long.csv` - HS results in long format
- `NS/NS_results_long.csv` - NS results in long format

Required columns:
- `epochs`, `embed_dim`, `model`
- `total_accuracy`, `semantic_accuracy`, `syntactic_accuracy`
- `training_time_seconds`
- `vocab_size`, `total_words`, `sentences`

## Output Location

All generated PNG files are saved to the `Result/` directory:
- `Graph1_Fair_Comparison.png`
- `Graph2_Accuracy_vs_Time.png`
- `Graph3_Normalized_Comparison.png`

## Key Metrics Explained

### Efficiency
```
Efficiency = Total Accuracy / (Training Time in minutes)
```
- Higher = Better accuracy per minute of training
- Useful for comparing time efficiency

### Words per Second
```
Words per Second = Total Words / Training Time (seconds)
```
- Higher = Faster data processing
- Shows throughput performance

### Accuracy per K Vocabulary
```
Accuracy per K Vocab = Total Accuracy / (Vocab Size / 1000)
```
- Normalizes accuracy by vocabulary size
- Fair comparison across different vocab sizes

### Accuracy per Million Words
```
Accuracy per M Words = Total Accuracy / (Total Words / 1,000,000)
```
- Normalizes accuracy by dataset size
- Fair comparison across different dataset sizes

## Customization

You can modify the script to:
- Change colors and styles (lines 13-18)
- Adjust figure sizes for different output formats
- Add more subplots or combine different metrics
- Export to different formats (PDF, SVG) by changing the savefig format
- Adjust data size ranges for fair comparison (lines 75-78, 125-128)

## Notes

- **Fair Comparison**: Graph 1 only compares configurations with similar vocab_size and total_words
- **Missing Data**: NS has fewer configurations than HS - the script handles this gracefully
- **High DPI**: Graphs use 300 DPI for publication-quality output
- **Colorblind-Friendly**: Colors chosen to be distinguishable
- **Data Size Impact**: Larger vocab_size and total_words generally require more training time but may achieve higher accuracy

## Troubleshooting

### Missing Data
- NS has only 2 configurations (1ep-600d, 10ep-600d) while HS has 6 configurations
- The script handles this gracefully by showing available data points only
- Fair comparison subplots will only show configurations that exist in both methods

### Data Size Ranges
If you want to adjust the data size ranges for fair comparison:
- Edit lines 75-78 for the first fair comparison (large data)
- Edit lines 125-128 for the second fair comparison (small data)
- Adjust the ranges based on your actual data distribution

## Example Interpretation

### Example 1: Fair Comparison
Looking at Graph 1, Subplot 1 (~520K vocab, ~800M words):
- If HS Skip-gram (1ep-600d) = 0.455 and NS Skip-gram (1ep-600d) = 0.583
- **Conclusion**: NS Skip-gram performs better with the same data size
- This is a **fair comparison** because both use similar vocab_size and total_words

### Example 2: Efficiency
Looking at Graph 1, Subplot 3 (Training Efficiency):
- If HS Skip-gram efficiency = 0.0008 and NS Skip-gram efficiency = 0.0010
- **Conclusion**: NS Skip-gram achieves better accuracy per minute of training
- NS is more time-efficient

### Example 3: Normalized Comparison
Looking at Graph 3, Subplot 1 (Accuracy per K Vocabulary):
- If HS Skip-gram = 0.0009 and NS Skip-gram = 0.0011
- **Conclusion**: NS Skip-gram achieves better accuracy per unit of vocabulary
- NS is more vocabulary-efficient
