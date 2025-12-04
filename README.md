# Word2Vec PyCUDA

A complete implementation of Word2Vec with both CBOW and Skip-gram models using Numba CUDA for GPU acceleration. This implementation includes Hierarchical Softmax (HS), Negative Sampling (NS), Exponential Table optimization, and Phrase Detection.

## Features

- **Two Models**: Complete implementations of both CBOW and Skip-gram architectures
- **GPU Acceleration**: Uses Numba CUDA for fast training on GPU
- **Multiple Training Methods**: 
  - Negative Sampling (NS) only
  - Hierarchical Softmax (HS) only
  - HS + NS combination
- **Datasets Support**: 
  - Text8 corpus
  - WMT14/WMT15 News corpus (combines 2012-2014, with multiple size options)
- **Phrase Detection**: Automatic phrase detection (e.g., "new york" → "new_york")
- **Evaluation**: Word analogy test, similarity metrics, and Gensim comparison
- **Visualization**: t-SNE plots, similarity heatmaps, and training comparisons
- **Gensim Integration**: Train and compare with Gensim models
- **Interactive Menu**: User-friendly command-line interface

## Architecture

Based on the original Word2Vec paper by Mikolov et al. (2013):
- **Skip-gram**: Predicts context words from center word
- **CBOW**: Predicts center word from context words
- **Negative Sampling**: Efficient training with negative samples
- **Hierarchical Softmax**: Huffman tree-based efficient word prediction
- **Subsampling**: Reduces frequent words to improve quality
- **Exponential Table**: Precomputed sigmoid values for faster training

## Requirements

- Python 3.8+
- CUDA-capable GPU (T4 recommended for Colab)
- Poetry for dependency management

## Installation

### Using Poetry (Recommended)

1. Install Poetry if you haven't already:
   ```bash
   pip install poetry
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/tanh1c/Word2Vec-PyCUDA.git
   cd Word2Vec-PyCUDA
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### For Google Colab:

1. Upload all files to Colab
2. Install dependencies:
   ```bash
   !pip install numba-cuda==0.4.0 gensim>=4.0.0 scikit-learn matplotlib seaborn tqdm requests pynvml
   ```
3. Run the complete pipeline:
   ```bash
   !python run_all.py
   ```

### Alternative: Using pip (without Poetry)

If you prefer using pip directly:

```bash
pip install numba-cuda==0.4.0 numpy>=1.20.0 gensim>=4.0.0 scikit-learn>=1.0.0 matplotlib>=3.5.0 seaborn>=0.11.0 tqdm>=4.60.0 requests>=2.25.0 pynvml>=11.0.0
```

## Usage

### Interactive Menu (Recommended)

Run the interactive menu for easy configuration:

```bash
python run_all.py
```

This will guide you through:
1. Dataset selection (Text8, WMT14/WMT15 News)
2. Training method (NS only, HS only, HS + NS)
3. Dataset size (for WMT14: Full, Tiny, Small, Medium, Custom)
4. Phrase detection (Yes/No)
5. Gensim training (Yes/No)
6. Stop after evaluation (Yes/No)

### Command Line Options

You can also use command-line arguments:

```bash
# Train with specific options
python run_all.py --dataset text8 --method ns --phrases --gensim

# Stop after evaluation (skip visualization)
python run_all.py --stop-after-eval

# Use specific dataset size
python run_all.py --dataset wmt14 --size tiny
```

### Individual Components

You can also use individual modules:

```python
from w2v_skipgram import train_skipgram
from w2v_cbow import train_cbow
from evaluation import word_analogy_test, train_gensim_models
from visualization import plot_tsne

# Train Skip-gram with HS + NS
train_skipgram(
    data_path="./data/text8_processed",
    output_path="./output/vectors_skipgram",
    epochs=10,
    embed_dim=100,
    min_occurs=5,
    hs=1,  # Enable Hierarchical Softmax
    negative=5  # Enable Negative Sampling
)

# Evaluate
accuracy, details = word_analogy_test("./output/vectors_skipgram")

# Visualize
plot_tsne("./output/vectors_skipgram", "./output/tsne.png")
```

## Hyperparameters

Default parameters (based on Mikolov's paper):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_dim` | 100 | Embedding dimension |
| `window` | 5 | Context window size |
| `negative_samples` | 5 | Number of negative samples |
| `min_count` | 5 | Minimum word frequency |
| `epochs` | 10 | Training epochs |
| `lr_max` | 0.025 | Initial learning rate |
| `lr_min` | 0.0001 | Final learning rate |
| `subsampling_threshold` | 1e-5 | Subsampling threshold |
| `freq_exponent` | 0.75 | Frequency exponent for negative sampling |
| `exp_table_size` | 1000 | Exponential table size for fast sigmoid |
| `max_exp` | 6 | Maximum exponent value |

## Output Files

The pipeline generates the following outputs in `./output/`:

### Model Files
- `vectors_skipgram`: Skip-gram word vectors (word2vec format)
- `vectors_cbow`: CBOW word vectors (word2vec format)
- `*_params.json`: Training parameters
- `*_stats.json`: Training statistics

### Evaluation Files
- `skipgram_eval.json`: Skip-gram evaluation results
- `cbow_eval.json`: CBOW evaluation results
- `model_comparison.json`: Detailed model comparison
- `gensim_comparison.json`: Comparison with Gensim models (if enabled)

### Gensim Models (if enabled)
- `gensim/vectors_skipgram`: Gensim Skip-gram vectors
- `gensim/vectors_cbow`: Gensim CBOW vectors
- `gensim/gensim_eval_skipgram.json`: Gensim Skip-gram evaluation
- `gensim/gensim_eval_cbow.json`: Gensim CBOW evaluation

### Visualizations
- `skipgram_tsne.png`: t-SNE plot for Skip-gram
- `cbow_tsne.png`: t-SNE plot for CBOW
- `skipgram_heatmap.png`: Similarity heatmap for Skip-gram
- `cbow_heatmap.png`: Similarity heatmap for CBOW
- `training_comparison.png`: Training time and statistics comparison
- `accuracy_comparison.png`: Accuracy comparison chart

## Performance

Actual performance on Google Colab T4 GPU (Text8 dataset: 16.6M words, 60,603 vocabulary):

### Training Time (10 epochs)

| Method | Skip-gram | CBOW |
|--------|-----------|------|
| **NS only** | 103.5s (~1.7 min) | 22.8s (~0.4 min) |
| **HS only** | 240.2s (~4.0 min) | 57.1s (~1.0 min) |
| **HS + NS** | 338.5s (~5.6 min) | 65.8s (~1.1 min) |

### Accuracy (Word Analogy Test)

| Method | Skip-gram | CBOW |
|--------|-----------|------|
| **NS only** | 28.37% | 24.89% |
| **HS only** | 27.80% | 19.34% |
| **HS + NS** | 29.72% | 24.14% |

### Memory Usage

- **GPU Memory**: ~126 MB (weights + inputs + auxiliary data)
- **Model Size**: 60,603 vocabulary × 100 dimensions = 6.06M parameters
- **Efficient**: Optimized for T4 GPU (15GB) with room for larger datasets

### Comparison with Gensim

| Model | Custom Implementation | Gensim | Speedup |
|-------|----------------------|--------|---------|
| **Skip-gram** | 103.96s, 28.15% | 355.00s, 27.57% | **3.41x faster** |
| **CBOW** | 23.37s, 25.32% | 114.87s, 28.75% | **4.91x faster** |

**Note**: Custom implementation is significantly faster (3-5x) with comparable accuracy. Gensim CBOW has slightly higher accuracy (+3.43%) but takes 4.91x longer to train.

## Technical Details

### CUDA Implementation

- Uses Numba CUDA for GPU acceleration
- Custom kernels for both Skip-gram and CBOW
- Efficient memory management for T4 GPU
- Parallel processing across all training examples
- Exponential table for fast sigmoid computation
- Early skip logic for gradient stability

### Hierarchical Softmax

- Huffman tree construction based on word frequencies
- Efficient binary tree traversal for word prediction
- Reduced computation complexity from O(V) to O(log V)
- Syn1 weight matrix for internal tree nodes

### Negative Sampling

- Dynamic table size based on vocabulary size
- Frequency-based negative sample selection
- Efficient GPU parallel sampling

### Phrase Detection

- Two-pass phrase detection algorithm
- Score-based phrase identification
- Automatic phrase rewriting in data files
- Compatible with original word2phrase.c implementation

## Troubleshooting

### Common Issues

1. **CUDA not available**: Check GPU availability with `numba.cuda.is_available()`
2. **Out of memory**: Reduce `embed_dim` or `cuda_threads_per_block`
3. **Slow training**: Ensure GPU is being used, not CPU
4. **Import errors**: Install all required dependencies using Poetry
5. **Gensim API errors**: Ensure Gensim >= 4.0.0 (uses `vector_size` instead of `size`)

### Memory Optimization

For limited GPU memory:
- Reduce `embed_dim` to 50
- Decrease `cuda_threads_per_block` to 16
- Use smaller context window (`window=3`)
- Disable phrase detection for large datasets

## Project Structure

```
w2v_implementation/
├── w2v_common.py          # Common utilities, vocabulary, HS tree, exp table
├── w2v_cbow.py            # CBOW training implementation
├── w2v_skipgram.py        # Skip-gram training implementation
├── data_handler.py        # Data downloading and preprocessing
├── evaluation.py          # Evaluation and Gensim integration
├── visualization.py       # Visualization utilities
├── run_all.py            # Main pipeline script with interactive menu
├── pyproject.toml        # Poetry dependencies
├── requirements.txt      # pip dependencies (legacy)
└── README.md            # This file
```

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Original word2vec implementation: https://code.google.com/archive/p/word2vec/

## License

This implementation is based on the original word2vec project and follows the GNU Lesser General Public License (LGPL-3.0).

## Contributing

Feel free to submit issues and enhancement requests!

## Author

- **tanh1c** - [GitHub](https://github.com/tanh1c)

