#!/usr/bin/env python3
"""
Parse HS result files and create CSV for visualization
"""

import re
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

def parse_file(file_path: str) -> Dict:
    """Parse a single result file and extract metrics"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract epoch and dimension from filename
    filename = os.path.basename(file_path)
    epoch_match = re.search(r'(\d+)\s+epoch', filename, re.IGNORECASE)
    dim_match = re.search(r'dim\s+(\d+)', filename, re.IGNORECASE)
    
    epochs = int(epoch_match.group(1)) if epoch_match else None
    embed_dim = int(dim_match.group(1)) if dim_match else None
    
    result = {
        'epochs': epochs,
        'embed_dim': embed_dim,
        'model_type': None,  # Will be set later
    }
    
    # Extract vocabulary size
    vocab_match = re.search(r'Vocab size:\s*([\d,]+)', content)
    if vocab_match:
        result['vocab_size'] = int(vocab_match.group(1).replace(',', ''))
    else:
        vocab_match = re.search(r'vocab size\s+([\d,]+)', content, re.IGNORECASE)
        if vocab_match:
            result['vocab_size'] = int(vocab_match.group(1).replace(',', ''))
        else:
            result['vocab_size'] = None
    
    # Extract total words - look for "Data loaded:" pattern which is more reliable
    data_loaded_match = re.search(r'Data loaded:\s*([\d,]+)\s+sentences,\s*([\d,]+)\s+total words', content, re.IGNORECASE)
    if data_loaded_match:
        result['sentences'] = int(data_loaded_match.group(1).replace(',', ''))
        result['total_words'] = int(data_loaded_match.group(2).replace(',', ''))
    else:
        # Fallback to separate searches
        words_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s+total words', content, re.IGNORECASE)
        if words_match:
            result['total_words'] = int(words_match.group(1).replace(',', ''))
        else:
            # Try looking for number with commas before "words"
            words_match = re.search(r'([\d,]+)\s+words', content, re.IGNORECASE)
            if words_match:
                result['total_words'] = int(words_match.group(1).replace(',', ''))
            else:
                result['total_words'] = None
        
        sentences_match = re.search(r'([\d,]+)\s+sentences', content, re.IGNORECASE)
        if sentences_match:
            result['sentences'] = int(sentences_match.group(1).replace(',', ''))
        else:
            result['sentences'] = None
    
    # Extract Skip-gram metrics
    sg_sem_match = re.search(r'Skip-gram semantic:\s+([\d.]+)', content)
    sg_syn_match = re.search(r'Skip-gram syntactic:\s+([\d.]+)', content)
    sg_total_match = re.search(r'Skip-gram total:\s+([\d.]+)', content)
    
    result['skipgram_semantic_accuracy'] = float(sg_sem_match.group(1)) if sg_sem_match else None
    result['skipgram_syntactic_accuracy'] = float(sg_syn_match.group(1)) if sg_syn_match else None
    result['skipgram_total_accuracy'] = float(sg_total_match.group(1)) if sg_total_match else None
    
    # Extract CBOW metrics
    cbow_sem_match = re.search(r'CBOW semantic:\s+([\d.]+)', content)
    cbow_syn_match = re.search(r'CBOW syntactic:\s+([\d.]+)', content)
    cbow_total_match = re.search(r'CBOW total:\s+([\d.]+)', content)
    
    result['cbow_semantic_accuracy'] = float(cbow_sem_match.group(1)) if cbow_sem_match else None
    result['cbow_syntactic_accuracy'] = float(cbow_syn_match.group(1)) if cbow_syn_match else None
    result['cbow_total_accuracy'] = float(cbow_total_match.group(1)) if cbow_total_match else None
    
    # Extract Skip-gram training time - look for pattern after "Skip-gram training completed!"
    sg_section = re.search(r'Skip-gram training completed!.*?(?=CBOW|$)', content, re.DOTALL | re.IGNORECASE)
    if sg_section:
        sg_time_match = re.search(r'Total training time:\s+([\d.]+)s', sg_section.group(0))
        if sg_time_match:
            result['skipgram_training_time_seconds'] = float(sg_time_match.group(1))
        else:
            result['skipgram_training_time_seconds'] = None
    else:
        result['skipgram_training_time_seconds'] = None
    
    # Extract CBOW training time - look for pattern after "CBOW training completed!"
    cbow_section = re.search(r'CBOW training completed!.*?(?=EVALUATING|$)', content, re.DOTALL | re.IGNORECASE)
    if cbow_section:
        cbow_time_match = re.search(r'Total training time:\s+([\d.]+)s', cbow_section.group(0))
        if cbow_time_match:
            result['cbow_training_time_seconds'] = float(cbow_time_match.group(1))
        else:
            result['cbow_training_time_seconds'] = None
    else:
        result['cbow_training_time_seconds'] = None
    
    # Extract epoch times (average per epoch)
    if result['skipgram_training_time_seconds'] and epochs:
        result['skipgram_time_per_epoch_seconds'] = result['skipgram_training_time_seconds'] / epochs
    else:
        result['skipgram_time_per_epoch_seconds'] = None
    
    if result['cbow_training_time_seconds'] and epochs:
        result['cbow_time_per_epoch_seconds'] = result['cbow_training_time_seconds'] / epochs
    else:
        result['cbow_time_per_epoch_seconds'] = None
    
    return result


def parse_all_files(base_dir: str) -> List[Dict]:
    """Parse all result files in the directory structure"""
    base_path = Path(base_dir)
    all_results = []
    
    # Iterate through epoch directories
    for epoch_dir in sorted(base_path.iterdir()):
        if not epoch_dir.is_dir():
            continue
        
        epoch_name = epoch_dir.name
        epoch_num = int(re.search(r'\d+', epoch_name).group()) if re.search(r'\d+', epoch_name) else None
        
        # Iterate through files in epoch directory
        for file_path in sorted(epoch_dir.glob('*.txt')):
            result = parse_file(str(file_path))
            result['epochs'] = epoch_num  # Override with directory epoch
            all_results.append(result)
    
    return all_results


def create_csv_wide_format(results: List[Dict], output_path: str):
    """Create CSV in wide format (one row per configuration)"""
    fieldnames = [
        'epochs',
        'embed_dim',
        'vocab_size',
        'total_words',
        'sentences',
        'skipgram_semantic_accuracy',
        'skipgram_syntactic_accuracy',
        'skipgram_total_accuracy',
        'skipgram_training_time_seconds',
        'skipgram_time_per_epoch_seconds',
        'cbow_semantic_accuracy',
        'cbow_syntactic_accuracy',
        'cbow_total_accuracy',
        'cbow_training_time_seconds',
        'cbow_time_per_epoch_seconds',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {field: result.get(field) for field in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Created wide format CSV: {output_path}")


def create_csv_long_format(results: List[Dict], output_path: str):
    """Create CSV in long format (one row per model per configuration) - better for visualization"""
    rows = []
    
    for result in results:
        # Skip-gram row
        rows.append({
            'epochs': result['epochs'],
            'embed_dim': result['embed_dim'],
            'model': 'Skip-gram',
            'semantic_accuracy': result.get('skipgram_semantic_accuracy'),
            'syntactic_accuracy': result.get('skipgram_syntactic_accuracy'),
            'total_accuracy': result.get('skipgram_total_accuracy'),
            'training_time_seconds': result.get('skipgram_training_time_seconds'),
            'time_per_epoch_seconds': result.get('skipgram_time_per_epoch_seconds'),
            'vocab_size': result.get('vocab_size'),
            'total_words': result.get('total_words'),
            'sentences': result.get('sentences'),
        })
        
        # CBOW row
        rows.append({
            'epochs': result['epochs'],
            'embed_dim': result['embed_dim'],
            'model': 'CBOW',
            'semantic_accuracy': result.get('cbow_semantic_accuracy'),
            'syntactic_accuracy': result.get('cbow_syntactic_accuracy'),
            'total_accuracy': result.get('cbow_total_accuracy'),
            'training_time_seconds': result.get('cbow_training_time_seconds'),
            'time_per_epoch_seconds': result.get('cbow_time_per_epoch_seconds'),
            'vocab_size': result.get('vocab_size'),
            'total_words': result.get('total_words'),
            'sentences': result.get('sentences'),
        })
    
    fieldnames = [
        'epochs',
        'embed_dim',
        'model',
        'semantic_accuracy',
        'syntactic_accuracy',
        'total_accuracy',
        'training_time_seconds',
        'time_per_epoch_seconds',
        'vocab_size',
        'total_words',
        'sentences',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Created long format CSV: {output_path}")


def main():
    base_dir = Path(__file__).parent
    print(f"Parsing results from: {base_dir}")
    
    results = parse_all_files(str(base_dir))
    
    print(f"\nFound {len(results)} result files:")
    for r in results:
        print(f"  - {r['epochs']} epoch, dim {r['embed_dim']}")
    
    # Create both formats
    wide_output = base_dir / 'HS_results_wide.csv'
    long_output = base_dir / 'HS_results_long.csv'
    
    create_csv_wide_format(results, str(wide_output))
    create_csv_long_format(results, str(long_output))
    
    print(f"\n✓ CSV files created successfully!")
    print(f"  - Wide format (one row per config): {wide_output.name}")
    print(f"  - Long format (one row per model): {long_output.name}")


if __name__ == '__main__':
    main()

