#!/usr/bin/env python3
"""
Parse NS result files and create CSV for visualization
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
    
    # Extract vocabulary size - try multiple formats
    vocab_match = re.search(r'Vocab size:\s*([\d,]+)', content)
    if vocab_match:
        result['vocab_size'] = int(vocab_match.group(1).replace(',', ''))
    else:
        vocab_match = re.search(r'vocab size\s+([\d,]+)', content, re.IGNORECASE)
        if vocab_match:
            result['vocab_size'] = int(vocab_match.group(1).replace(',', ''))
        else:
            # Try format from FINAL SUMMARY: " -Vocabulary: 520,233"
            vocab_match = re.search(r' -Vocabulary:\s+([\d,]+)', content)
            if vocab_match:
                result['vocab_size'] = int(vocab_match.group(1).replace(',', ''))
            else:
                result['vocab_size'] = None
    
    # Extract total words and sentences - try multiple formats
    # Format 1: "Data loaded: 41,051,876 sentences, 800,000,000 total words"
    # Format 2: From FINAL SUMMARY " -Words: 800,000,000" and " -Sentences: 41,051,876"
    
    data_loaded_match = re.search(r'Data loaded:\s*([\d,]+)\s+sentences,\s*([\d,]+)\s+total words', content, re.IGNORECASE)
    if data_loaded_match:
        result['sentences'] = int(data_loaded_match.group(1).replace(',', ''))
        result['total_words'] = int(data_loaded_match.group(2).replace(',', ''))
    else:
        # Try format from FINAL SUMMARY
        data_processed_section = re.search(r'Data Processed:.*?(?=Output|$)', content, re.DOTALL | re.IGNORECASE)
        if data_processed_section:
            words_match = re.search(r' -Words:\s+([\d,]+)', data_processed_section.group(0))
            sentences_match = re.search(r' -Sentences:\s+([\d,]+)', data_processed_section.group(0))
            
            if words_match:
                result['total_words'] = int(words_match.group(1).replace(',', ''))
            else:
                result['total_words'] = None
            
            if sentences_match:
                result['sentences'] = int(sentences_match.group(1).replace(',', ''))
            else:
                result['sentences'] = None
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
    
    # Extract Skip-gram metrics - try both formats
    # Format 1 (old): "  Skip-gram semantic:  0.5712" or "Skip-gram semantic:  0.5712"
    # Format 2 (new): "Skip-gram accuracy: 0.6534" followed by " -Semantic:  0.6293" and " -Syntactic: 0.6604"
    
    # Try old format first
    sg_sem_match = re.search(r'Skip-gram semantic:\s+([\d.]+)', content)
    sg_syn_match = re.search(r'Skip-gram syntactic:\s+([\d.]+)', content)
    sg_total_match = re.search(r'Skip-gram total:\s+([\d.]+)', content)
    
    # Check if we have complete old format data
    has_old_format = sg_sem_match and sg_syn_match and sg_total_match
    
    # If old format incomplete, try new format from FINAL SUMMARY
    if not has_old_format:
        # Look for "Skip-gram accuracy:" followed by lines with " -Semantic:" and " -Syntactic:"
        # Search specifically in FINAL SUMMARY section
        final_summary = re.search(r'FINAL SUMMARY.*?(?=Output Files|$)', content, re.DOTALL | re.IGNORECASE)
        if final_summary:
            summary_text = final_summary.group(0)
            # Find Skip-gram section - from "Skip-gram accuracy:" to just before "CBOW accuracy:"
            sg_acc_section = re.search(r'Skip-gram accuracy:\s+([\d.]+).*?(?=CBOW accuracy|Difference|Training Times|$)', summary_text, re.DOTALL)
            if sg_acc_section:
                section_text = sg_acc_section.group(0)
                # Extract total accuracy (from the first line)
                sg_total_match_new = re.search(r'Skip-gram accuracy:\s+([\d.]+)', section_text)
                # Extract semantic and syntactic (with leading space and dash)
                sg_sem_match_new = re.search(r' -Semantic:\s+([\d.]+)', section_text)
                sg_syn_match_new = re.search(r' -Syntactic:\s+([\d.]+)', section_text)
                
                # Use new format if found - replace old matches
                if sg_total_match_new:
                    sg_total_match = sg_total_match_new
                if sg_sem_match_new:
                    sg_sem_match = sg_sem_match_new
                if sg_syn_match_new:
                    sg_syn_match = sg_syn_match_new
    
    result['skipgram_semantic_accuracy'] = float(sg_sem_match.group(1)) if sg_sem_match else None
    result['skipgram_syntactic_accuracy'] = float(sg_syn_match.group(1)) if sg_syn_match else None
    result['skipgram_total_accuracy'] = float(sg_total_match.group(1)) if sg_total_match else None
    
    # Extract CBOW metrics - try both formats
    # Format 1 (old): "  CBOW semantic:  0.4764" or "CBOW semantic:  0.4764"
    # Format 2 (new): "CBOW accuracy: 0.5686" followed by " -Semantic:  0.4635" and " -Syntactic: 0.5991"
    
    # Try old format first
    cbow_sem_match = re.search(r'CBOW semantic:\s+([\d.]+)', content)
    cbow_syn_match = re.search(r'CBOW syntactic:\s+([\d.]+)', content)
    cbow_total_match = re.search(r'CBOW total:\s+([\d.]+)', content)
    
    # Check if we have complete old format data
    has_old_format = cbow_sem_match and cbow_syn_match and cbow_total_match
    
    # If old format incomplete, try new format from FINAL SUMMARY
    if not has_old_format:
        # Look for "CBOW accuracy:" followed by lines with " -Semantic:" and " -Syntactic:"
        # Search specifically in FINAL SUMMARY section
        final_summary = re.search(r'FINAL SUMMARY.*?(?=Output Files|$)', content, re.DOTALL | re.IGNORECASE)
        if final_summary:
            summary_text = final_summary.group(0)
            # Find CBOW section - from "CBOW accuracy:" to just before "Difference" or "Training Times"
            cbow_acc_section = re.search(r'CBOW accuracy:\s+([\d.]+).*?(?=Difference|Training Times|Data Processed|$)', summary_text, re.DOTALL)
            if cbow_acc_section:
                section_text = cbow_acc_section.group(0)
                # Extract total accuracy (from the first line)
                cbow_total_match_new = re.search(r'CBOW accuracy:\s+([\d.]+)', section_text)
                # Extract semantic and syntactic (with leading space and dash)
                cbow_sem_match_new = re.search(r' -Semantic:\s+([\d.]+)', section_text)
                cbow_syn_match_new = re.search(r' -Syntactic:\s+([\d.]+)', section_text)
                
                # Use new format if found
                if cbow_total_match_new:
                    cbow_total_match = cbow_total_match_new
                if cbow_sem_match_new:
                    cbow_sem_match = cbow_sem_match_new
                if cbow_syn_match_new:
                    cbow_syn_match = cbow_syn_match_new
    
    result['cbow_semantic_accuracy'] = float(cbow_sem_match.group(1)) if cbow_sem_match else None
    result['cbow_syntactic_accuracy'] = float(cbow_syn_match.group(1)) if cbow_syn_match else None
    result['cbow_total_accuracy'] = float(cbow_total_match.group(1)) if cbow_total_match else None
    
    # Extract Skip-gram training time - try both formats
    # Format 1: "Total training time: 1474.73s" in training section
    # Format 2: " -Skip-gram: 1474.73s" in FINAL SUMMARY
    sg_time_match = None
    
    # Try format 1: in training section
    sg_section = re.search(r'Skip-gram training completed!.*?(?=CBOW|$)', content, re.DOTALL | re.IGNORECASE)
    if sg_section:
        sg_time_match = re.search(r'Total training time:\s+([\d.]+)s', sg_section.group(0))
    
    # Try format 2: in FINAL SUMMARY
    if not sg_time_match:
        final_summary = re.search(r'Training Times:.*?(?=Data|$)', content, re.DOTALL | re.IGNORECASE)
        if final_summary:
            sg_time_match = re.search(r' -Skip-gram:\s+([\d.]+)s', final_summary.group(0))
    
    result['skipgram_training_time_seconds'] = float(sg_time_match.group(1)) if sg_time_match else None
    
    # Extract CBOW training time - try both formats
    # Format 1: "Total training time: 818.85s" in training section
    # Format 2: " -CBOW: 818.85s" in FINAL SUMMARY
    cbow_time_match = None
    
    # Try format 1: in training section
    cbow_section = re.search(r'CBOW training completed!.*?(?=EVALUATING|$)', content, re.DOTALL | re.IGNORECASE)
    if cbow_section:
        cbow_time_match = re.search(r'Total training time:\s+([\d.]+)s', cbow_section.group(0))
    
    # Try format 2: in FINAL SUMMARY
    if not cbow_time_match:
        final_summary = re.search(r'Training Times:.*?(?=Data|$)', content, re.DOTALL | re.IGNORECASE)
        if final_summary:
            cbow_time_match = re.search(r' -CBOW:\s+([\d.]+)s', final_summary.group(0))
    
    result['cbow_training_time_seconds'] = float(cbow_time_match.group(1)) if cbow_time_match else None
    
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
    """Parse all result files in the directory"""
    base_path = Path(base_dir)
    all_results = []
    
    # NS files are directly in the directory, not in subdirectories
    for file_path in sorted(base_path.glob('NS*.txt')):
        result = parse_file(str(file_path))
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
    wide_output = base_dir / 'NS_results_wide.csv'
    long_output = base_dir / 'NS_results_long.csv'
    
    create_csv_wide_format(results, str(wide_output))
    create_csv_long_format(results, str(long_output))
    
    print(f"\n✓ CSV files created successfully!")
    print(f"  - Wide format (one row per config): {wide_output.name}")
    print(f"  - Long format (one row per model): {long_output.name}")


if __name__ == '__main__':
    main()

