#!/usr/bin/env python3
"""
Parse "Most similar words" sections from HS result files and create CSV for similarity heatmap
"""

import re
import csv
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_similarity_section(content: str, model_type: str) -> List[Dict]:
    """
    Parse "Most similar words" section from content.
    Returns list of dictionaries with word, similar_word, similarity_score, model_type
    """
    results = []
    
    # Find the "Most similar words:" section
    start_marker = "Most similar words:"
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return results
    
    # Find where this section ends (next section or end of file)
    section_content = content[start_idx:]
    
    # Pattern to match word sections
    # Example:
    # king:
    #   word1: 0.1234
    #   word2: 0.5678
    # queen:
    #   word3: 0.1234
    
    # Split by word headers (word followed by colon on its own line)
    word_pattern = r'^(\w+):\s*$'
    
    lines = section_content.split('\n')
    current_word = None
    
    for line in lines:
        line = line.strip()
        
        # Check if this is a word header (word:)
        word_match = re.match(r'^(\w+):\s*$', line)
        if word_match:
            current_word = word_match.group(1)
            continue
        
        # Check if this is a similarity entry (  word: 0.1234)
        if current_word and line:
            similarity_match = re.match(r'^\s*(\S+):\s+([\d.]+)\s*$', line)
            if similarity_match:
                similar_word = similarity_match.group(1)
                similarity_score = float(similarity_match.group(2))
                
                results.append({
                    'test_word': current_word,
                    'similar_word': similar_word,
                    'similarity_score': similarity_score,
                    'model_type': model_type
                })
            # Stop if we hit a section break or empty line that's not part of a word list
            elif not line.startswith('  ') and line:
                # Check if this is a new section (like "Word pair similarities:")
                if ':' in line and not re.match(r'^\s+\w+:', line):
                    break
    
    return results


def parse_file_similarity(file_path: str) -> Dict:
    """Parse a single result file and extract similarity data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract epoch and dimension from filename
    filename = os.path.basename(file_path)
    epoch_match = re.search(r'(\d+)\s+epoch', filename, re.IGNORECASE)
    dim_match = re.search(r'dim\s+(\d+)', filename, re.IGNORECASE)
    
    epochs = int(epoch_match.group(1)) if epoch_match else None
    embed_dim = int(dim_match.group(1)) if dim_match else None
    
    # Find Skip-gram similarity section
    sg_start = content.find('STEP 5: EVALUATING SKIP-GRAM MODEL')
    sg_end = content.find('STEP 6: EVALUATING CBOW MODEL')
    
    sg_similarity = []
    if sg_start != -1 and sg_end != -1:
        sg_section = content[sg_start:sg_end]
        if 'Most similar words:' in sg_section:
            sg_similarity = parse_similarity_section(sg_section, 'Skip-gram')
    
    # Find CBOW similarity section
    cbow_start = content.find('STEP 6: EVALUATING CBOW MODEL')
    cbow_end = content.find('STEP 7:', cbow_start)
    if cbow_end == -1:
        cbow_end = len(content)
    
    cbow_similarity = []
    if cbow_start != -1:
        cbow_section = content[cbow_start:cbow_end]
        if 'Most similar words:' in cbow_section:
            cbow_similarity = parse_similarity_section(cbow_section, 'CBOW')
    
    return {
        'epochs': epochs,
        'embed_dim': embed_dim,
        'skipgram_similarity': sg_similarity,
        'cbow_similarity': cbow_similarity
    }


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
            result = parse_file_similarity(str(file_path))
            result['epochs'] = epoch_num  # Override with directory epoch
            all_results.append(result)
    
    return all_results


def create_similarity_csv(results: List[Dict], output_path: str):
    """Create CSV file with similarity data for heatmap"""
    rows = []
    
    for result in results:
        epochs = result['epochs']
        embed_dim = result['embed_dim']
        
        # Add Skip-gram similarities
        for sim in result.get('skipgram_similarity', []):
            rows.append({
                'epochs': epochs,
                'embed_dim': embed_dim,
                'model': 'Skip-gram',
                'test_word': sim['test_word'],
                'similar_word': sim['similar_word'],
                'similarity_score': sim['similarity_score'],
            })
        
        # Add CBOW similarities
        for sim in result.get('cbow_similarity', []):
            rows.append({
                'epochs': epochs,
                'embed_dim': embed_dim,
                'model': 'CBOW',
                'test_word': sim['test_word'],
                'similar_word': sim['similar_word'],
                'similarity_score': sim['similarity_score'],
            })
    
    fieldnames = [
        'epochs',
        'embed_dim',
        'model',
        'test_word',
        'similar_word',
        'similarity_score',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Created similarity CSV: {output_path}")
    print(f"  Total rows: {len(rows)}")


def create_similarity_matrix_csv(results: List[Dict], output_path: str):
    """
    Create a matrix-style CSV where each row is a configuration,
    and columns are test_word + similar_word combinations.
    Better for creating heatmaps directly.
    """
    # Collect all unique test_word + similar_word combinations
    all_pairs = set()
    for result in results:
        for sim in result.get('skipgram_similarity', []) + result.get('cbow_similarity', []):
            pair = f"{sim['test_word']}_{sim['similar_word']}"
            all_pairs.add(pair)
    
    all_pairs = sorted(all_pairs)
    
    # Build matrix
    rows = []
    for result in results:
        epochs = result['epochs']
        embed_dim = result['embed_dim']
        
        # Create row for Skip-gram
        sg_row = {
            'epochs': epochs,
            'embed_dim': embed_dim,
            'model': 'Skip-gram'
        }
        
        # Create a dictionary for quick lookup
        sg_dict = {}
        for sim in result.get('skipgram_similarity', []):
            pair = f"{sim['test_word']}_{sim['similar_word']}"
            sg_dict[pair] = sim['similarity_score']
        
        for pair in all_pairs:
            sg_row[pair] = sg_dict.get(pair, None)
        
        rows.append(sg_row)
        
        # Create row for CBOW
        cbow_row = {
            'epochs': epochs,
            'embed_dim': embed_dim,
            'model': 'CBOW'
        }
        
        cbow_dict = {}
        for sim in result.get('cbow_similarity', []):
            pair = f"{sim['test_word']}_{sim['similar_word']}"
            cbow_dict[pair] = sim['similarity_score']
        
        for pair in all_pairs:
            cbow_row[pair] = cbow_dict.get(pair, None)
        
        rows.append(cbow_row)
    
    fieldnames = ['epochs', 'embed_dim', 'model'] + all_pairs
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Created similarity matrix CSV: {output_path}")
    print(f"  Total configurations: {len(rows)}")
    print(f"  Total similarity pairs: {len(all_pairs)}")


def create_word_focused_csv(results: List[Dict], output_path: str):
    """
    Create CSV organized by test words - better for analyzing specific words across configurations
    """
    rows = []
    
    for result in results:
        epochs = result['epochs']
        embed_dim = result['embed_dim']
        
        # Group by test word
        test_words = {}
        
        for sim in result.get('skipgram_similarity', []) + result.get('cbow_similarity', []):
            test_word = sim['test_word']
            if test_word not in test_words:
                test_words[test_word] = {'skipgram': [], 'cbow': []}
        
        for sim in result.get('skipgram_similarity', []):
            test_words[sim['test_word']]['skipgram'].append((sim['similar_word'], sim['similarity_score']))
        
        for sim in result.get('cbow_similarity', []):
            test_words[sim['test_word']]['cbow'].append((sim['similar_word'], sim['similarity_score']))
        
        # Create rows - one per test word
        for test_word, similarities in test_words.items():
            # Sort by similarity score (descending)
            sg_sorted = sorted(similarities['skipgram'], key=lambda x: x[1], reverse=True)[:5]
            cbow_sorted = sorted(similarities['cbow'], key=lambda x: x[1], reverse=True)[:5]
            
            # Create row with top 5 for each model
            row = {
                'epochs': epochs,
                'embed_dim': embed_dim,
                'test_word': test_word,
            }
            
            # Add Skip-gram top 5
            for i in range(5):
                if i < len(sg_sorted):
                    row[f'sg_word_{i+1}'] = sg_sorted[i][0]
                    row[f'sg_score_{i+1}'] = sg_sorted[i][1]
                else:
                    row[f'sg_word_{i+1}'] = None
                    row[f'sg_score_{i+1}'] = None
            
            # Add CBOW top 5
            for i in range(5):
                if i < len(cbow_sorted):
                    row[f'cbow_word_{i+1}'] = cbow_sorted[i][0]
                    row[f'cbow_score_{i+1}'] = cbow_sorted[i][1]
                else:
                    row[f'cbow_word_{i+1}'] = None
                    row[f'cbow_score_{i+1}'] = None
            
            rows.append(row)
    
    fieldnames = ['epochs', 'embed_dim', 'test_word'] + \
                 [f'sg_word_{i+1}' for i in range(5)] + \
                 [f'sg_score_{i+1}' for i in range(5)] + \
                 [f'cbow_word_{i+1}' for i in range(5)] + \
                 [f'cbow_score_{i+1}' for i in range(5)]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Created word-focused CSV: {output_path}")
    print(f"  Total rows: {len(rows)}")


def main():
    base_dir = Path(__file__).parent
    print(f"Parsing similarity data from: {base_dir}")
    
    results = parse_all_files(str(base_dir))
    
    print(f"\nFound {len(results)} result files:")
    for r in results:
        sg_count = len(r.get('skipgram_similarity', []))
        cbow_count = len(r.get('cbow_similarity', []))
        print(f"  - {r['epochs']} epoch, dim {r['embed_dim']}: {sg_count} SG pairs, {cbow_count} CBOW pairs")
    
    # Create multiple formats
    similarity_csv = base_dir / 'HS_similarity_long.csv'
    matrix_csv = base_dir / 'HS_similarity_matrix.csv'
    word_focused_csv = base_dir / 'HS_similarity_by_word.csv'
    
    create_similarity_csv(results, str(similarity_csv))
    create_word_focused_csv(results, str(word_focused_csv))
    
    # Matrix format might be too wide, create it but warn user
    try:
        create_similarity_matrix_csv(results, str(matrix_csv))
    except Exception as e:
        print(f"⚠️  Could not create matrix CSV (might be too wide): {e}")
    
    print(f"\n✓ CSV files created successfully!")
    print(f"  - Long format (recommended for heatmap): {similarity_csv.name}")
    print(f"  - Word-focused format: {word_focused_csv.name}")
    if matrix_csv.exists():
        print(f"  - Matrix format: {matrix_csv.name}")


if __name__ == '__main__':
    main()

