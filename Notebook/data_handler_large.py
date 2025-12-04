# Copyright 2024 Word2Vec Implementation
# Large dataset download utilities - combining multiple sources for ~1B+ words
# Similar to word2vec-master/demo-train-big-model-v1.sh

import os
import pathlib
import re
import time
import zipfile
import gzip
import tarfile
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import requests
import tqdm

from data_handler import clean_text_remove_punctuation, preprocess_wmt14_news


def download_wmt14_multi_year(output_dir: str = "./data", years: List[int] = None) -> List[str]:
    """
    Download multiple years of WMT14 News Crawl datasets.
    
    Args:
        output_dir: Output directory for datasets
        years: List of years to download (default: [2012, 2013, 2014, 2015, 2016])
    
    Returns:
        List of paths to downloaded news files
    """
    if years is None:
        years = [2012, 2013, 2014, 2015, 2016]
    
    output_path = os.path.join(output_dir, "wmt14_multi")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    base_url = "http://www.statmt.org/wmt14/training-monolingual-news-crawl"
    
    for year in years:
        train_file = f"news.{year}.en.shuffled"
        train_gz = f"{train_file}.gz"
        train_url = f"{base_url}/{train_gz}"
        
        news_file = os.path.join(output_path, train_file)
        gz_path = os.path.join(output_path, train_gz)
        
        # Check if already exists
        if os.path.isfile(news_file):
            print(f"WMT14 News {year} already exists at: {news_file}")
            downloaded_files.append(news_file)
            continue
        
        # Download if missing
        if not os.path.isfile(gz_path):
            print(f"Downloading WMT14 News {year} ({train_gz})...")
            try:
                with requests.get(train_url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(gz_path, 'wb') as f:
                        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, 
                                     desc=f"Downloading {year}") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Warning: Could not download {train_url}: {e}")
                print(f"   Skipping year {year}")
                continue
        
        # Extract if needed
        if os.path.isfile(gz_path) and not os.path.isfile(news_file):
            print(f"Extracting {gz_path}...")
            try:
                with gzip.open(gz_path, "rb") as source, open(news_file, "wb") as target:
                    target.write(source.read())
                # Keep gz file for now (can be removed later to save space)
                downloaded_files.append(news_file)
                print(f"✓ Extracted {train_file}")
            except Exception as e:
                print(f"⚠️  Error extracting {gz_path}: {e}")
                continue
    
    print(f"\n✓ Downloaded {len(downloaded_files)} WMT14 News files")
    return downloaded_files


def download_1billion_word_benchmark(output_dir: str = "./data") -> Optional[str]:
    """
    Download 1-billion-word-language-modeling-benchmark dataset.
    This is a large dataset with ~1 billion words.
    
    Args:
        output_dir: Output directory
    
    Returns:
        Path to extracted dataset directory, or None if failed
    """
    output_path = os.path.join(output_dir, "1billion_word")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    extracted_dir = os.path.join(output_path, "1-billion-word-language-modeling-benchmark-r13output")
    if os.path.isdir(extracted_dir):
        print(f"1-billion-word dataset already extracted at: {extracted_dir}")
        return extracted_dir
    
    tar_file = "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    tar_path = os.path.join(output_path, tar_file)
    url = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    
    # Download if missing
    if not os.path.isfile(tar_path):
        print(f"Downloading 1-billion-word benchmark dataset...")
        print(f"⚠️  WARNING: This is a very large file (~1.6GB compressed, ~6.8GB uncompressed)")
        print(f"   It may take a long time to download and extract.")
        
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(tar_path, 'wb') as f:
                    with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, 
                                 desc="Downloading 1B benchmark") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error downloading 1-billion-word dataset: {e}")
            return None
    
    # Extract if needed
    if os.path.isfile(tar_path) and not os.path.isdir(extracted_dir):
        print(f"Extracting 1-billion-word dataset (this may take a while)...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(output_path)
            print(f"✓ Extracted to {extracted_dir}")
            return extracted_dir
        except Exception as e:
            print(f"⚠️  Error extracting {tar_path}: {e}")
            return None
    
    return extracted_dir if os.path.isdir(extracted_dir) else None


def combine_datasets_for_preprocessing(news_files: List[str], 
                                      benchmark_dir: Optional[str] = None,
                                      output_combined_file: str = "./data/combined_large.txt") -> str:
    """
    Combine multiple dataset files into one large file for preprocessing.
    This creates a single combined file that can be processed like a single WMT14 file.
    
    Args:
        news_files: List of WMT14 news file paths
        benchmark_dir: Path to 1-billion-word benchmark directory (optional)
        output_combined_file: Path to output combined file
    
    Returns:
        Path to combined file
    """
    print(f"\nCombining datasets into: {output_combined_file}")
    print(f"  WMT14 files: {len(news_files)}")
    if benchmark_dir:
        print(f"  1-billion-word benchmark: {benchmark_dir}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_combined_file), exist_ok=True)
    
    total_lines = 0
    
    with open(output_combined_file, 'w', encoding='utf-8') as outfile:
        # Combine WMT14 files
        for i, news_file in enumerate(news_files):
            if not os.path.isfile(news_file):
                print(f"⚠️  Warning: {news_file} not found, skipping")
                continue
            
            print(f"  Adding WMT14 file {i+1}/{len(news_files)}: {os.path.basename(news_file)}")
            line_count = 0
            
            with open(news_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Basic cleaning - remove empty lines
                    cleaned = line.strip()
                    if cleaned:
                        outfile.write(cleaned + '\n')
                        line_count += 1
                        total_lines += 1
            
            print(f"    → Added {line_count:,} lines")
        
        # Add 1-billion-word benchmark if provided
        if benchmark_dir and os.path.isdir(benchmark_dir):
            benchmark_subdir = os.path.join(benchmark_dir, 
                                          "training-monolingual.tokenized.shuffled")
            if os.path.isdir(benchmark_subdir):
                print(f"\n  Adding 1-billion-word benchmark files...")
                benchmark_files = sorted([f for f in os.listdir(benchmark_subdir) 
                                        if f.endswith('.txt')])
                
                for i, filename in enumerate(benchmark_files):
                    filepath = os.path.join(benchmark_subdir, filename)
                    print(f"    Adding benchmark file {i+1}/{len(benchmark_files)}: {filename}")
                    
                    line_count = 0
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            cleaned = line.strip()
                            if cleaned:
                                outfile.write(cleaned + '\n')
                                line_count += 1
                                total_lines += 1
                    
                    print(f"      → Added {line_count:,} lines")
    
    # Get file size
    file_size = os.path.getsize(output_combined_file) / (1024**3)  # GB
    
    print(f"\n✓ Combined dataset created:")
    print(f"  File: {output_combined_file}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Size: {file_size:.2f} GB")
    print(f"\n💡 Estimated words: ~{total_lines * 20:,} (assuming ~20 words/line)")
    
    return output_combined_file


def download_large_dataset(output_dir: str = "./data", 
                          wmt14_years: List[int] = None,
                          include_1billion: bool = True) -> str:
    """
    Download and combine multiple large datasets to create a ~1B+ word corpus.
    
    This function:
    1. Downloads multiple years of WMT14 News (2012-2016)
    2. Optionally downloads 1-billion-word benchmark
    3. Combines them into a single file for preprocessing
    
    Args:
        output_dir: Output directory
        wmt14_years: List of years to download (default: [2012, 2013, 2014])
        include_1billion: Whether to include 1-billion-word benchmark
    
    Returns:
        Path to combined dataset file
    """
    if wmt14_years is None:
        wmt14_years = [2012, 2013, 2014]  # Start with 3 years for ~800M words
    
    print("="*60)
    print("  DOWNLOADING LARGE DATASET (~1B+ WORDS)")
    print("="*60)
    print(f"\nThis will download multiple datasets and combine them.")
    print(f"Estimated total size: ~3-5 GB compressed, ~10-15 GB uncompressed")
    print(f"Estimated words: ~800M-1.6B words\n")
    
    # Download WMT14 multiple years
    print("\n📥 STEP 1: Downloading WMT14 News (multiple years)")
    print(f"  Years: {wmt14_years}")
    wmt14_files = download_wmt14_multi_year(output_dir, years=wmt14_years)
    
    # Download 1-billion-word benchmark (optional)
    benchmark_dir = None
    if include_1billion:
        print("\n📥 STEP 2: Downloading 1-billion-word benchmark")
        benchmark_dir = download_1billion_word_benchmark(output_dir)
    
    # Combine datasets
    print("\n📦 STEP 3: Combining datasets")
    combined_file = combine_datasets_for_preprocessing(
        wmt14_files,
        benchmark_dir=benchmark_dir,
        output_combined_file=os.path.join(output_dir, "combined_large.txt")
    )
    
    print("\n" + "="*60)
    print("  ✅ LARGE DATASET READY")
    print("="*60)
    print(f"\nCombined dataset: {combined_file}")
    print(f"\nNext steps:")
    print(f"  1. Preprocess using: preprocess_wmt14_news('{combined_file}', output_dir)")
    print(f"  2. This will create a dataset similar to Google News corpus size")
    
    return combined_file


if __name__ == "__main__":
    # Example usage
    combined = download_large_dataset(
        output_dir="./data",
        wmt14_years=[2012, 2013, 2014],  # ~800M words
        include_1billion=False  # Set to True to add ~1B more words
    )
    print(f"\n✓ Large dataset ready at: {combined}")

