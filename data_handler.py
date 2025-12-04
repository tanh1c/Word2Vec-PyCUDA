# Copyright 2024 Word2Vec Implementation
# Data handling utilities for text8 dataset

import os
import pathlib
import re
import time
import zipfile
import gzip
import json
from typing import List, Tuple, Dict
from collections import defaultdict
import requests
import tqdm
from gensim.scripts import segment_wiki
from gensim import utils


def clean_text_remove_punctuation(text: str) -> str:
    """
    Clean text by removing punctuation and normalizing whitespace.
    Similar to word2vec preprocessing - only keeps letters and spaces.
    
    Args:
        text: Input text line
    
    Returns:
        Cleaned text with only lowercase letters and spaces
    """
    if not text:
        return ""
    
    # Replace tabs and newlines with spaces
    text = re.sub(r'[\t\n]', ' ', text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r'[ ]{2,}', ' ', text)
    
    # Remove all punctuation, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z ]', '', text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    return text


def detect_phrases(text: str, word_counts: Dict[str, int], bigram_counts: Dict[Tuple[str, str], int], 
                   train_words: int, min_count: int = 5, threshold: float = 100.0) -> str:
    """
    Detect and combine phrases in text based on bigram scores.
    Based on word2phrase.c TrainModel() function.
    
    Args:
        text: Input text (space-separated words)
        word_counts: Dictionary mapping words to their counts
        bigram_counts: Dictionary mapping (word1, word2) tuples to bigram counts
        train_words: Total number of words in training data
        min_count: Minimum word count threshold
        threshold: Score threshold for phrase formation (higher = fewer phrases)
    
    Returns:
        Text with phrases combined (e.g., "new york" -> "new_york")
    """
    words = text.split()
    if len(words) < 2:
        return text
    
    result = []
    i = 0
    while i < len(words):
        if i == len(words) - 1:
            # Last word, no bigram possible
            result.append(words[i])
            break
        
        word1 = words[i]
        word2 = words[i + 1]
        
        # Check if both words meet min_count
        count1 = word_counts.get(word1, 0)
        count2 = word_counts.get(word2, 0)
        
        if count1 < min_count or count2 < min_count:
            # One word doesn't meet threshold, keep as separate
            result.append(word1)
            i += 1
            continue
        
        # Calculate bigram score
        bigram = (word1, word2)
        count_bigram = bigram_counts.get(bigram, 0)
        
        if count_bigram == 0:
            # Bigram not found, keep as separate
            result.append(word1)
            i += 1
            continue
        
        # Score formula from word2phrase.c line 285
        # score = (pab - min_count) / pa / pb * train_words
        score = (count_bigram - min_count) / count1 / count2 * train_words
        
        if score > threshold:
            # Combine into phrase
            result.append(f"{word1}_{word2}")
            i += 2  # Skip both words
        else:
            # Keep as separate
            result.append(word1)
            i += 1
    
    return " ".join(result)


def learn_phrase_vocab(data_path: str, min_count: int = 5) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], int]:
    """
    Learn vocabulary and bigram counts from training data.
    Based on word2phrase.c LearnVocabFromTrainFile() function.
    
    Args:
        data_path: Path to training data directory
        min_count: Minimum word count threshold
    
    Returns:
        Tuple of (word_counts, bigram_counts, total_words)
    """
    word_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    total_words = 0
    
    # Get all data files
    data_files = [f for f in os.listdir(data_path) if f.startswith("0")]
    data_files.sort()
    
    print(f"Learning phrase vocabulary from {len(data_files)} files...")
    
    for file_idx, filename in enumerate(data_files):
        filepath = os.path.join(data_path, filename)
        last_word = None
        start = True
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    start = True
                    last_word = None
                    continue
                
                words = line.split()
                for word in words:
                    word = word.lower().strip()
                    if not word:
                        continue
                    
                    total_words += 1
                    
                    # Count unigram
                    word_counts[word] += 1
                    
                    # Count bigram (if not at start of sentence)
                    if not start and last_word:
                        bigram = (last_word, word)
                        bigram_counts[bigram] += 1
                    
                    last_word = word
                    start = False
                
                # Reset at end of line
                start = True
                last_word = None
        
        if (file_idx + 1) % 10 == 0:
            print(f"  Processed {file_idx + 1}/{len(data_files)} files...")
    
    # Filter words below min_count
    filtered_word_counts = {w: c for w, c in word_counts.items() if c >= min_count}
    
    print(f"Vocabulary: {len(filtered_word_counts):,} words (min_count={min_count})")
    print(f"Bigrams: {len(bigram_counts):,} unique bigrams")
    print(f"Total words: {total_words:,}")
    
    return filtered_word_counts, bigram_counts, total_words


def apply_phrases_to_data(data_path: str, output_path: str, word_counts: Dict[str, int], 
                          bigram_counts: Dict[Tuple[str, str], int], train_words: int,
                          min_count: int = 5, threshold: float = 100.0) -> str:
    """
    Apply phrase detection to all data files.
    
    Args:
        data_path: Input data directory
        output_path: Output data directory
        word_counts: Word count dictionary
        bigram_counts: Bigram count dictionary
        train_words: Total number of words
        min_count: Minimum word count
        threshold: Phrase score threshold
    
    Returns:
        Path to output directory
    """
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    data_files = [f for f in os.listdir(data_path) if f.startswith("0")]
    data_files.sort()
    
    print(f"Applying phrase detection (threshold={threshold}) to {len(data_files)} files...")
    
    for file_idx, filename in enumerate(data_files):
        input_filepath = os.path.join(data_path, filename)
        output_filepath = os.path.join(output_path, filename)
        
        with open(input_filepath, 'r', encoding='utf-8') as fin, \
             open(output_filepath, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write('\n')
                    continue
                
                # Apply phrase detection
                processed_line = detect_phrases(line, word_counts, bigram_counts, 
                                                train_words, min_count, threshold)
                fout.write(processed_line + '\n')
        
        if (file_idx + 1) % 10 == 0:
            print(f"  Processed {file_idx + 1}/{len(data_files)} files...")
    
    print(f"Phrase detection complete. Output: {output_path}")
    return output_path


def preprocess_with_phrases(data_path: str, output_path: str, min_count: int = 5, 
                            threshold1: float = 200.0, threshold2: float = 100.0) -> str:
    """
    Preprocess data with phrase detection (2 passes, like word2phrase).
    
    Args:
        data_path: Input data directory
        output_path: Final output directory
        min_count: Minimum word count
        threshold1: First pass threshold (higher, fewer phrases)
        threshold2: Second pass threshold (lower, more phrases)
    
    Returns:
        Path to final output directory
    """
    print(f"Preprocessing with phrase detection...")
    print(f"  Input: {data_path}")
    print(f"  Output: {output_path}")
    print(f"  Threshold 1: {threshold1} (first pass)")
    print(f"  Threshold 2: {threshold2} (second pass)")
    
    # Step 1: Learn vocabulary and bigram counts
    print("\nStep 1: Learning vocabulary and bigram counts...")
    word_counts, bigram_counts, train_words = learn_phrase_vocab(data_path, min_count)
    
    # Step 2: First pass (threshold1)
    print(f"\nStep 2: First pass phrase detection (threshold={threshold1})...")
    temp_path1 = output_path + "_phrase1"
    apply_phrases_to_data(data_path, temp_path1, word_counts, bigram_counts, 
                          train_words, min_count, threshold1)
    
    # Step 3: Relearn vocabulary from first pass
    print("\nStep 3: Relearning vocabulary from first pass...")
    word_counts2, bigram_counts2, train_words2 = learn_phrase_vocab(temp_path1, min_count)
    
    # Step 4: Second pass (threshold2)
    print(f"\nStep 4: Second pass phrase detection (threshold={threshold2})...")
    apply_phrases_to_data(temp_path1, output_path, word_counts2, bigram_counts2, 
                          train_words2, min_count, threshold2)
    
    # Cleanup temp directory
    import shutil
    if os.path.exists(temp_path1):
        shutil.rmtree(temp_path1)
        print(f"Cleaned up temporary directory: {temp_path1}")
    
    print(f"\nPhrase preprocessing complete: {output_path}")
    return output_path


def download_wmt14_news(output_dir: str = "./data") -> str:
    """
    Download and combine multiple years of WMT14 News Crawl dataset.
    Downloads years 2011, 2012, 2013 and combines them into a single file.
    Returns path to combined news file.
    """
    years = [2012, 2013]  # Download multiple years
    base_url = "http://www.statmt.org/wmt14/training-monolingual-news-crawl"
    
    output_path = os.path.join(output_dir, "wmt14")
    combined_file = os.path.join(output_path, "news.combined.en.shuffled")
    
    # Create output directory
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Check if combined file already exists
    if os.path.isfile(combined_file):
        print(f"WMT14 News combined file already exists at: {combined_file}")
        return combined_file
    
    # Download and extract each year
    downloaded_files = []
    for year in years:
        train_file = f"news.{year}.en.shuffled"
        train_gz = f"{train_file}.gz"
        train_url = f"{base_url}/{train_gz}"
        news_file = os.path.join(output_path, train_file)
        gz_path = os.path.join(output_path, train_gz)
        
        # Check if already extracted
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
                downloaded_files.append(news_file)
                print(f"✓ Extracted {train_file}")
                # Remove gz file to save space
                os.remove(gz_path)
            except Exception as e:
                print(f"⚠️  Error extracting {gz_path}: {e}")
                continue
    
    if not downloaded_files:
        raise FileNotFoundError("No WMT14 News files were successfully downloaded")
    
    # Combine all downloaded files into one
    print(f"\nCombining {len(downloaded_files)} WMT14 News files into: {combined_file}")
    total_lines = 0
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for i, news_file in enumerate(downloaded_files):
            if not os.path.isfile(news_file):
                print(f"⚠️  Warning: {news_file} not found, skipping")
                continue
            
            print(f"  Adding file {i+1}/{len(downloaded_files)}: {os.path.basename(news_file)}")
            line_count = 0
            
            with open(news_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    cleaned = line.strip()
                    if cleaned:  # Skip empty lines
                        outfile.write(cleaned + '\n')
                        line_count += 1
                        total_lines += 1
            
            print(f"    → Added {line_count:,} lines")
    
    # Get file size
    file_size = os.path.getsize(combined_file) / (1024**3)  # GB
    
    print(f"\n✓ Combined WMT14 News dataset created:")
    print(f"  File: {combined_file}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Size: {file_size:.2f} GB")
    print(f"  Estimated words: ~{total_lines * 20:,} (assuming ~20 words/line)")
    
    return combined_file


def download_wikipedia(output_dir: str = "./data") -> str:
    """
    Download Wikipedia dataset (similar to myw2v demo).
    Downloads a partial Wikipedia dump and processes it.
    """
    # Wikipedia dump URLs (partial dumps for demo)
    urls = [
        "https://dumps.wikimedia.org/enwiki/20210801/enwiki-20210801-pages-articles-multistream6.xml-p958046p1483661.bz2"
    ]
    filenames = [re.sub(r".*/([^/]+)$", r"\1", url) for url in urls]
    
    output_path = os.path.join(output_dir, "wikipedia")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Download files
    for i, (url, filename) in enumerate(zip(urls, filenames)):
        file_path = os.path.join(output_path, filename)
        if os.path.isfile(file_path):
            print(f"Wikipedia file {filename} already exists, skipping download")
        else:
            print(f"Downloading {filename} from {url}...")
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
    
    return output_path


def preprocess_wikipedia(wiki_dir: str, output_dir: str, words_per_sentence: int = 1000) -> str:
    """
    Preprocess Wikipedia XML dump into sentence files (similar to myw2v demo).
    """
    print(f"Preprocessing Wikipedia dump from: {wiki_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Words per sentence: {words_per_sentence}")
    
    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("0")]
    if existing_files:
        print(f"Found {len(existing_files)} existing processed files. Skipping preprocessing.")
        return output_dir
    
    # Find XML files
    xml_files = [f for f in os.listdir(wiki_dir) if f.endswith('.xml.bz2')]
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {wiki_dir}")
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process each XML file
    all_sentences = []
    for xml_file in xml_files:
        xml_path = os.path.join(wiki_dir, xml_file)
        json_file = xml_file.replace('.xml.bz2', '.json.gz')
        json_path = os.path.join(wiki_dir, json_file)
        
        # Convert XML to JSON if needed
        if not os.path.isfile(json_path):
            print(f"Converting {xml_file} to {json_file}...")
            segment_wiki.segment_and_write_all_articles(xml_path, json_path)
        
        # Process JSON file
        print(f"Processing {json_file}...")
        sentences = _process_wikipedia_json(json_path, words_per_sentence)
        all_sentences.extend(sentences)
        print(f"  Extracted {len(sentences):,} sentences from {json_file}")
    
    print(f"Total sentences: {len(all_sentences):,}")
    
    # Save to files (similar to myw2v format)
    sentences_per_file = 100000
    file_count = 0
    current_file_sentences = []
    
    for i, sentence in enumerate(all_sentences):
        current_file_sentences.append(sentence)
        
        # Write file when it reaches sentences_per_file or we're at the end
        if len(current_file_sentences) >= sentences_per_file or i == len(all_sentences) - 1:
            filename = f"{file_count:04d}"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for sent in current_file_sentences:
                    f.write(sent + '\n')
            
            print(f"Wrote {len(current_file_sentences):,} sentences to {filename}")
            file_count += 1
            current_file_sentences = []
    
    print(f"Preprocessing complete. Created {file_count} files in {output_dir}")
    return output_dir


def _process_wikipedia_json(json_path: str, words_per_sentence: int) -> List[str]:
    """
    Process Wikipedia JSON file and extract sentences.
    Similar to myw2v demo's _clean_up function.
    """
    sentences = []
    
    with utils.open(json_path, 'rb') as f:
        for line in f:
            try:
                article = json.loads(line)
                
                # Process title
                title_sents = _clean_up_text(article.get("title", ""))
                sentences.extend(title_sents)
                
                # Process section texts
                for section_text in article.get("section_texts", []):
                    section_sents = _clean_up_text(section_text)
                    sentences.extend(section_sents)
                    
            except (json.JSONDecodeError, KeyError):
                continue
    
    return sentences


def _clean_up_text(text: str) -> List[str]:
    """
    Clean up text similar to myw2v demo.
    """
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'[?\n.]+', text)
    clean_sentences = []
    
    for sent_str in sentences:
        # Clean up text
        a = re.sub(r'[\t\n]', ' ', sent_str)
        b = re.sub(r'[ ]{2,}', ' ', a)
        c = re.sub(r'[^a-zA-Z ]', '', b)  # Keep only letters and spaces
        words = [w.lower() for w in c.split() if w]
        
        if len(words) >= 2:  # Only keep sentences with at least 2 words
            clean_sentences.append(' '.join(words))
    
    return clean_sentences


def download_text8(output_dir: str = "./data") -> str:
    """
    Download text8 dataset from http://mattmahoney.net/dc/text8.zip
    Returns path to downloaded text8 file.
    """
    url = "http://mattmahoney.net/dc/text8.zip"
    output_path = os.path.join(output_dir, "text8")
    text8_file = os.path.join(output_path, "text8")
    
    # Create output directory
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if os.path.isfile(text8_file):
        print(f"Text8 file already exists at: {text8_file}")
        return text8_file
    
    zip_path = os.path.join(output_path, "text8.zip")
    
    print(f"Downloading text8 from {url}...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    # Remove zip file to save space
    os.remove(zip_path)
    
    print(f"Text8 dataset ready at: {text8_file}")
    return text8_file


def preprocess_wmt14_news(news_file_path: str, output_dir: str, words_per_sentence: int = 1000, 
                        max_sentences: int = None, max_files: int = None, use_phrases: bool = False,
                        phrase_threshold1: float = 200.0, phrase_threshold2: float = 100.0) -> str:
    """
    Preprocess WMT14 news file into sentence files compatible with myw2v format.
    
    This function now removes punctuation (commas, periods, etc.) and normalizes text.
    NOTE: If you have previously processed data that still contains punctuation,
    you need to delete the old processed files and reprocess to apply the cleaning.
    
    Args:
        news_file_path: Path to WMT14 news file
        output_dir: Output directory for processed files
        words_per_sentence: Number of words per sentence (default: 1000)
        max_sentences: Maximum number of sentences to process (None = all)
        max_files: Maximum number of files to create (None = all)
        use_phrases: Whether to apply phrase detection (default: False)
        phrase_threshold1: First pass phrase threshold (default: 200.0)
        phrase_threshold2: Second pass phrase threshold (default: 100.0)
    
    Returns:
        Path to output directory
    """
    print(f"Preprocessing WMT14 news file: {news_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Words per sentence: {words_per_sentence}")
    print("Note: Punctuation will be removed from text (commas, periods, etc.)")
    if max_sentences:
        print(f"Max sentences: {max_sentences:,}")
    if max_files:
        print(f"Max files: {max_files}")
    if use_phrases:
        print(f"Phrase detection: Enabled (threshold1={phrase_threshold1}, threshold2={phrase_threshold2})")
    
    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("0")]
    if existing_files:
        print(f"Found {len(existing_files)} existing processed files. Skipping preprocessing.")
        print("⚠️  WARNING: If these files contain punctuation, delete them and reprocess to apply cleaning.")
        return output_dir
    
    # Step 1: Basic preprocessing
    temp_dir = output_dir + "_temp"
    pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Read news file (one sentence per line)
    sentences = []
    sentence_count = 0
    
    with open(news_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Clean text: remove punctuation and normalize
            cleaned_line = clean_text_remove_punctuation(line)
            if cleaned_line:  # Skip empty lines after cleaning
                # Split into words and group into chunks
                words = cleaned_line.split()
                for i in range(0, len(words), words_per_sentence):
                    sentence_words = words[i:i + words_per_sentence]
                    if len(sentence_words) >= 2:  # Skip very short sentences
                        sentences.append(" ".join(sentence_words))
                        sentence_count += 1
                        
                        # Stop if we've reached max_sentences
                        if max_sentences and sentence_count >= max_sentences:
                            print(f"Reached max_sentences limit: {max_sentences:,}")
                            break
                
                # Break outer loop if we've reached max_sentences
                if max_sentences and sentence_count >= max_sentences:
                    break
    
    print(f"Total sentences: {len(sentences):,}")
    
    # Save to temporary files (similar to myw2v format)
    sentences_per_file = 100000
    file_count = 0
    current_file_sentences = []
    
    for i, sentence in enumerate(sentences):
        current_file_sentences.append(sentence)
        
        # Write file when it reaches sentences_per_file or we're at the end
        if len(current_file_sentences) >= sentences_per_file or i == len(sentences) - 1:
            filename = f"{file_count:04d}"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for sent in current_file_sentences:
                    f.write(sent + '\n')
            
            print(f"Wrote {len(current_file_sentences):,} sentences to {filename}")
            file_count += 1
            current_file_sentences = []
            
            # Stop if we've reached max_files
            if max_files and file_count >= max_files:
                print(f"Reached max_files limit: {max_files}")
                break
    
    # Step 2: Apply phrase detection if enabled
    if use_phrases:
        print("\nApplying phrase detection...")
        preprocess_with_phrases(temp_dir, output_dir, min_count=5, 
                               threshold1=phrase_threshold1, threshold2=phrase_threshold2)
        # Cleanup temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        # Just move files from temp to output
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.move(temp_dir, output_dir)
    
    print(f"Preprocessing complete. Created {file_count} files in {output_dir}")
    return output_dir


def preprocess_text8(text8_file_path: str, output_dir: str, words_per_sentence: int = 1000,
                    use_phrases: bool = False, phrase_threshold1: float = 200.0, 
                    phrase_threshold2: float = 100.0) -> str:
    """
    Preprocess text8 file into sentence files compatible with myw2v format.
    
    Args:
        text8_file_path: Path to text8 file
        output_dir: Output directory for processed files
        words_per_sentence: Number of words per sentence (default: 1000)
        use_phrases: Whether to apply phrase detection (default: False)
        phrase_threshold1: First pass phrase threshold (default: 200.0)
        phrase_threshold2: Second pass phrase threshold (default: 100.0)
    
    Returns:
        Path to output directory
    """
    print(f"Preprocessing text8 file: {text8_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Words per sentence: {words_per_sentence}")
    if use_phrases:
        print(f"Phrase detection: Enabled (threshold1={phrase_threshold1}, threshold2={phrase_threshold2})")
    
    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("0")]
    if existing_files:
        print(f"Found {len(existing_files)} existing processed files. Skipping preprocessing.")
        return output_dir
    
    # Step 1: Basic preprocessing
    temp_dir = output_dir + "_temp"
    pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Read text8 file (single long line)
    with open(text8_file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Split into words
    words = text.split()
    print(f"Total words: {len(words):,}")
    
    # Group into sentences
    sentences = []
    for i in range(0, len(words), words_per_sentence):
        sentence_words = words[i:i + words_per_sentence]
        if len(sentence_words) >= 2:  # Skip very short sentences
            sentences.append(" ".join(sentence_words))
    
    print(f"Created {len(sentences):,} sentences")
    
    # Save to temporary files (similar to myw2v format)
    sentences_per_file = 100000
    file_count = 0
    current_file_sentences = []
    
    for i, sentence in enumerate(sentences):
        current_file_sentences.append(sentence)
        
        # Write file when it reaches sentences_per_file or we're at the end
        if len(current_file_sentences) >= sentences_per_file or i == len(sentences) - 1:
            filename = f"{file_count:04d}"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for sent in current_file_sentences:
                    f.write(sent + '\n')
            
            print(f"Wrote {len(current_file_sentences):,} sentences to {filename}")
            file_count += 1
            current_file_sentences = []
    
    # Step 2: Apply phrase detection if enabled
    if use_phrases:
        print("\nApplying phrase detection...")
        preprocess_with_phrases(temp_dir, output_dir, min_count=5, 
                               threshold1=phrase_threshold1, threshold2=phrase_threshold2)
        # Cleanup temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        # Just move files from temp to output
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.move(temp_dir, output_dir)
    
    print(f"Preprocessing complete. Created {file_count} files in {output_dir}")
    return output_dir


def get_data_file_names(path: str, seed: int) -> List[str]:
    """Get shuffled list of data file names."""
    import numpy as np
    rng = np.random.default_rng(seed=seed)
    qq = [fn for fn in os.listdir(path) if fn.startswith("0")]
    # Sort first to ensure consistent shuffling
    data_files = sorted(qq)
    rng.shuffle(data_files)
    return data_files


def read_all_data_files(data_path: str, file_names: List[str], word_to_idx: dict) -> Tuple[List[int], List[int], List[int]]:
    """
    Read all data files and convert words to indices.
    Returns (inputs, offsets, lengths) compatible with myw2v format.
    """
    from collections import defaultdict
    
    start = time.time()
    inps, offs, lens = [], [], []
    offset_total = 0
    stats = defaultdict(int)
    
    for fn in file_names:
        fp = os.path.join(data_path, fn)
        ok_lines = 0
        too_short_lines = 0
        with open(fp, encoding="utf-8") as f:
            for line in f:
                words = [word for word in re.split(r"[ .]+", line.strip()) if word]
                if len(words) < 2:
                    too_short_lines += 1
                    continue
                idcs = [word_to_idx[w] for w in words if w in word_to_idx]
                le = len(idcs)
                ok_lines += 1
                offs.append(offset_total)
                lens.append(le)
                inps.extend(idcs)
                offset_total += le
        stats["file_read_lines_ok"] += ok_lines
        stats["one_word_sentence_lines_which_were_ignored"] += too_short_lines

    print(f"read_all_data_files() STATS: {stats}")
    tot_tm = time.time()-start
    print(f"read_all_data_files() Total time {tot_tm} s for {len(file_names)} files (avg {tot_tm/len(file_names)} s/file)")
    return inps, offs, lens
