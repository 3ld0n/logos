# Script for Step 1: Prepare the Environment and Data
# This script handles installing dependencies (commented out for safety), downloading or loading KJV Bible data,
# preprocessing it, and preparing verses for further processing.
# Decisions:
# - Data Source: Using a GitHub JSON for KJV to avoid parsing raw text. Alternative: Use the Project Gutenberg TXT from previous scripts, but JSON is more structured and easier to handle. If JSON is unavailable, fall back to TXT parsing as in earlier code.
# - Filtering: Here, we filter for Genesis for testing. Decision: Start small to avoid memory issues; option to remove filter for full Bible, but increase batch_size in later steps.
# - Normalization: No lemmatization here; defer to spaCy in later steps. Option: Add NLTK lemmatizer if needed for archaic words.

import requests
import json

def load_bible_data():
    """Fetch KJV JSON from GitHub or load from local file if available."""
    url = "https://raw.githubusercontent.com/tushortz/Bible/master/json/kjv.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        bible_json = json.loads(response.text)
        verses = [verse['fields'] for verse in bible_json]  # Extract verse dicts assuming this structure
    except Exception as e:
        print(f"Error fetching data: {e}. Falling back to local file if exists.")
        # Alternative: Load from local 'kjv.json' - decision: Implement fallback for offline use.
        try:
            with open('kjv.json', 'r') as f:
                bible_json = json.load(f)
            verses = [verse['fields'] for verse in bible_json]
        except FileNotFoundError:
            raise ValueError("No local KJV JSON found. Please download manually.")
    return verses

def preprocess_verses(verses):
    """Preprocess verses: Normalize text (e.g., strip extra spaces) and filter for testing."""
    # Decision: Filter to Genesis (book_id == 1) for testing. Option: Set to None for full Bible, but monitor memory (full ~31k verses).
    test_verses = [v for v in verses] # if v.get('book_id') == 1]  # Assuming 'book_id' key; adjust based on actual JSON structure.
    for v in test_verses:
        v['text'] = v['text'].strip()  # Basic normalization; option: Add more like lowercasing, but preserve case for NER.
    return test_verses

if __name__ == "__main__":
    # Install dependencies if needed (run once manually)
    # import os
    # os.system('pip install requests')
    
    verses = load_bible_data()
    processed_verses = preprocess_verses(verses)
    with open('processed_verses.json', 'w') as f:
        json.dump(processed_verses, f, indent=2)
    print(f"Processed {len(processed_verses)} verses and saved to 'processed_verses.json'.")