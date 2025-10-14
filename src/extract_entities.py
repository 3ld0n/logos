# Script for Step 2: Perform Basic Named Entity Recognition (NER)
# This script loads the preprocessed verses, applies spaCy's pre-trained NER, extracts entities,
# and stores them in a list for further use.
# Decisions:
# - Model Selection: Using 'en_core_web_lg' for better accuracy on historical texts. Alternative: 'en_core_web_sm' for faster processing but lower precision; or 'en_core_web_trf' if transformers are installed for even better results.
# - Batch Processing: Using nlp.pipe() with batch_size=1000 for efficiency. Decision: Balance speed vs. memory; option to reduce to 500 if RAM is low.
# - Entity Storage: List of dicts for simplicity. Alternative: Use pandas DataFrame immediately for easier analysis in Step 7.

import spacy
import json

def load_processed_verses(file_path='processed_verses.json'):
    """Load preprocessed verses from JSON."""
    with open(file_path, 'r') as f:
        verses = json.load(f)
    return verses

def extract_entities(verses, nlp):
    """Extract entities using spaCy NER."""
    text_generator = (v['text'] for v in verses)
    verse_docs = list(nlp.pipe(text_generator, batch_size=1000))  # Efficient batch processing
    entities = []
    for i, (v, doc) in enumerate(zip(verses, verse_docs)):
        for ent in doc.ents:
            entities.append({
                "book": v.get('book'),  # Adjust keys based on JSON structure
                "chapter": v.get('chapter'),
                "verse": v.get('verse'),
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,  # Optional: For context
                "end": ent.end_char
            })
    return entities

if __name__ == "__main__":
    # Load spaCy model - decision: Large model for accuracy; download if not present.
    nlp = spacy.load("en_core_web_lg")  # Run 'python -m spacy download en_core_web_lg' if needed.
    
    verses = load_processed_verses()
    entities = extract_entities(verses, nlp)
    with open('basic_entities.json', 'w') as f:
        json.dump(entities, f, indent=2)
    print(f"Extracted {len(entities)} entities and saved to 'basic_entities.json'.")