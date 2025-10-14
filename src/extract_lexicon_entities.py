# Script for Step 4: Integrate Bible-Specific Lexicon from OpenBible.info
# This script fetches and integrates lexicons for places and people, adds them as patterns,
# and re-extracts entities.
# Decisions:
# - Source Selection: Using OpenBible for places and BradyStephenson for people. Alternative: Viz.Bible for more comprehensive, but requires request; decision: Use open GitHub sources.
# - Confidence Filter: For places, filter >0.5. Option: Set to 0 for all, but risks noise.
# - Matcher Type: Using EntityRuler for consistency. Alternative: PhraseMatcher for faster large lists.

import spacy
from spacy.pipeline import EntityRuler
import requests
import json
import pandas as pd

def fetch_places_lexicon():
    """Fetch and parse OpenBible places lexicon."""
    url = "https://raw.githubusercontent.com/openbibleinfo/Bible-Geocoding-Data/master/ancient.json"
    response = requests.get(url)
    places = []
    for line in response.text.splitlines():
        print(line)
        data = json.loads(line)
        if "name" in data and data.get("confidence", 0) > 0.5:  # Decision: Filter high-confidence
            places.append({"label": "PLACE", "pattern": [{"LOWER": data["name"].lower()}]})
    return places

def fetch_people_lexicon():
    """Fetch and parse BradyStephenson people CSV."""
    url = "https://raw.githubusercontent.com/BradyStephenson/bible-data/master/BibleData-Person.csv"
    df = pd.read_csv(url)
    people_patterns = [{"label": "PERSON", "pattern": [{"LOWER": name.lower()}]} for name in df['name'].unique()]
    return people_patterns

def integrate_lexicons(nlp, places, people):
    """Add lexicon patterns to EntityRuler."""
    ruler = EntityRuler(nlp)
    ruler.add_patterns(places + people)
    nlp.add_pipe(ruler, after="ner")
    return nlp

def load_processed_verses(file_path='processed_verses.json'):
    """Load preprocessed verses from JSON."""
    with open(file_path, 'r') as f:
        verses = json.load(f)
    return verses

def extract_lexicon_entities(verses, nlp):
    """Extract entities with integrated lexicons."""
    text_generator = (v['text'] for v in verses)
    verse_docs = list(nlp.pipe(text_generator, batch_size=1000))
    entities = []
    for i, (v, doc) in enumerate(zip(verses, verse_docs)):
        for ent in doc.ents:
            entities.append({
                "book": v.get('book'),
                "chapter": v.get('chapter'),
                "verse": v.get('verse'),
                "entity": ent.text,
                "label": ent.label_
            })
    return entities

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    
    places = fetch_places_lexicon()
    people = fetch_people_lexicon()
    nlp = integrate_lexicons(nlp, places, people)
    
    verses = load_processed_verses()
    entities = extract_lexicon_entities(verses, nlp)
    with open('lexicon_entities.json', 'w') as f:
        json.dump(entities, f, indent=2)
    print(f"Extracted {len(entities)} entities with lexicons and saved to 'lexicon_entities.json'.")