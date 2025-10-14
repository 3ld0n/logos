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
    url = "https://raw.githubusercontent.com/openbibleinfo/Bible-Geocoding-Data/refs/heads/main/data/ancient.jsonl"
    response = requests.get(url)
    response.raise_for_status()

    # Parse JSONL: one JSON object per line
    name_to_best_confidence = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract possible names
        candidate_names = []
        if isinstance(rec.get("name"), str):
            candidate_names.append(rec.get("name"))
        if isinstance(rec.get("names"), list):
            candidate_names.extend([n for n in rec.get("names") if isinstance(n, str)])

        conf_val = rec.get("confidence") or rec.get("Confidence") or rec.get("score")
        try:
            confidence = float(conf_val) if conf_val is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        for nm in candidate_names:
            name = nm.strip()
            if not name:
                continue
            prev = name_to_best_confidence.get(name)
            if prev is None or confidence > prev:
                name_to_best_confidence[name] = confidence

    places = []
    for name, confidence in name_to_best_confidence.items():
        if confidence > 0.5:  # Decision: Filter high-confidence
            places.append({"label": "PLACE", "pattern": [{"LOWER": name.lower()}]})
    return places

def fetch_people_lexicon():
    """Fetch and parse BradyStephenson people CSV."""
    url = "https://raw.githubusercontent.com/BradyStephenson/bible-data/master/BibleData-Person.csv"
    df = pd.read_csv(url)
    # Auto-detect a name-like column (handles different header spellings)
    columns_lower = {str(c).lower(): c for c in df.columns}
    name_col = None
    for key in ["name", "person", "label", "title", "displayname", "fullname"]:
        for lower, original in columns_lower.items():
            if key in lower:
                name_col = original
                break
        if name_col is not None:
            break
    if name_col is None:
        raise ValueError(f"Could not find a name-like column in people CSV. Columns: {list(df.columns)}")

    names = (
        df[name_col]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .unique()
    )
    people_patterns = [{"label": "PERSON", "pattern": [{"LOWER": name.lower()}]} for name in names]
    return people_patterns

def integrate_lexicons(nlp, places, people):
    """Add lexicon patterns to EntityRuler."""
    # spaCy v3+: add by factory name, then configure
    try:
        ruler = nlp.add_pipe("entity_ruler", after="ner")
    except ValueError:
        # If 'ner' not in pipeline, append at end
        ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(places + people)
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