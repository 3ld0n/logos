# Script for Step 6: Combine with Dependency Parsing for Context
# This script extracts entities and uses dependency parsing to find contextual relations,
# like actors and actions, storing enhanced data.
# Decisions:
# - Parse Focus: Targeting nsubj + VERB. Alternative: Expand to obj, etc., for full triples prep.
# - Output: List of dicts with context. Option: Integrate into entities list or separate relations file.

import spacy
import json

def load_processed_verses(file_path='processed_verses.json'):
    """Load preprocessed verses from JSON."""
    with open(file_path, 'r') as f:
        verses = json.load(f)
    return verses

def parse_dependencies(verses, nlp):
    """Extract entities and dependency-based context."""
    text_generator = (v['text'] for v in verses)
    verse_docs = list(nlp.pipe(text_generator, batch_size=1000))
    contexts = []
    for i, (v, doc) in enumerate(zip(verses, verse_docs)):
        for token in doc:
            if token.dep_ == "nsubj" and token.ent_type_ == "PERSON" and token.head.pos_ == "VERB":
                contexts.append({
                    "book": v.get('book'),
                    "chapter": v.get('chapter'),
                    "verse": v.get('verse'),
                    "actor": token.text,
                    "action": token.head.text
                })
                # Decision: Add more patterns, e.g., for objects: if child.dep_ == 'dobj'.
    return contexts

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")  # Could use custom model from Step 5.
    
    verses = load_processed_verses()
    contexts = parse_dependencies(verses, nlp)
    with open('dependency_contexts.json', 'w') as f:
        json.dump(contexts, f, indent=2)
    print(f"Extracted {len(contexts)} dependency contexts and saved to 'dependency_contexts.json'.")