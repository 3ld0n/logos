import spacy
import json
import re
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer # Updated imports for REBEL model
import pandas as pd
import nltk # Often needed for tokenization in transformers pipelines

# Load spaCy model
nlp = spacy.load("en_core_web_lg")  # Decision: Large model for better accuracy; alternative: en_core_web_sm for speed

def load_verses(file_path="processed_verses.json"):
    """Load preprocessed verses from JSON."""
    with open(file_path, "r") as f:
        verses = json.load(f)
    return verses

def load_genesis_verses(file_path="processed_verses.json"):
    """Load only Genesis verses (book_id=1) from JSON."""
    all_verses = load_verses(file_path)
    genesis_verses = [v for v in all_verses if v.get("book_id") == 1]
    print(f"Loaded {len(genesis_verses)} Genesis verses.")
    return genesis_verses

def extract_dependencies(verses):
    """Extract relationships using spaCy dependency parsing."""
    text_generator = (v["text"] for v in verses)
    verse_docs = list(nlp.pipe(text_generator, batch_size=1000))  # Batch processing for efficiency
    triples = []
    for v, doc in zip(verses, verse_docs):
        for token in doc:
            # Look for subject-verb-object patterns
            if token.dep_ == "nsubj" and token.ent_type_ in ["PERSON", "DIVINE"]:
                verb = token.head
                if verb.pos_ == "VERB":
                    # Find direct or prepositional object
                    obj = None
                    for child in verb.children:
                        if child.dep_ in ["dobj", "pobj"] and child.ent_type_ in ["PERSON", "PLACE", "DIVINE"]:
                            obj = child
                            break
                    if obj:
                        triples.append({
                            "book": v.get("book"),
                            "chapter": v.get("chapter"),
                            "verse": v.get("verse"),
                            "subject": token.text,
                            "predicate": verb.text,
                            "object": obj.text
                        })
    return triples

def match_verbs(verses):
    """Match common biblical verbs using regex."""
    verb_patterns = r"\b(spake|spoke|slew|begat|created|said)\b"  # Decision: Focus on key verbs; expand as needed
    triples = []
    for v in verses:
        matches = re.findall(verb_patterns, v["text"], re.IGNORECASE)
        if matches:
            # Simplified: Assume entities nearby; refine with spaCy for precision
            doc = nlp(v["text"])
            for token in doc:
                if token.text.lower() in matches and token.pos_ == "VERB":
                    subj = obj = None
                    for child in token.children:
                        if child.dep_ == "nsubj" and child.ent_type_ in ["PERSON", "DIVINE"]:
                            subj = child
                        if child.dep_ in ["dobj", "pobj"] and child.ent_type_ in ["PERSON", "PLACE", "DIVINE"]:
                            obj = child
                    if subj and obj:
                        triples.append({
                            "book": v.get("book"),
                            "chapter": v.get("chapter"),
                            "verse": v.get("verse"),
                            "subject": subj.text,
                            "predicate": token.text,
                            "object": obj.text
                        })
    return triples

def bert_relation_extraction(verses):
    """Use BERT for complex relations (optional)."""
    # Decision: Use only for ambiguous cases; alternative: Skip for simplicity
    try:
        # Load REBEL model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

        # Initialize the text2text-generation pipeline
        # REBEL is a text2text model that generates relations
        rel_ext = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1 # -1 for CPU, 0 for first GPU
        )

        triples = []
        for v in verses:
            text = v["text"]
            # REBEL expects text to be marked with [CLS] tokens
            input_text = text # REBEL handles tokenization internally; no need for manual [CLS]

            # The REBEL model extracts triples in a specific format like "<triplet> subj | rel | obj <triplet>"
            # We need to parse this output.
            result = rel_ext(input_text, return_tensors=True, return_text=False)
            generated_text = tokenizer.decode(result[0]['generated_token_ids'][0], skip_special_tokens=True)
            
            # Parse the generated text into triples
            extracted_relations = parse_rebel_output(generated_text)

            for rel in extracted_relations:
                triples.append({
                    "book": v.get("book"),
                    "chapter": v.get("chapter"),
                    "verse": v.get("verse"),
                    "subject": rel["head"],
                    "predicate": rel["type"],
                    "object": rel["tail"]
                })
        return triples
    except Exception as e:
        print(f"BERT extraction skipped: {e}")
        return []

def parse_rebel_output(text):
    """Parses the output of the REBEL model to extract subject, predicate, object triples."""
    extracted_relations = []
    current_head = ""
    current_type = ""
    for triple_match in re.finditer(r"<triplet>(.*?)</triplet>", text):
        triple_content = triple_match.group(1).strip()
        parts = [p.strip() for p in triple_content.split("|")]
        
        if len(parts) == 3:
            current_head = parts[0]
            current_type = parts[1]
            current_tail = parts[2]
            extracted_relations.append({"head": current_head, "type": current_type, "tail": current_tail})
    return extracted_relations

def validate_triples(triples):
    """Basic validation; manual or API-based checks can be added."""
    # Decision: Basic deduplication; alternative: Use Bible Gateway API for verse lookup
    df = pd.DataFrame(triples)
    df = df.drop_duplicates(subset=["subject", "predicate", "object", "book", "chapter", "verse"])
    return df.to_dict("records")

def main():
    print("load_versese starting")
    verses = load_genesis_verses()
    print("load_versese completed")
    #print("extract_dependencies starting")
    #dep_triples = extract_dependencies(verses)
    #print("extract_dependencies completed")
    #print("match_verbs starting")
    #verb_triples = match_verbs(verses)
    #print("match_verbs completed")
    bert_triples = bert_relation_extraction(verses)  # Uncomment if BERT is needed
    all_triples = bert_triples #dep_triples + verb_triples  # + bert_triples
    print("validate_triples starting")
    validated_triples = validate_triples(all_triples)
    print("validate_triples completed")
    print("writing to file starting")
    with open("candidate_triples.json", "w") as f:
        json.dump(validated_triples, f, indent=2)
    print(f"Extracted {len(validated_triples)} candidate triples to 'candidate_triples.json'.")
    print("writing to file completed")
if __name__ == "__main__":
    main()