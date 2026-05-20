import json
import re
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

def load_genesis_verses(file_path="/Users/eldon/Documents/GitHub/logos/src/processed_verses.json"):
    """Load only Genesis verses (book_id=1) from JSON."""
    with open(file_path, "r") as f:
        all_verses = json.load(f)
    genesis_verses = [v for v in all_verses if v.get("book_id") == 1]
    print(f"Loaded {len(genesis_verses)} Genesis verses.")
    return genesis_verses

def parse_rebel_output(text):
    """Parses the output of the REBEL model to extract subject, predicate, object triples."""
    extracted_relations = []
    for triple_match in re.finditer(r"<triplet>(.*?)</triplet>", text):
        triple_content = triple_match.group(1).strip()
        parts = [p.strip() for p in triple_content.split("|")]
        
        if len(parts) == 3:
            current_head = parts[0]
            current_type = parts[1]
            current_tail = parts[2]
            extracted_relations.append({"head": current_head, "type": current_type, "tail": current_tail})
    return extracted_relations

def bert_relation_extraction(verses):
    """Use BERT for complex relations using REBEL model."""
    try:
        print("Loading REBEL model...")
        # Load REBEL model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

        # Initialize the text2text-generation pipeline
        rel_ext = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1  # -1 for CPU, 0 for first GPU
        )
        print("REBEL model loaded successfully!")

        triples = []
        for i, v in enumerate(verses):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing verse {i+1}/{len(verses)}")
            
            text = v["text"]
            input_text = text

            # The REBEL model extracts triples in a specific format like "<triplet> subj | rel | obj <triplet>"
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
        print(f"BERT extraction failed: {e}")
        return []

def main():
    print("Loading Genesis verses...")
    verses = load_genesis_verses()
    
    print("Starting BERT relation extraction...")
    bert_triples = bert_relation_extraction(verses)
    
    print(f"Extracted {len(bert_triples)} triples from BERT relation extraction")
    
    print("Writing results to file...")
    with open("bert_candidate_triples.json", "w") as f:
        json.dump(bert_triples, f, indent=2)
    
    print(f"Results written to 'bert_candidate_triples.json'")

if __name__ == "__main__":
    main()
