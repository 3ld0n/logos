# Script for Step 6: Combine with Dependency Parsing for Context
# This script extracts entities and uses dependency parsing to find contextual relations,
# like actors and actions, storing enhanced data.
# Decisions:
# - Parse Focus: Targeting nsubj + VERB. Alternative: Expand to obj, etc., for full triples prep.
# - Output: List of dicts with context. Option: Integrate into entities list or separate relations file.

import spacy
import json
try:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models  # noqa: F401 - ensure base registries are loaded
    import allennlp_models.structured_prediction  # noqa: F401 - ensure SRL registries are loaded
    try:
        from allennlp_models.pretrained import load_predictor as _pretrained_load_predictor
    except Exception as e:
        _pretrained_load_predictor = None
        import traceback
        print("WARNING: AllenNLP import failed. SRL will be skipped.")
        traceback.print_exc()
    _HAS_ALLENNLP = True
except Exception as e:
    _HAS_ALLENNLP = False
    import traceback
    print("WARNING: AllenNLP import failed. SRL will be skipped.")
    traceback.print_exc()

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

def _extract_span(words, tags, role_prefix):
    """Extract a contiguous span for a role like 'ARG0', 'ARG1', 'ARG2', or 'V'."""
    tokens = []
    collecting = False
    for w, t in zip(words, tags):
        if t == f"B-{role_prefix}":
            tokens = [w]
            collecting = True
        elif t == f"I-{role_prefix}" and collecting:
            tokens.append(w)
        elif collecting and t.startswith("B-"):
            break
    return " ".join(tokens).strip() if tokens else ""

def extract_srl_relations(verses, predictor):
    """Run SRL to extract candidate (subj, pred, obj) per verse.
    Preference: ARG0 as subject, predicate 'V', ARG1 as object (fallback ARG2).
    """
    if not _HAS_ALLENNLP:
        raise RuntimeError(
            "AllenNLP is not installed. Install with: pip install 'numpy<2' 'torch==2.0.1' 'torchvision==0.15.2' allennlp==2.10.1 allennlp-models==2.10.1"
        )
    candidates = []
    for v in verses:
        text = v.get('text', '') or ''
        if not text:
            continue
        try:
            result = predictor.predict(sentence=text)
        except Exception:
            continue

        words = result.get('words') or []
        for frame in result.get('verbs', []) or []:
            tags = frame.get('tags') or []
            if not tags or len(tags) != len(words):
                continue
            subj = _extract_span(words, tags, 'ARG0')
            pred = frame.get('verb') or _extract_span(words, tags, 'V')
            obj = _extract_span(words, tags, 'ARG1') or _extract_span(words, tags, 'ARG2')

            if not pred:
                continue
            if not subj and not obj:
                continue

            book_display = v.get('book') or v.get('book_id')
            chapter = v.get('chapter')
            verse = v.get('verse')
            verse_ref = f"{book_display} {chapter}:{verse}"

            candidates.append({
                "subj": subj if subj else None,
                "pred": pred,
                "obj": obj if obj else None,
                "verse_ref": verse_ref
            })
    return candidates

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    verses = load_processed_verses()

    # Existing dependency-based contexts (kept)
    contexts = parse_dependencies(verses, nlp)
    with open('dependency_contexts.json', 'w') as f:
        json.dump(contexts, f, indent=2)

    # AllenNLP SRL for candidate relations
    # Prefer pretrained loader to ensure correct registry/config
    candidates = []
    predictor = None
    if _HAS_ALLENNLP:
        if _pretrained_load_predictor is not None:
            try:
                predictor = _pretrained_load_predictor("semantic-role-labeling")
            except Exception:
                try:
                    predictor = _pretrained_load_predictor("structured-prediction-srl-bert")
                except Exception:
                    predictor = None
        if predictor is None:
            try:
                from allennlp.predictors.predictor import Predictor as _Predictor
                predictor = _Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
                )
            except Exception:
                predictor = None

    if predictor is not None:
        candidates = extract_srl_relations(verses, predictor)
        print(f"Extracted {len(candidates)} SRL candidate relations and saved to 'srl_candidate_relations.json'.")
    else:
        print("AllenNLP SRL not available; writing empty SRL candidates.")

    with open('srl_candidate_relations.json', 'w') as f:
        json.dump(candidates, f, indent=2)