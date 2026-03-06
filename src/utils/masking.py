from typing import List, Set, Tuple
import spacy

_nlp = None  # will be loaded on first use

def _get_nlp():
    global _nlp
    if _nlp is None:
        # _nlp = spacy.load("en_core_web_sm")
        _nlp = spacy.load("en_core_web_lg")
    return _nlp

_TITLES = {
    "mr", "mrs", "ms", "dr", "prof", "president", "sir", "madam"
}

def _normalise_person_name(name: str) -> Set[str]:
    """
    Normalise a PERSON name into a set of tokens.

    "Dr. John Smith" -> {"john", "smith"}
    "John" -> {"john"}
    """
    tokens = []
    for tok in name.lower().replace(".", "").split():
        if tok not in _TITLES:
            tokens.append(tok)
    return set(tokens)

def mask_person_entities(text: str) -> str:
    """
    Replace PERSON named entities with [PERSON].
    This works on entity spans so spacing doesnt get mangled.
    Naive.

    e.g., "John Smith met Mary for coffee" -> "[PERSON] met [PERSON] for coffee"
    """
    if not text:
        return text

    nlp = _get_nlp()
    doc = nlp(text)
    
    result = []
    last_idx = 0  # keep track of where we are in the string

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue

        # add text before the entity
        result.append(text[last_idx:ent.start_char])

        # replace the whole name with one token
        result.append("[PERSON]")

        last_idx = ent.end_char

    # add whatever text is left at the end
    result.append(text[last_idx:])

    return "".join(result)

def mask_person_entities_pair(question: str, answer: str) -> Tuple[str, str]:
    """
    Entity-aware masking.

    Same real-world person gets the same [PERSON_x] tag across
    question and answer, even if referred to as:
      - "John"
      - "John Smith"
      - "Mr. Smith"

    Matching is based on token overlap and surname agreement.
    """
    if not question and not answer:
        return question, answer

    nlp = _get_nlp()

    entity_keys: List[Set[str]] = []
    entity_tags: List[str] = []

    def _find_or_create_entity(norm_tokens: Set[str]) -> str:
        # try to match against existing entities
        for i, existing in enumerate(entity_keys):
            # subset match or shared surname
            if (
                norm_tokens <= existing
                or existing <= norm_tokens
                or (len(norm_tokens & existing) > 0 and len(existing) > 1)
            ):
                return entity_tags[i]

        # new entity
        tag = f"[PERSON_{len(entity_tags) + 1}]"
        entity_keys.append(norm_tokens)
        entity_tags.append(tag)
        return tag

    def _mask_text(text: str) -> str:
        if not text:
            return text

        doc = nlp(text)
        result = []
        last_idx = 0

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            norm = _normalise_person_name(ent.text)
            tag = _find_or_create_entity(norm)

            result.append(text[last_idx:ent.start_char])
            result.append(tag)
            last_idx = ent.end_char

        result.append(text[last_idx:])
        return "".join(result)

    # question first = stable IDs
    masked_question = _mask_text(question)
    masked_answer = _mask_text(answer)

    return masked_question, masked_answer