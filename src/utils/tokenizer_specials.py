from typing import List, Tuple

def build_clarity_special_tokens(max_person_tags: int = 32) -> List[str]:
    # NOTE: keep this stable across train and eval, or you'll get embedding mismatches
    specials = [
        "[QUESTION]",
        "[ANSWER]",
        "[PERSON]",
        "[CD_LOW]",
        "[CD_HIGH]",
    ]
    specials += [f"[PERSON_{i}]" for i in range(1, max_person_tags + 1)]
    return specials

def add_clarity_special_tokens(tokenizer, model=None, max_person_tags: int = 32) -> Tuple[int, List[str]]:
    """
    Adds CLARITY special tokens to tokenizer. Optionally resizes model embeddings.

    Returns:
        (num_added, token_list)
    """
    specials = build_clarity_special_tokens(max_person_tags=max_person_tags)

    existing = set(tokenizer.get_vocab().keys())
    # don't rely on this too hard, HF handles duplicates, but it's nice for debug prints
    to_add = [t for t in specials if t not in existing]

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": specials})

    # Resize embeddings if a model was passed
    if model is not None and num_added > 0:
        # Works for plain HF models and many wrappers
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
            model.model.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "resize_token_embeddings"):
            model.model.model.resize_token_embeddings(len(tokenizer))
        else:
            raise ValueError(
                "Could not find a resize_token_embeddings method on model. "
                "Pass the underlying HF model instead."
            )

    return num_added, specials