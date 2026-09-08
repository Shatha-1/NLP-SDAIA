"""Lab 3 starter: NER label alignment."""


def align_labels(word_ids, word_labels):
    """Map word-level BIO label ids onto subword tokens.

    The first subword piece of a word gets that word's label; special tokens
    (word_id is None) and continuation pieces (same word_id as the previous
    token) get -100 so the loss ignores them.
    """
    aligned = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            aligned.append(-100)
        else:
            aligned.append(word_labels[word_id])
        previous_word_id = word_id
    return aligned
