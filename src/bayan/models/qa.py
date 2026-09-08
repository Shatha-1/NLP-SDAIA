"""Lab 3 starter: extractive QA post-processing."""


def best_span(start_logits, end_logits, offsets, *, null_score, null_threshold, max_answer_len=30, top_k=20):
    """Constrained span search over start/end logits with honest null handling.

    offsets[i] is (start_char, end_char) for token i, or None for special/padding
    tokens that can never be a span endpoint. Among the top_k start and top_k end
    candidates (restricted to valid, non-special positions), keep only spans where
    end >= start and length <= max_answer_len (this is what rejects an "inverted"
    span built naively from independent argmax(start) / argmax(end)).

    null_score is the caller-supplied no-answer score (typically CLS start+end
    logits). If null_score minus the best valid span score is >= null_threshold,
    the no-answer prediction wins (SQuAD2-style verdict).
    """
    valid_indices = [i for i, off in enumerate(offsets) if off is not None]

    start_candidates = sorted(valid_indices, key=lambda i: start_logits[i], reverse=True)[:top_k]
    end_candidates = sorted(valid_indices, key=lambda i: end_logits[i], reverse=True)[:top_k]

    best = None  # (score, start_idx, end_idx)
    for s in start_candidates:
        for e in end_candidates:
            if e < s or (e - s + 1) > max_answer_len:
                continue
            score = float(start_logits[s] + end_logits[e])
            if best is None or score > best[0]:
                best = (score, s, e)

    if best is None:
        return {"answer": None, "score_diff": None}

    best_score, best_s, best_e = best
    score_diff = float(null_score) - best_score

    if score_diff >= null_threshold:
        return {"answer": None, "score_diff": score_diff, "best_non_null_score": best_score}

    start_char, _ = offsets[best_s]
    _, end_char = offsets[best_e]
    return {
        "answer": {"start_char": start_char, "end_char": end_char, "score": best_score},
        "score_diff": score_diff,
    }
