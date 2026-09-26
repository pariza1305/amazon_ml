"""Pairwise feature engineering for (S1, candidate) pairs surviving blocking.
Reused identically at train time and inference time to avoid skew."""
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

FEATURE_NAMES = [
    "name_jaccard", "name_jaro_winkler", "name_lev_ratio", "name_token_sort_ratio",
    "name_common_tokens", "name_exact_core", "name_len_diff",
    "addr_jaccard", "addr_lev_ratio", "addr_house_match", "addr_pin_match",
    "addr_locality_match", "addr_len_diff",
    "same_country", "block_score", "is_source2",
]


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_features(s1_norm, cand_norm, block_score, cand_is_source2):
    n1_tokens, n2_tokens = s1_norm["name_tokens"], cand_norm["name_tokens"]
    n1, n2 = s1_norm["name_core"], cand_norm["name_core"]
    a1_tokens, a2_tokens = s1_norm["addr_tokens"], cand_norm["addr_tokens"]
    a1, a2 = s1_norm["addr_norm"], cand_norm["addr_norm"]

    feats = {
        "name_jaccard": _jaccard(n1_tokens, n2_tokens),
        "name_jaro_winkler": fuzz.WRatio(n1, n2) / 100.0 if n1 and n2 else 0.0,
        "name_lev_ratio": Levenshtein.normalized_similarity(n1, n2) if n1 and n2 else 0.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
        "name_common_tokens": len(set(n1_tokens) & set(n2_tokens)),
        "name_exact_core": 1.0 if n1 and n1 == n2 else 0.0,
        "name_len_diff": abs(len(n1) - len(n2)),
        "addr_jaccard": _jaccard(a1_tokens, a2_tokens),
        "addr_lev_ratio": Levenshtein.normalized_similarity(a1, a2) if a1 and a2 else 0.0,
        "addr_house_match": 1.0 if s1_norm["addr_house_no"] and s1_norm["addr_house_no"] == cand_norm["addr_house_no"] else 0.0,
        "addr_pin_match": 1.0 if s1_norm["addr_pin"] and s1_norm["addr_pin"] == cand_norm["addr_pin"] else 0.0,
        "addr_locality_match": 1.0 if s1_norm["addr_locality"] and s1_norm["addr_locality"] == cand_norm["addr_locality"] else 0.0,
        "addr_len_diff": abs(len(a1) - len(a2)),
        "same_country": 1.0 if s1_norm["country_norm"] == cand_norm["country_norm"] else 0.0,
        "block_score": block_score,
        "is_source2": 1.0 if cand_is_source2 else 0.0,
    }
    return [feats[name] for name in FEATURE_NAMES]


def add_relative_rank_features(pair_rows):
    """pair_rows: list of dicts with 's1_id' and 'score' (model prob or block_score).
    Adds 'rank_within_entity' (0=best) and 'score_gap_to_next' in place, per s1_id group.
    Call after scoring; pair_rows must be pre-sorted or will be sorted here."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in pair_rows:
        groups[row["s1_id"]].append(row)
    for s1_id, rows in groups.items():
        rows.sort(key=lambda r: r["score"], reverse=True)
        for i, row in enumerate(rows):
            row["rank_within_entity"] = i
            row["score_gap_to_next"] = (row["score"] - rows[i + 1]["score"]) if i + 1 < len(rows) else row["score"]
    return pair_rows
