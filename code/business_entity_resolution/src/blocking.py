"""Candidate generation (blocking) for entity resolution.

Multi-stage, country-agnostic:
  1. Bucket S2/S3 by normalized country (no hard-coded country list).
  2. Within a country bucket, three INDEPENDENT blocking routes:
       - name tokens (IDF-weighted inverted index)
       - exact PIN match
       - exact locality match
     Each route is ranked and capped on its own (per_route_k), THEN unioned.
  3. (Optional) an overall cap after the union, if candidate-set size needs
     to be bounded further.

Why independent routes + union, not one combined score + single cutoff:
a true match with a strong PIN/locality signal but weak-or-no name overlap
can get crowded out of a single combined-score top-K by a flood of
candidates that share a moderately common name token (IDF-weighted scores
from name matches can dwarf the flat PIN/locality bonus once the corpus is
large). Ranking+capping each signal independently guarantees a PIN-only or
locality-only true match still gets a fair shot at survival, instead of
competing directly against name-token noise for the same K slots.
This was reported as the #1 blocking-recall bottleneck after real-scale
experiments (5M+5M rows, top_k=20: 58.8% recall; top_k=50: 63.7% recall,
i.e. ~36% of true matches never reached the classifier) — see context.md,
2026-09-27.

Output: candidate_pairs dict {s1_entity_id: [(s2_or_s3_id, score), ...]}
"""
from collections import defaultdict
import math

STOP_NAME_TOKENS = {"corp", "inc", "ltd", "pvt", "co", "llc", "llp", "plc",
                     "lp", "pc", "lc", "pllc", "gmbh", "sa", "and", "the", "of", "a", "an"}


def build_token_index(records, token_field):
    """records: list of (entity_id, normalized_dict). Returns {token: set(entity_ids)}."""
    index = defaultdict(set)
    for eid, norm in records:
        tokens = norm.get(token_field) or ()
        for t in tokens:
            if token_field == "name_tokens" and t in STOP_NAME_TOKENS:
                continue
            if len(t) < 2:
                continue
            index[t].add(eid)
    return index


def token_idf(index, n_docs):
    """IDF weight per token: rarer tokens score higher."""
    return {t: math.log((n_docs + 1) / (len(ids) + 1)) + 1.0 for t, ids in index.items()}


def _name_scores(s1_tokens_name, name_index, name_idf):
    scores = defaultdict(float)
    for t in s1_tokens_name:
        if t in STOP_NAME_TOKENS or len(t) < 2:
            continue
        w = name_idf.get(t, 1.0)
        for cid in name_index.get(t, ()):
            scores[cid] += w
    return scores


def candidates_for_entity(s1_tokens_name, s1_pin, s1_locality,
                           name_index, name_idf, pin_index, locality_index,
                           top_k=20, per_route_k=None, final_top_k=None):
    """Three independent routes (name / pin / locality), each ranked and
    capped at per_route_k, then unioned. final_top_k optionally truncates
    the union afterward (None = keep the whole union, prioritizing recall).

    top_k is kept as the per_route_k default for backward compatibility
    with existing callers (train.py/predict.py --top-k).
    Returns list of (candidate_id, combined_score) sorted desc.
    """
    per_route_k = per_route_k if per_route_k is not None else top_k

    name_scores = _name_scores(s1_tokens_name, name_index, name_idf)
    name_ranked_ids = {cid for cid, _ in
                       sorted(name_scores.items(), key=lambda kv: kv[1], reverse=True)[:per_route_k]}

    pin_ids = set(pin_index.get(s1_pin, ())) if s1_pin else set()
    if len(pin_ids) > per_route_k:
        pin_ids = set(sorted(pin_ids, key=lambda cid: name_scores.get(cid, 0.0), reverse=True)[:per_route_k])

    locality_ids = set(locality_index.get(s1_locality, ())) if s1_locality else set()
    if len(locality_ids) > per_route_k:
        locality_ids = set(sorted(locality_ids, key=lambda cid: name_scores.get(cid, 0.0), reverse=True)[:per_route_k])

    union_ids = name_ranked_ids | pin_ids | locality_ids

    combined = {}
    for cid in union_ids:
        s = name_scores.get(cid, 0.0)
        if cid in pin_ids:
            s += 5.0
        if cid in locality_ids:
            s += 1.0
        combined[cid] = s

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    if final_top_k is not None:
        ranked = ranked[:final_top_k]
    return ranked


def build_country_bucket_indexes(records_by_country, country):
    """records_by_country: {country_norm: [(eid, norm_dict), ...]}. Builds all indexes for one bucket."""
    recs = records_by_country.get(country, [])
    name_index = build_token_index(recs, "name_tokens")
    name_idf = token_idf(name_index, max(len(recs), 1))
    pin_index = defaultdict(set)
    locality_index = defaultdict(set)
    for eid, norm in recs:
        if norm.get("addr_pin"):
            pin_index[norm["addr_pin"]].add(eid)
        if norm.get("addr_locality"):
            locality_index[norm["addr_locality"]].add(eid)
    return name_index, name_idf, pin_index, locality_index
