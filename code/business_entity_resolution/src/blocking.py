"""Candidate generation (blocking) for entity resolution.

Multi-stage, country-agnostic:
  1. Bucket S2/S3 by normalized country (no hard-coded country list).
  2. Within a country bucket, inverted-index blocking on:
     - rare/significant name tokens (IDF-weighted)
     - address locality + PIN tokens
  3. Score candidates by combined token overlap, cap at top-K per S1 entity.

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


def candidates_for_entity(s1_tokens_name, s1_pin, s1_locality,
                           name_index, name_idf, pin_index, locality_index,
                           top_k=20):
    """Returns list of (candidate_id, score) sorted desc, capped at top_k."""
    scores = defaultdict(float)
    seen_any_key = defaultdict(int)

    for t in s1_tokens_name:
        if t in STOP_NAME_TOKENS or len(t) < 2:
            continue
        w = name_idf.get(t, 1.0)
        for cid in name_index.get(t, ()):
            scores[cid] += w
            seen_any_key[cid] += 1

    if s1_pin:
        for cid in pin_index.get(s1_pin, ()):
            scores[cid] += 5.0  # strong signal, fixed weight
            seen_any_key[cid] += 1

    if s1_locality:
        for cid in locality_index.get(s1_locality, ()):
            scores[cid] += 1.0
            seen_any_key[cid] += 1

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


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
