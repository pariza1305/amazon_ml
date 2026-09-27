"""Full inference pipeline: test sources -> normalize -> blocking -> features
-> trained model -> threshold -> matching_results.tsv + candidate_pairs.tsv.

Enforces every output-format rule from the guidelines:
  - exactly one row per test_source1 entity (including countries unseen in training)
  - empty string for no-match
  - no duplicate entity IDs within a list, no duplicate source1_entity_id rows
  - matched IDs are a subset of that entity's candidate IDs

Usage: python3 predict.py [--n1 N] [--n23 N] [--top-k K] [--threshold T]
(omit --threshold to use the value calibrated in threshold.json from train.py)
"""
import argparse
import csv
import json
import os
import sys
import time
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from build_candidates import load_and_normalize, build_indexes
from blocking import candidates_for_entity
from features import compute_features, FEATURE_NAMES

SRC_DIR = __file__.rsplit("/", 1)[0]


def load_threshold_config():
    """Returns the full threshold.json dict (threshold + the blocking params
    training was calibrated with), or a safe fallback if it doesn't exist."""
    path = os.path.join(SRC_DIR, "threshold.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"threshold": 0.5, "top_k": 20, "per_route_k": None, "final_top_k": None}


def write_tsv(path, rows, id_col_name):
    """rows: list of (source1_entity_id, comma_joined_ids_or_empty)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", id_col_name])
        for eid, ids in rows:
            w.writerow([eid, ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=None, help="limit test S1 rows (debug only; omit for full run)")
    ap.add_argument("--n23", type=int, default=None, help="limit test S2/S3 rows (debug only; omit for full run)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="per-route candidate cap; defaults to threshold.json's value from training, "
                         "falling back to 20 if that file is missing")
    ap.add_argument("--per-route-k", type=int, default=None)
    ap.add_argument("--final-top-k", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_threshold_config()
    threshold = args.threshold if args.threshold is not None else cfg["threshold"]
    top_k = args.top_k if args.top_k is not None else cfg.get("top_k", 20)
    per_route_k = args.per_route_k if args.per_route_k is not None else cfg.get("per_route_k")
    final_top_k = args.final_top_k if args.final_top_k is not None else cfg.get("final_top_k")
    t0 = time.time()

    s1 = load_and_normalize("dataset/test/test_source1.tsv", nrows=args.n1)
    s2 = load_and_normalize("dataset/test/test_source2.tsv", nrows=args.n23)
    s3 = load_and_normalize("dataset/test/test_source3.tsv", nrows=args.n23)
    print(f"[{time.time()-t0:.0f}s] loaded {len(s1)} test S1 / {len(s2)} S2 / {len(s3)} S3 "
          f"(threshold={threshold})")

    s2_norm, s3_norm = dict(s2), dict(s3)
    valid_cand_ids = set(s2_norm) | set(s3_norm)
    indexes, _ = build_indexes(s2, s3)
    print(f"[{time.time()-t0:.0f}s] built indexes ({len(indexes)} countries: {sorted(indexes)})")

    booster = lgb.Booster(model_file=os.path.join(SRC_DIR, "model.txt"))

    candidate_rows = []   # (s1_entity_id, cand_id)
    match_rows = []        # (s1_entity_id, cand_id)
    n_no_country_match = 0

    for eid, norm in s1:
        country = norm["country_norm"]
        if country not in indexes:
            # e.g. a country combination with zero S2/S3 records at all -> no candidates possible
            n_no_country_match += 1
            candidate_rows.append((eid, ""))
            match_rows.append((eid, ""))
            continue
        name_idx, name_idf, pin_idx, loc_idx = indexes[country]
        cands = candidates_for_entity(norm["name_tokens"], norm["addr_pin"], norm["addr_locality"],
                                       name_idx, name_idf, pin_idx, loc_idx,
                                       top_k=top_k, per_route_k=per_route_k, final_top_k=final_top_k)
        if not cands:
            candidate_rows.append((eid, ""))
            match_rows.append((eid, ""))
            continue

        feats_list, cand_ids = [], []
        for cand_id, score in cands:
            cand_norm = s2_norm.get(cand_id) or s3_norm.get(cand_id)
            if cand_norm is None:
                continue
            is_s2 = cand_id.startswith("S2-")
            feats_list.append(compute_features(norm, cand_norm, score, is_s2))
            cand_ids.append(cand_id)

        # dedupe candidate ids while preserving order (defensive; blocking already dedupes)
        seen = set()
        dedup_cand_ids = []
        dedup_feats = []
        for cid, ft in zip(cand_ids, feats_list):
            if cid not in seen:
                seen.add(cid)
                dedup_cand_ids.append(cid)
                dedup_feats.append(ft)
        cand_ids, feats_list = dedup_cand_ids, dedup_feats

        candidate_rows.append((eid, ",".join(cand_ids) if cand_ids else ""))

        if not feats_list:
            match_rows.append((eid, ""))
            continue

        Xp = pd.DataFrame(feats_list, columns=FEATURE_NAMES)
        probs = booster.predict(Xp)
        matched = [cid for cid, p in zip(cand_ids, probs) if p >= threshold]
        # safety: matched must be subset of candidates (true by construction) and
        # must only reference ids that exist in the test set (true by construction)
        matched = [m for m in matched if m in valid_cand_ids]
        match_rows.append((eid, ",".join(matched) if matched else ""))

    write_tsv("output/candidate_pairs.tsv", candidate_rows, "candidate_entity_ids")
    write_tsv("output/matching_results.tsv", match_rows, "matched_entity_ids")

    n_with_matches = sum(1 for _, ids in match_rows if ids)
    n_with_cands = sum(1 for _, ids in candidate_rows if ids)
    avg_cands = sum(len(ids.split(",")) if ids else 0 for _, ids in candidate_rows) / max(len(candidate_rows), 1)
    print(f"[{time.time()-t0:.0f}s] wrote output/candidate_pairs.tsv and output/matching_results.tsv")
    print(f"  {len(match_rows)} total S1 rows; {n_with_cands} with >=1 candidate; "
          f"{n_with_matches} predicted with >=1 match; avg candidates/entity: {avg_cands:.2f}; "
          f"{n_no_country_match} entities in a country with zero S2/S3 records")


if __name__ == "__main__":
    main()
