"""Trains the pairwise matching classifier on candidate_pairs generated from
the training sources, labeled against train_ground_truth.tsv. Includes a
held-out validation split (grouped by S1 entity, no leakage) used to
calibrate the match/no-match decision threshold against the official
macro F0.5 metric.

Usage: python3 train.py [--n1 N] [--n23 N] [--top-k K] [--val-frac F] [--seed S]
Produces:
  model.txt      LightGBM booster
  threshold.json chosen decision threshold + the val F0.5 it achieved
"""
import argparse
import json
import random
import sys
import time
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from build_candidates import load_and_normalize, build_indexes
from blocking import candidates_for_entity
from features import compute_features, FEATURE_NAMES
from eval import macro_f_beta, blocking_recall_ceiling


def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for row in df.itertuples(index=False):
        matches = set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
        gt[row.source1_entity_id] = matches
    return gt


def split_entities(s1_ids, val_frac, seed):
    ids = list(s1_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def build_rows(s1, indexes, s2_norm, s3_norm, gt, top_k, per_route_k=None, final_top_k=None):
    rows = []
    for eid, norm in s1:
        country = norm["country_norm"]
        if country not in indexes:
            continue
        name_idx, name_idf, pin_idx, loc_idx = indexes[country]
        cands = candidates_for_entity(norm["name_tokens"], norm["addr_pin"], norm["addr_locality"],
                                       name_idx, name_idf, pin_idx, loc_idx,
                                       top_k=top_k, per_route_k=per_route_k, final_top_k=final_top_k)
        true_matches = gt.get(eid, set())
        for cand_id, score in cands:
            cand_norm = s2_norm.get(cand_id) or s3_norm.get(cand_id)
            if cand_norm is None:
                continue
            is_s2 = cand_id.startswith("S2-")
            feats = compute_features(norm, cand_norm, score, is_s2)
            rows.append({
                "s1_id": eid, "cand_id": cand_id, "feats": feats,
                "label": 1 if cand_id in true_matches else 0,
            })
    return rows


def sweep_threshold(val_rows, probs, val_gt, all_val_ids):
    by_entity = {}
    for row, p in zip(val_rows, probs):
        by_entity.setdefault(row["s1_id"], []).append((row["cand_id"], p))

    best_t, best_score = 0.5, -1.0
    for t in [i / 100 for i in range(5, 96, 5)]:
        preds = {}
        for eid, cand_scores in by_entity.items():
            matched = {cid for cid, p in cand_scores if p >= t}
            preds[eid] = matched
        score = macro_f_beta(preds, val_gt, all_entity_ids=all_val_ids)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=20000,
                    help="number of S1 rows to use, drawn as a TRUE RANDOM SAMPLE "
                         "of the full file (not the first N) for a representative split")
    ap.add_argument("--n23", type=int, default=1000000,
                    help="row-prefix cap on S2/S3 (dev-mode I/O limit; full run omits this)")
    ap.add_argument("--top-k", type=int, default=20, help="per-route candidate cap (see blocking.py)")
    ap.add_argument("--per-route-k", type=int, default=None,
                    help="override per-route cap independently of --top-k (defaults to --top-k)")
    ap.add_argument("--final-top-k", type=int, default=None,
                    help="optional cap on the unioned candidate set size (default: no cap, "
                         "prioritizing recall over candidate-set size until blocking is tuned)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    s1 = load_and_normalize("dataset/train/train_source1.tsv", sample_n=args.n1, seed=args.seed)
    s2 = load_and_normalize("dataset/train/train_source2.tsv", nrows=args.n23)
    s3 = load_and_normalize("dataset/train/train_source3.tsv", nrows=args.n23)
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    print(f"[{time.time()-t0:.0f}s] loaded {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 / {len(gt)} gt rows")

    indexes, _ = build_indexes(s2, s3)
    s2_norm, s3_norm = dict(s2), dict(s3)
    print(f"[{time.time()-t0:.0f}s] built indexes ({len(indexes)} countries)")

    train_ids, val_ids = split_entities([eid for eid, _ in s1], args.val_frac, args.seed)
    s1_train = [(e, n) for e, n in s1 if e in train_ids]
    s1_val = [(e, n) for e, n in s1 if e in val_ids]
    print(f"[{time.time()-t0:.0f}s] split: {len(s1_train)} train entities, {len(s1_val)} val entities")

    train_rows = build_rows(s1_train, indexes, s2_norm, s3_norm, gt, args.top_k,
                             args.per_route_k, args.final_top_k)
    val_rows = build_rows(s1_val, indexes, s2_norm, s3_norm, gt, args.top_k,
                           args.per_route_k, args.final_top_k)
    n_pos_train = sum(r["label"] for r in train_rows)
    print(f"[{time.time()-t0:.0f}s] train candidate rows: {len(train_rows)} (pos={n_pos_train}); "
          f"val candidate rows: {len(val_rows)}")

    val_candidate_sets = {}
    for row in val_rows:
        val_candidate_sets.setdefault(row["s1_id"], set()).add(row["cand_id"])
    val_gt = {eid: gt.get(eid, set()) for eid, _ in s1_val}
    ceiling = blocking_recall_ceiling(val_candidate_sets, val_gt)
    print(f"[{time.time()-t0:.0f}s] blocking recall ceiling on val split: {ceiling:.3f}")

    if n_pos_train == 0:
        print("WARNING: zero positive training rows in this sample - results below are not "
              "meaningful. Increase --n1/--n23.")

    Xtr = pd.DataFrame([r["feats"] for r in train_rows], columns=FEATURE_NAMES)
    ytr = [r["label"] for r in train_rows]
    n_neg = len(ytr) - n_pos_train
    params = {
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "scale_pos_weight": max(n_neg / max(n_pos_train, 1), 1.0),
        "num_leaves": 31, "learning_rate": 0.1,
    }
    booster = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=100)
    model_path = "code/business_entity_resolution/src/model.txt"
    booster.save_model(model_path)
    print(f"[{time.time()-t0:.0f}s] model trained and saved to {model_path}")

    if val_rows:
        Xval = pd.DataFrame([r["feats"] for r in val_rows], columns=FEATURE_NAMES)
        probs = booster.predict(Xval)
        all_val_ids = [eid for eid, _ in s1_val]
        best_t, best_f05 = sweep_threshold(val_rows, probs, val_gt, all_val_ids)
        print(f"[{time.time()-t0:.0f}s] best threshold={best_t:.2f} -> val macro F0.5={best_f05:.4f}")
        with open("code/business_entity_resolution/src/threshold.json", "w") as f:
            json.dump({"threshold": best_t, "val_macro_f05": best_f05,
                       "val_recall_ceiling": ceiling, "n1": args.n1, "n23": args.n23,
                       "top_k": args.top_k, "per_route_k": args.per_route_k,
                       "final_top_k": args.final_top_k}, f, indent=2)
    else:
        print("no val rows produced - threshold not calibrated")

    importances = dict(zip(FEATURE_NAMES, booster.feature_importance()))
    print("feature importance:", sorted(importances.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
