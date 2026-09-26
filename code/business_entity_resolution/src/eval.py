"""Official F0.5 macro-average scorer, matching the challenge's exact formula.
Used for local validation (train.py holdout, blocking tuning, threshold search)."""


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted: set, truth: set):
    """Returns F0.5 for one S1 entity. Singleton (empty truth) -> 1.0 if predicted
    also empty, else 0.0 (any false merge on a true singleton is fully penalized)."""
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_beta(precision, recall, beta=0.5)


def macro_f_beta(predictions: dict, truth: dict, all_entity_ids=None):
    """predictions, truth: {s1_entity_id: set(matched_ids)}.
    all_entity_ids: optional full id list to ensure every entity is scored
    (missing predictions treated as empty set, per contest scoring)."""
    ids = all_entity_ids if all_entity_ids is not None else set(truth) | set(predictions)
    scores = []
    for eid in ids:
        pred = predictions.get(eid, set())
        true = truth.get(eid, set())
        scores.append(score_entity(pred, true))
    return sum(scores) / len(scores) if scores else 0.0


def blocking_recall_ceiling(candidate_sets: dict, truth: dict):
    """% of true-match ids present in the candidate set, per entity, averaged
    only over entities that actually have true matches (singletons trivially pass)."""
    ratios = []
    for eid, true_ids in truth.items():
        if not true_ids:
            continue
        cand = candidate_sets.get(eid, set())
        found = len(true_ids & cand)
        ratios.append(found / len(true_ids))
    return sum(ratios) / len(ratios) if ratios else 1.0
