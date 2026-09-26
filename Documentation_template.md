# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Draft status (2026-09-26): sections 1-4 below reflect the pipeline as
> actually built and code-reviewed. Section 5 (Results) is a placeholder —
> it needs real numbers from a full-scale run (see
> `code/business_entity_resolution/README.md`, "What's left") before this
> can be submitted. Update this note when Results is filled in for real.

---

## 1. Executive Summary

A classic blocking-plus-classifier pipeline: country-bucketed inverted-index
blocking narrows ~10M Source 2/3 candidate records down to a small,
capped-size candidate set per Source 1 entity, then a LightGBM binary
classifier scores each (Source 1, candidate) pair on 16 hand-engineered
name/address similarity features, with a decision threshold calibrated
directly against the official macro F0.5 metric on a held-out validation
split. No pretrained models, embeddings, or external data are used, so the
license/parameter-count constraint and the no-external-lookup rule are
satisfied by construction.

---

## 2. Methodology

### 2.1 Problem Analysis

Noise patterns identified in the training data (see `context.md` for the
dated log of this analysis): legal-suffix variation (Corp/Corporation,
Pvt/Private, Ltd/Limited), `&` vs. "and", word-order transpositions and
typos in names; address abbreviation variants (Rd/Road, St/Street),
landmark-based references ("Near X"), missing PIN/state components, and
municipal-numbering format differences between US and India addresses. The
test set adds a third country (France) absent from training, so every
normalization and blocking rule was written to be structural (regex/token-
based) rather than keyed to specific country names or formats — confirmed
empirically: a small test-set run detected France records and blocked/
scored them with no special-casing required.

### 2.2 Solution Strategy

**Approach type:** Blocking + Classifier (classic ML, no embeddings/LLM).

**Core innovation / key design choices:**
- Blocking combines three independent signals (IDF-weighted name tokens,
  exact PIN match, exact locality-token match) into one score, capped at a
  tunable top-K per Source 1 entity, rather than relying on a single
  blocking key — this is what keeps the candidate set small (a criterion
  the guidelines grade independently of match F0.5) while still catching
  matches that share an address but not a name, or vice versa.
- The candidate set fed to the model (`candidate_pairs.tsv`) and the model's
  actual inference input are the same object by construction (one function,
  `generate_candidates`/`candidates_for_entity`, used identically in
  training, candidate-file generation, and inference) — eliminates a class
  of bugs where the reported candidate set doesn't match what the model
  really saw.
- Decision threshold is chosen by directly maximizing the official macro
  F0.5 formula on a grouped (no-leakage) validation split, not a generic
  0.5 cutoff or a proxy metric like accuracy/AUC — since F0.5 weights
  precision 2x over recall, the right threshold is not the "natural"
  midpoint of the classifier's output.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:**
- Country bucket (exact match on normalized country string; open-vocabulary,
  no hard-coded country list — this is what lets France entities block
  correctly despite being unseen in training)
- IDF-weighted overlap on normalized, stopword-stripped name tokens (legal
  suffixes like Inc/Ltd/Pvt and stopwords like "and"/"the" excluded so they
  don't dominate the score)
- Exact match on parsed address PIN/ZIP-like digit run
- Exact match on parsed address locality token (last comma-separated
  address segment)

These three signals are combined into a single additive score per
candidate, then ranked and capped at `top_k` (currently 20, tunable via
`--top-k` on `train.py`/`predict.py`) — see `code/business_entity_resolution/src/blocking.py`.

**Candidate pairs generated:** [TODO — fill in from a full-scale run;
`code/business_entity_resolution/src/predict.py`'s printed summary reports
total rows and avg candidates/entity]

**How true matches were not lost (recall-ceiling measurement):**
`code/business_entity_resolution/src/eval.py`'s `blocking_recall_ceiling()`
computes, per validation-split Source 1 entity with at least one true match,
what fraction of its true matches are present in the candidate set — this
is measured automatically inside `train.py` and printed as "blocking recall
ceiling on val split." [TODO — replace with the real full-scale number;
small-sample runs during development showed low numbers, but that's a
sampling artifact of testing against a row-prefix subset of Source 2/3, not
a real result — see `context.md`, 2026-09-26 entries.]

---

## 4. Matching Model

**Features used** (16 total, see `code/business_entity_resolution/src/features.py::FEATURE_NAMES`):
- **Name features:** token Jaccard, Jaro-Winkler (via rapidfuzz WRatio),
  normalized Levenshtein similarity, token-sort ratio (word-order
  insensitive), common-token count, exact-match-after-suffix-stripping flag,
  length difference
- **Address features:** token Jaccard, normalized Levenshtein similarity,
  house-number exact-match flag, PIN exact-match flag, locality exact-match
  flag, length difference
- **Other:** same-country flag, the blocking-stage combined score
  (passed through as a feature, not recomputed), source indicator
  (Source 2 vs. Source 3, since each source's noise profile differs)

**Model type:** LightGBM binary classifier (`objective: binary`), MIT-
licensed, no parameter-count concern (a gradient-boosted tree ensemble, not
an LLM) — trivially satisfies the "MIT/Apache-2.0, ≤ 8B parameters"
constraint. Class imbalance (true matches are a small fraction of all
candidate pairs) handled via `scale_pos_weight`.

**Threshold selection method:** Grid sweep over thresholds 0.05–0.95
(step 0.05) on a held-out validation split, evaluated with the exact
official F0.5 macro-average formula (`code/business_entity_resolution/src/eval.py::macro_f_beta`),
selecting the threshold that maximizes it. Validation split is grouped by
Source 1 entity (no leakage of a matched pair across train/val). Result
saved to `threshold.json` alongside the model.

---

## 5. Results & Error Analysis

**[TODO — placeholder, not yet filled in with real numbers.]**
Requires a full-scale run per `code/business_entity_resolution/README.md`.
Do not submit this document with this section still marked TODO.

- **F_0.5 Score (macro), validation split:** TODO
- **Blocking recall ceiling, validation split:** TODO
- **Avg candidates per Source 1 entity (candidate_pairs.tsv):** TODO
- **Common false positives (wrong merges):** TODO — inspect val-split rows
  where the model predicted a match but ground truth disagrees; group by
  which features were high despite being a non-match
- **Common false negatives (missed matches):** TODO — split further into
  "missed at blocking stage" (true match never appeared in candidates —
  blocking recall-ceiling problem) vs. "missed at model stage" (candidate
  was present but scored below threshold)

---

## 6. Conclusion

[TODO — 2-3 sentences once Results is filled in.]

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`:
- `src/normalize.py` — text normalization (name/address/country)
- `src/blocking.py` — candidate generation (country bucket + inverted index)
- `src/features.py` — pairwise feature engineering
- `src/build_candidates.py` — data loading + index-building glue
- `src/train.py` — trains the model, calibrates the decision threshold
  against F0.5 on a held-out split; entry point to reproduce `model.txt` +
  `threshold.json`
- `src/eval.py` — official F0.5 scorer + blocking recall-ceiling helper
- `src/predict.py` — entry point to reproduce `output/matching_results.tsv`
  and `output/candidate_pairs.tsv` from `model.txt` + `threshold.json`

See `code/business_entity_resolution/README.md` for exact setup and run
commands, current status, and known shortcomings, and repo-root
`context.md` for the full dated development log.

### B. Additional Results

[TODO — charts/tables once available.]

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
