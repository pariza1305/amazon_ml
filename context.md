# Context Log — Amazon ML Challenge 2026 (Entity Resolution)

Running log of what's been done, decisions made, and current state. Append, don't rewrite history.

## Repo layout
- `` = contest root (matches required paths: dataset/, utils/, output/, code/, Documentation_template.md)
- `dataset/` is git-ignored (huge: ~2.5GB total, train S1=2.2M rows, S2=5.0M, S3=5.3M; test S1=1.7M, S2=4.9M, S3=5.1M rows)
- `code/business_entity_resolution/src/` = pipeline source
- Full plan doc (phases 0-9): https://claude.ai/artifact/9hz5kE7sNWdS1HQPZkUAYQ

## Environment
- Python 3.10.12, pandas 2.3.3 pre-installed
- Installed: lightgbm, scikit-learn, rapidfuzz

## Progress log

### 2026-09-25 — Phase 0/2: setup + normalize.py
- Dataset placed by user in `dataset/{train,test}/`
- Created dirs: `code/business_entity_resolution/src/`, `output/`
- Row counts confirmed (see above)
- Built `src/normalize.py`:
  - `normalize_name(raw)` → name_norm (suffix-canonicalized), name_core (suffix/stopword-stripped), name_core_sorted (order-insensitive), name_tokens
  - `normalize_address(raw)` → addr_norm, addr_tokens, addr_pin, addr_house_no, addr_locality (landmark phrases like "near X" stripped before parsing)
  - `normalize_country(raw)` → plain casefold, no hard-coded country list
  - Fixed bug: leading house number was double-counted as PIN (4-6 digit regex matched it) — now PIN search excludes the house-number prefix
  - Tested manually on India + US sample rows, looks correct
- Committed to git. .gitignore added (excludes dataset/, __pycache__, __MACOSX)

## Next up
- Phase 3: `src/blocking.py` — country bucketing + inverted-index blocking (name tokens, address tokens/pin) + top-K cap
- Then Phase 4 features, Phase 5 train.py, Phase 6 eval.py, Phase 7 predict.py

### 2026-09-26 — Phase 3: blocking.py
- Built `src/blocking.py`: country bucketing (dict keyed by country_norm, no hard-coded list) + inverted-index blocking on IDF-weighted name tokens + exact PIN + locality tokens
- `candidates_for_entity()` combines all 3 signals into one score, caps at top_k (default 20)
- Tested on a small synthetic India example: correctly picked the true match, excluded an unrelated record
- Not yet run against real data / not yet tuned (recall ceiling vs candidate size) — that's the Phase 6 tuning loop, comes after features+train exist so we have something to validate end-to-end

### 2026-09-26 — repo flattened + Phase 4: features.py
- Moved everything from `student_resource/` up to repo root per user request: `dataset/`, `code/`, `output/`, `utils/`, `Documentation_template.md` now live directly under `C:\dev\amazon_ml`. Renamed contest `README.md` -> `CHALLENGE_BRIEF.md` to avoid clobbering repo root README. Updated `.gitignore` paths accordingly. Git recorded these as renames (history preserved).
- Built `src/features.py`: 16 pairwise features (FEATURE_NAMES) covering name similarity (jaccard/jaro-winkler/levenshtein/token-sort/common-tokens/exact-match/length-diff), address similarity (jaccard/levenshtein/house-number/pin/locality match/length-diff), same_country flag, block_score passthrough, source indicator
- `add_relative_rank_features()`: post-scoring helper, adds rank-within-entity and score-gap-to-next-best (per S1 entity) — used to help the model separate a clear best match from several mediocre look-alikes
- Tested `compute_features()` on a typo'd name-match example, values look sane (high jaro-winkler/levenshtein despite low token-jaccard on the typo, as expected)

### 2026-09-26 — git usage stopped
- Per user instruction: no more git add/commit from here on. Working tree only, no version snapshots via git for now.

### 2026-09-26 — build_candidates.py, train.py, eval.py; environment limitation found
- Built `src/build_candidates.py`: load_and_normalize (pandas -> normalize_row per row), bucket_by_country, build_indexes (per-country name/pin/locality indexes), generate_candidates. Smoke-tested on 5K S1 vs 400K sampled S2+S3 rows — wired correctly, but recall/candidate numbers from a small row-prefix SAMPLE are not meaningful (see below).
- Built `src/train.py`: labels candidates against train_ground_truth, computes features for every (S1, candidate) pair, trains a LightGBM binary classifier (scale_pos_weight for class imbalance), saves `model.txt`, prints feature importances. Ran on 5K S1 / 300K+300K S2+S3 sample — pipeline runs end-to-end, model trains. Feature importance sane: block_score and addr similarity dominate, same_country/name_exact_core near-zero importance (expected, since blocking already filters to same country almost always).
- **IMPORTANT CAVEAT**: recall-ceiling numbers from these small runs (e.g. "698/4711") are NOT real — they're an artifact of loading only the first N rows of S2/S3 (5.3M rows total each), so most true matches for the sampled S1 entities simply aren't in that row-prefix sample. Real recall-ceiling/candidate-size numbers require running against the FULL S2/S3 files.
- Built `src/eval.py`: exact F0.5 formula (macro-averaged per S1 entity, singleton = 1.0/0.0 handling). Verified against the guideline's own worked example (0.714) — passes.
- **Environment limitation discovered**: the device-bridge shell (mcp__remote-devices__device_bash) runs each call in a fresh sandboxed process; `nohup ... &` background jobs do NOT survive between calls (confirmed: process was gone on next call). Each call is capped ~170s. Pure-Python row-by-row normalization over the full 10.6M S2+S3 rows would take ~850s+ just to load/normalize — too long for this per-call model.
- **Decision**: keep validating pipeline logic here on small samples (fast feedback), then hand off a single full-scale command for the user to run directly in their own terminal (unlimited time there) once the pipeline is complete end-to-end (train + predict + package). Will flag clearly when that handoff point is reached.
- Cleaned up: removed a dead/unused `build_training_table` function left over from an early train.py draft.

## Next up
- `predict.py`: run full inference pipeline (normalize -> blocking -> features -> trained model -> threshold/grouping) on test sources, write matching_results.tsv + candidate_pairs.tsv, enforce all output-format rules
- Threshold calibration against macro F0.5 using a held-out validation split (currently train.py has no train/val split — needs adding before threshold tuning is meaningful)
- Then: the full-scale run handoff (see above)

### 2026-09-26 — handoff documentation
- Added `code/business_entity_resolution/requirements.txt` (pandas, lightgbm, scikit-learn, rapidfuzz)
- Added `code/business_entity_resolution/README.md`: setup steps, data placement instructions, pipeline file table with status, prioritized "what's left" list, known shortcomings, and a "new contributor picking this up cold" section — written so a colleague can clone the repo and continue without needing the user in the loop. Points back to this context.md as the detailed history.
- User plans to push to GitHub at some point for colleagues to continue; flagged in the README that git hasn't been used this session (per user request) and context.md is the only history until someone commits it.

### 2026-09-26 — train/val split + threshold calibration added to train.py
- Rewrote `src/train.py`: splits S1 entities (grouped, no leakage) into train/val by `--val-frac` (default 0.2), trains only on train rows, sweeps decision threshold 0.05-0.95 on val rows to maximize macro F0.5 (via eval.py), saves `threshold.json` (chosen threshold + val F0.5 + val recall ceiling + the sample sizes used).
- Tested on 5K S1 / 300K+300K S2+S3 sample: ran end-to-end, val macro F0.5 = 0.127 at threshold 0.80 — LOW NUMBER IS EXPECTED, same sampling artifact as before (val recall ceiling only 0.042 on this sample, i.e. blocking barely sees any true matches because S2/S3 are only a 300K-row prefix of 5M+ rows). This is a pipeline-correctness check, not a real performance number.
- `model.txt` and `threshold.json` in the repo right now are from this tiny sample — explicitly NOT for submission (README.md already flags this for model.txt; threshold.json has the same caveat).

### 2026-09-26 — predict.py built, format-validated against organizer's validator
- Built `src/predict.py`: full test-set inference (normalize -> blocking -> features -> trained model -> threshold from threshold.json -> matching_results.tsv + candidate_pairs.tsv), with dedup/subset-safety checks on candidate IDs before writing.
- Ran on a 2K-row test-S1 debug slice against a 300K-row S2/S3 sample. Confirmed **France entities flow through with no special-casing** (3 countries detected: france, india, us) — the no-hardcoding requirement is satisfied by construction.
- Ran organizer's `utils/validate_submission.py` against the output — caught and fixed a real bug: `candidate_pairs.tsv` was being written with the `matched_entity_ids` header instead of the required `candidate_entity_ids` header. Fixed; both files now pass every format rule except "missing S1 entities," which is expected and correct given we intentionally ran on a 2K-row (not full 1.73M-row) debug slice.
- `--check-ids` validator flag (checks matched/candidate IDs actually exist in test set) not yet run — needs full S2/S3 loaded, deferred to the full-scale run.

## Status summary (all phases)
- Done + tested at small scale: normalize, blocking, features, eval (F0.5 scorer, verified against guideline example), train (with val split + threshold calibration), predict (format-validated against organizer's script)
- NOT done: full-scale run (see prior caveat — needs a real terminal, not this per-call sandboxed bridge), blocking/model tuning using real numbers, Documentation_template.md fill-in, final zip packaging
- Every piece of the pipeline now exists end-to-end; what's left is running it at real scale and tuning based on real numbers, not building new components

### 2026-09-26 — documentation pass for git commit
- Updated `code/business_entity_resolution/README.md`: status table now reflects predict.py as done + format-validated; "What's left" rewritten to reflect the pipeline being functionally complete, with exact copy-pasteable commands for the full-scale run (train.py -> predict.py -> validate_submission.py --check-ids).
- Filled in a first draft of `Documentation_template.md` (repo root): Executive Summary, Methodology, Candidate Generation/Blocking, Matching Model sections written from the actual implementation. Results section explicitly left as TODO with a clear placeholder note at the top of the file — do not submit until that's replaced with real numbers.
- Per user request, resuming git usage from this point: about to run `git add` + `git commit` (still no push) so this state is committed for colleagues to pull.
