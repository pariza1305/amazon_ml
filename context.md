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

### 2026-09-27 — blocking.py rewritten: independent routes + union (teammates' finding)
- Teammates ran real experiments on Kaggle (full compute, no timeout there) with the OLD single-combined-score blocking:
  - 1K S1 / 5K+5K S2/S3, top_k=20: recall=0.3%, F0.5=0.055 (sanity-check scale only, not meaningful)
  - 20K S1 / 1M+1M, top_k=20: recall=13.2%, F0.5=0.283
  - 20K S1 / 5M+5M, top_k=20: recall=58.8%, F0.5=0.670
  - 20K S1 / 5M+5M, top_k=50: recall=63.7%, F0.5=0.701
  - Diagnosis: ~36% of true matches never reach the classifier (blocking recall ceiling, not model quality, is the #1 bottleneck). Also flagged: current --n1 takes the FIRST N rows of S1, not a random sample, so validation isn't representative.
- **Rewrote `blocking.py`**: `candidates_for_entity` now runs three INDEPENDENT routes (name-token IDF score, exact PIN, exact locality), ranks+caps each on its own (`per_route_k`, defaults to `top_k`), then UNIONS them, instead of one combined score with a single global top-K cutoff. Rationale: a true match with a strong PIN/locality signal but weak/no name overlap can get crowded out of a combined ranking by many candidates sharing a moderately-rare name token (their IDF weight can exceed the flat PIN/locality bonus once the corpus is large) -- independent routes guarantee it survives regardless of how the other route ranks. `final_top_k` param added for an optional cap AFTER the union (default None = no cap, prioritizing recall per teammates' stated target of ~80-90%, until size/recall tradeoff is tuned).
- **Fixed S1 sampling**: `build_candidates.load_and_normalize` gained a `sample_n` param that reads the FULL file then takes a true random sample (`df.sample`), vs `nrows` which is still available for S2/S3 dev-mode row caps (order-independent, since final run uses full files anyway). `train.py --n1` now uses `sample_n` (true random sample) instead of `nrows` (first-N).
- Updated `train.py`/`predict.py`: new `--per-route-k` / `--final-top-k` flags, `threshold.json` now also records the blocking params used so `predict.py` can reuse them automatically.
- **Verified the fix with a deterministic synthetic test** (not real-scale data — see below for why): constructed 25 noise candidates with inflated IDF-weighted name scores (~6.4 each, exceeding the PIN route's flat +5 bonus) plus one true match sharing zero name tokens but the correct PIN. OLD-style single-combined-score top-20: true match dropped (False). NEW route-union: true match kept (True). This proves the mechanism is fixed; it does not by itself give a new real recall-ceiling number.
- **IMPORTANT LIMITATION FOUND**: this cloud-to-laptop bridge (device_bash) caps every call at ~178s. Tried to reproduce teammates' 1M+1M experiment here for a real before/after comparison; the index build alone for 500K+500K records did not finish within that window (confirmed via `timeout 170` + redirected log: load finished at 92s, index build for 1M combined records still running when killed at 170s). **This environment cannot reproduce Kaggle-scale runs** -- real recall-ceiling numbers for the new blocking must come from teammates re-running on Kaggle, not from here.
- Ran small-scale sanity checks only (n1=3000, n23=300K): pipeline runs end-to-end with the new code, no errors, output format unaffected. Recall ceiling at this scale is still dominated by the "S2/S3 row-prefix sample is tiny relative to 5M+ real files" artifact, same as always -- not informative about the union-blocking fix's real impact.

## Next up (for whoever runs this on Kaggle)
Re-run the SAME experiment sizes your teammates already used, with this updated code, for an apples-to-apples before/after:
```bash
# matches "Experiment 2" (was 13.2% recall / F0.5=0.283 with old blocking)
python3 code/business_entity_resolution/src/train.py --n1 20000 --n23 1000000 --top-k 20

# matches "Experiment 3" (was 58.8% recall / F0.5=0.670)
python3 code/business_entity_resolution/src/train.py --n1 20000 --n23 5000000 --top-k 20

# matches "Experiment 4" (was 63.7% recall / F0.5=0.701)
python3 code/business_entity_resolution/src/train.py --n1 20000 --n23 5000000 --top-k 50
```
Compare the printed "blocking recall ceiling on val split" against the numbers above. If it's meaningfully higher (target ~80-90% per teammates' stated goal), the union-blocking fix is validated at real scale; if not, the per_route_k/final_top_k knobs need further tuning (try raising per_route_k independently of top_k, or check whether locality-route candidate counts are exploding and drowning the union in noise -- that's the most likely secondary failure mode of this fix, untested at real scale).

Also note: `--n1` now takes a TRUE RANDOM sample (not first-N) as of this change -- numbers won't be bit-for-bit comparable to teammates' logged runs for that reason alone, on top of the blocking change. If an apples-to-apples read on blocking ALONE is wanted first, could temporarily pin `--seed` and compare, or test on the first-20K-rows-equivalent by another means -- not done here, flagging as an option.
