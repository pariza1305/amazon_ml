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
