# Business Entity Resolution — Amazon ML Challenge 2026

Status as of 2026-09-26 (Day 2 of 3). **Not competition-ready yet** — see
"What's left" below before assuming this can be submitted as-is.

Full running log of every change, decision and caveat: **`../../context.md`**
(repo root). Read that first if you want the blow-by-blow; this file is the
setup + handoff summary.

## Quick setup (new machine / new contributor)

```bash
# 1. Clone the repo, then get the challenge data into place yourself
#    (not in git — see "Data" below):
#    dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
#    dataset/test/{test_source1,test_source2,test_source3}.tsv
#    utils/validate_submission.py
#    Documentation_template.md
#    All of these go at the REPO ROOT (C:\dev\amazon_ml locally), not under this folder.

# 2. Python 3.10+, then:
pip install -r requirements.txt

# 3. Sanity-check the pipeline on a small sample (~2 min):
cd <repo_root>
python3 code/business_entity_resolution/src/train.py --n1 5000 --n23 300000 --top-k 20
```

If step 3 prints a trained model + feature importances with no errors, the
pipeline is wired correctly on your machine.

## Data

The dataset (~2.5GB, 7 TSV files) is **git-ignored** — too big for the repo.
Whoever needs it downloads `student_resource.zip` from the Unstop portal and
extracts `dataset/`, `utils/validate_submission.py`, and
`Documentation_template.md` into the repo root.

Row counts (train/test): S1 ≈ 2.2M/1.7M, S2 ≈ 5.0M/4.9M, S3 ≈ 5.3M/5.1M.

## Pipeline (repo-root-relative paths)

| File | Purpose | Status |
|---|---|---|
| `code/business_entity_resolution/src/normalize.py` | Name/address/country text normalization | Done, unit-tested |
| `code/business_entity_resolution/src/blocking.py` | Country-bucketed inverted-index candidate generation | Done, tested on synthetic + small real-data samples |
| `code/business_entity_resolution/src/features.py` | 16 pairwise similarity features | Done, unit-tested |
| `code/business_entity_resolution/src/build_candidates.py` | Loads real TSVs, builds indexes, generates candidates | Done, smoke-tested on a small row sample |
| `code/business_entity_resolution/src/train.py` | Grouped train/val split, labels candidates vs ground truth, trains LightGBM, calibrates decision threshold against F0.5 on val, saves `model.txt` + `threshold.json` | Runs end-to-end on a small sample |
| `code/business_entity_resolution/src/eval.py` | Official F0.5 macro scorer + blocking recall-ceiling helper | Done, verified against the guideline's own worked example |
| `code/business_entity_resolution/src/predict.py` | Full test-set inference → `output/matching_results.tsv` + `output/candidate_pairs.tsv`, with dedup/subset safety checks | Done, **format-validated against the organizer's own `utils/validate_submission.py`** (one real header bug found + fixed) |

`normalize` + `blocking` + `features` are library modules imported by
`train.py` (fit) and `predict.py` (infer) — you don't run them directly.
Run order for a full cycle: `train.py` first (produces `model.txt` +
`threshold.json`), then `predict.py` (consumes both, produces the two
output TSVs).

## What's left (in priority order)

The pipeline is functionally complete end-to-end (normalize -> blocking ->
features -> train w/ val split + threshold calibration -> predict ->
format-validated output). Everything below is "run it for real and tune it,"
not "build something new."

1. **Full-scale run.** Everything so far has only been run on small
   row-prefix samples (a few hundred thousand rows) for speed. The real
   files are 5M+ rows per source; a full run has NOT been done and current
   F0.5/recall/candidate-count numbers are NOT meaningful — see the
   "IMPORTANT CAVEAT" entries in `context.md` dated 2026-09-26. **This needs
   to run in a real terminal with no timeout.** Rough sizing:
   loading+normalizing ~10.6M S2+S3 rows alone took ~48s per 600K rows in
   testing, so full-scale is on the order of tens of minutes; budget
   accordingly. If it's too slow, the bottleneck is `normalize.py` running
   row-by-row in pure Python — vectorizing it with pandas `.apply`/regex-on-
   Series or multiprocessing is the obvious speedup.

   Commands, run from the repo root, in order:
   ```bash
   # 1. Train on the FULL training set (omit --n1/--n23 to use every row;
   #    pass them only to bound a debug run). Produces model.txt + threshold.json.
   python3 code/business_entity_resolution/src/train.py

   # 2. Run inference on the FULL test set. Produces output/matching_results.tsv
   #    and output/candidate_pairs.tsv.
   python3 code/business_entity_resolution/src/predict.py

   # 3. Validate output format before uploading anywhere.
   python3 utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test --check-ids
   ```
2. **Blocking tuning** — once step 1 gives real recall-ceiling numbers,
   sweep `--top-k` and the blocking keys in `blocking.py` to trade off recall
   vs. candidate-set size (the latter is separately graded — see
   guidelines). Re-run `train.py` after any blocking change since it affects
   what the model trains on.
3. **Re-check threshold calibration** on the real val split (train.py does
   this automatically, but sanity-check `threshold.json`'s val F0.5 looks
   reasonable, not like the ~0.13 seen on tiny samples).
4. **Upload `output/matching_results.tsv` to the leaderboard** — nothing has
   been submitted yet. Do this as soon as step 1 produces a valid file, even
   before tuning — a mediocre scored submission beats no submission.
5. **Fill in `Documentation_template.md`** (repo root) — a first draft of
   the methodology/blocking/features sections already exists; the Results
   section still needs real numbers from step 1.
6. **Assemble the final `<team_name>_submission.zip`** per the guidelines'
   exact structure once everything above is done.

## Known shortcomings / things to double-check

- `blocking.py`'s recall ceiling has never been measured against real data —
  only a hand-built synthetic example. It could easily be too aggressive
  (missing true matches) or too loose (large candidate sets, hurting the
  separately-graded blocking score). Measure before trusting it.
- `model.txt` currently in the repo was trained on a 5K-row sample — **do
  not use it for a real submission**, retrain after the full-scale data
  pipeline exists.
- No handling yet for country values outside the blocking index at
  inference time beyond returning an empty candidate list (correct behavior
  for a genuine singleton, but confirm this is actually what happens for
  France once test data is run through it — it's untested on non-US/India
  data since training has none).
- `git` is currently NOT being used for this work session (by explicit
  request) — `context.md` is the only change history until someone commits.
  **Before pushing to GitHub, review `context.md` and consider squashing it
  into a normal commit history**, or at least commit it as-is so the log
  travels with the repo.

## For a new contributor picking this up cold

1. Read `context.md` top to bottom — it's the real history.
2. Read this file's "What's left" section and pick the next unchecked item.
3. Run the Quick Setup sanity check above before changing anything, to
   confirm your environment matches what's been validated so far.
4. Keep appending to `context.md` as you go (same format: dated entries,
   what changed, what was tested, what's known-broken) — that's the
   single source of truth for where things stand.
