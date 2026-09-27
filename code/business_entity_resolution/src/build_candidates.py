"""Runs normalize + blocking end-to-end over real source files, producing
candidate_pairs for a given S1 slice against full S2/S3.

Usage (as a library, imported by train.py / predict.py):
    from build_candidates import load_and_normalize, build_indexes, generate_candidates
"""
import sys
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from normalize import normalize_row
from blocking import build_country_bucket_indexes, candidates_for_entity


def load_and_normalize(path, nrows=None, sample_n=None, seed=42):
    """Returns list of (entity_id, norm_dict) for one source file.

    nrows: read only the first N rows (fast I/O cap; NOT a representative
        sample -- fine for S2/S3 dev-mode row caps, since the final run
        uses the full files anyway).
    sample_n: read the full file, then take a true random sample of
        sample_n rows before normalizing (representative; use this for S1
        when the validation split needs to reflect the whole population,
        not just the file's row order). Ignored if nrows is also set and
        smaller than sample_n would require (nrows still caps I/O first).
        Mutually informative with nrows: if both given, reads nrows rows
        then randomly samples sample_n of those (rarely what you want --
        prefer passing sample_n alone for a true population-level sample).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed)
    out = []
    for row in df.itertuples(index=False):
        norm = normalize_row(row.business_name, row.business_address, row.country)
        out.append((row.entity_id, norm))
    return out


def bucket_by_country(records):
    buckets = {}
    for eid, norm in records:
        buckets.setdefault(norm["country_norm"], []).append((eid, norm))
    return buckets


def build_indexes(s2_records, s3_records):
    """Returns dict: country -> (name_idx, name_idf, pin_idx, loc_idx, source_flag_lookup)."""
    s2_by_country = bucket_by_country(s2_records)
    s3_by_country = bucket_by_country(s3_records)
    countries = set(s2_by_country) | set(s3_by_country)
    indexes = {}
    is_source2 = {eid: True for eid, _ in s2_records}
    is_source2.update({eid: False for eid, _ in s3_records})
    for country in countries:
        combined = s2_by_country.get(country, []) + s3_by_country.get(country, [])
        combined_by_country = {country: combined}
        name_idx, name_idf, pin_idx, loc_idx = build_country_bucket_indexes(combined_by_country, country)
        indexes[country] = (name_idx, name_idf, pin_idx, loc_idx)
    return indexes, is_source2


def generate_candidates(s1_records, indexes, top_k=20, per_route_k=None, final_top_k=None):
    """Returns {s1_entity_id: [(cand_id, score), ...]}"""
    out = {}
    for eid, norm in s1_records:
        country = norm["country_norm"]
        if country not in indexes:
            out[eid] = []
            continue
        name_idx, name_idf, pin_idx, loc_idx = indexes[country]
        cands = candidates_for_entity(
            norm["name_tokens"], norm["addr_pin"], norm["addr_locality"],
            name_idx, name_idf, pin_idx, loc_idx,
            top_k=top_k, per_route_k=per_route_k, final_top_k=final_top_k)
        out[eid] = cands
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    N = 5000
    s1 = load_and_normalize("dataset/train/train_source1.tsv", nrows=N)
    s2 = load_and_normalize("dataset/train/train_source2.tsv", nrows=200000)
    s3 = load_and_normalize("dataset/train/train_source3.tsv", nrows=200000)
    print(f"loaded {len(s1)} S1, {len(s2)} S2, {len(s3)} S3 in {time.time()-t0:.1f}s")
    t1 = time.time()
    indexes, is_source2 = build_indexes(s2, s3)
    print(f"built indexes for {len(indexes)} countries in {time.time()-t1:.1f}s")
    t2 = time.time()
    cands = generate_candidates(s1, indexes, top_k=20)
    print(f"generated candidates for {len(cands)} S1 entities in {time.time()-t2:.1f}s")
    n_with_cands = sum(1 for v in cands.values() if v)
    avg_cands = sum(len(v) for v in cands.values()) / max(len(cands), 1)
    print(f"entities with >=1 candidate: {n_with_cands}/{len(cands)}, avg candidates/entity: {avg_cands:.2f}")
    sample = list(cands.items())[:3]
    print("sample:", sample)
