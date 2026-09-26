"""Text normalization for business names and addresses. Pure offline string
transforms - no network/geocoding calls, no country hard-coding."""
import re
import unicodedata

LEGAL_SUFFIXES = {
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt",
    "company": "co", "co": "co",
    "llc": "llc", "llp": "llp", "plc": "plc",
    "lp": "lp", "pc": "pc", "lc": "lc",
    "pllc": "pllc", "gmbh": "gmbh", "sa": "sa",
}
ADDR_ABBR = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl",
    "apartment": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg",
    "floor": "fl", "fl": "fl",
    "suite": "ste", "ste": "ste",
    "unit": "unit",
}
STOPWORDS_NAME = {"the", "and", "of", "a", "an"}
_PUNCT_RE = re.compile(r"[^\w\s&]")
_WS_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\b(\d{4,6})\b")
_LEADING_NUM_RE = re.compile(r"^(\d+[a-zA-Z]?)\b")
_NEAR_RE = re.compile(r"\bnear\b.*$", re.IGNORECASE)


def _base_clean(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_name(raw: str):
    """Returns dict of normalized name variants used downstream."""
    s = _base_clean(raw)
    tokens = s.split()
    canon_tokens = [LEGAL_SUFFIXES.get(t, t) for t in tokens]
    core_tokens = [t for t in canon_tokens if t not in LEGAL_SUFFIXES and t not in STOPWORDS_NAME]
    return {
        "name_norm": " ".join(canon_tokens),          # suffix-canonicalized, order preserved
        "name_core": " ".join(core_tokens),             # suffixes/stopwords stripped, order preserved
        "name_core_sorted": " ".join(sorted(core_tokens)),  # word-order-insensitive
        "name_tokens": tuple(core_tokens),
    }


def normalize_address(raw: str):
    s = _base_clean(raw)
    s_no_landmark = _NEAR_RE.sub("", s).strip()
    tokens_full = s_no_landmark.split()
    norm_tokens = [ADDR_ABBR.get(t, t) for t in tokens_full]
    leading = _LEADING_NUM_RE.match(s_no_landmark)
    house_no = leading.group(1) if leading else ""
    tail = s_no_landmark[len(house_no):] if house_no else s_no_landmark
    digit_runs = _DIGIT_RUN_RE.findall(tail)
    pin = digit_runs[-1] if digit_runs else ""
    locality = ""
    parts = [p.strip() for p in raw.split(",")] if isinstance(raw, str) else []
    if parts:
        locality = _base_clean(parts[-1])
    return {
        "addr_norm": " ".join(norm_tokens),
        "addr_tokens": tuple(norm_tokens),
        "addr_pin": pin,
        "addr_house_no": house_no,
        "addr_locality": locality,
    }


def normalize_country(raw: str) -> str:
    return _base_clean(raw)


def normalize_row(name: str, address: str, country: str) -> dict:
    out = {}
    out.update(normalize_name(name))
    out.update(normalize_address(address))
    out["country_norm"] = normalize_country(country)
    return out
