"""
clean_tatqa.py
==============
Reusable TAT-QA data cleaning and normalization pipeline.
Week 2 -- Finance AI Research Project (Aug 28 - Sep 3, 2026)

Usage:
    from clean_tatqa import load_and_clean, SCALE_MULTIPLIERS

    train, dev, test = load_and_clean(raw_dir="path/to/dataset_raw")
"""

import json
import re
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE_MULTIPLIERS = {
    "":          1,
    "percent":   1,          # already a % -- no multiplication
    "thousand":  1_000,
    "million":   1_000_000,
    "billion":   1_000_000_000,
}

VALID_SCALES      = set(SCALE_MULTIPLIERS.keys())
VALID_ANSWER_TYPES = {"span", "multi-span", "arithmetic", "count", ""}

FISCAL_YEAR_PATTERNS = [
    r"fiscal year", r"fiscal\s+\d{4}", r"fy\s*\d{2,4}",
    r"year ended", r"year ending", r"twelve months ended",
    r"52.week", r"53.week",
]
CALENDAR_YEAR_PATTERNS = [
    r"calendar year", r"january \d{4}", r"december \d{4}",
    r"jan.*\d{4}", r"\bq[1-4]\b",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def try_float(val):
    """Try to parse val as float; return None on failure."""
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None


def normalize_answer_value(answer, scale):
    """
    Compute the canonical numeric value for a single numeric answer.
    Returns None for multi-span / text answers.
    """
    if not isinstance(answer, list) or len(answer) != 1:
        return None
    num = try_float(answer[0])
    if num is None:
        return None
    return round(num * SCALE_MULTIPLIERS.get(scale, 1), 6)


def detect_year_type(question_text):
    """
    Classify a question as referencing a fiscal / calendar / ambiguous /
    unspecified year based on keyword patterns.
    """
    q = question_text.lower()
    is_fiscal   = any(re.search(p, q) for p in FISCAL_YEAR_PATTERNS)
    is_calendar = any(re.search(p, q) for p in CALENDAR_YEAR_PATTERNS)
    if is_fiscal and not is_calendar:  return "fiscal"
    if is_calendar and not is_fiscal:  return "calendar"
    if is_fiscal and is_calendar:      return "ambiguous"
    return "unspecified"


def extract_years(question_text):
    """Extract all 4-digit years mentioned in the question text."""
    return sorted(set(re.findall(r"\b(19|20)\d{2}\b", question_text)))


def assign_reasoning_type(answer_type, answer_from, req_comparison):
    """Assign Week-1 reasoning type from schema fields."""
    at = answer_type or ""
    af = answer_from or ""
    rc = bool(req_comparison)
    if at == "":                                return "unknown (test)"
    if at == "arithmetic" and af == "table-text": return "multi-step"
    if rc:                                      return "comparison"
    if at == "arithmetic":                      return "arithmetic"
    if at == "count":                           return "count"
    if at == "multi-span":                      return "multi-span"
    return "lookup"

# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def clean_question(q, split="unknown"):
    """
    Normalize and enrich a single question dict.
    Gold answer, derivation, and rel_paragraphs are NEVER modified.
    All new fields are prefixed with 'meta_'.
    """
    cq = deepcopy(q)

    # ── Type coercions (repair broken fields) ────────────────────────────
    if cq.get("answer") is None:
        cq["answer"] = []
    if not isinstance(cq.get("answer"), list):
        cq["answer"] = [cq["answer"]]

    if cq.get("derivation") is None:
        cq["derivation"] = ""
    if not isinstance(cq.get("derivation"), str):
        cq["derivation"] = str(cq["derivation"])

    if cq.get("rel_paragraphs") is None:
        cq["rel_paragraphs"] = []

    for bool_field in ("req_comparison",):
        if cq.get(bool_field) is None:
            cq[bool_field] = False

    for str_field in ("scale", "answer_type", "answer_from"):
        if cq.get(str_field) is None:
            cq[str_field] = ""

    # ── Metadata enrichment ──────────────────────────────────────────────
    cq["meta_year_type"]        = detect_year_type(cq.get("question", ""))
    cq["meta_years_mentioned"]  = extract_years(cq.get("question", ""))
    cq["meta_has_derivation"]   = bool(cq["derivation"].strip())
    cq["meta_n_answers"]        = len(cq["answer"])
    cq["meta_canonical_value"]  = normalize_answer_value(cq["answer"], cq["scale"])
    cq["meta_scale_multiplier"] = SCALE_MULTIPLIERS.get(cq["scale"], 1)
    cq["meta_reasoning_type"]   = assign_reasoning_type(
        cq["answer_type"], cq["answer_from"], cq["req_comparison"]
    )
    return cq


def clean_record(rec, split="unknown"):
    """
    Normalize and enrich a full TAT-QA record (table + paragraphs + questions).
    """
    cr = deepcopy(rec)
    cr["meta_record_id"]   = f"{split}_{cr['table']['uid']}"
    cr["meta_split"]       = split
    cr["meta_n_paragraphs"] = len(cr.get("paragraphs", []))
    cr["meta_n_table_rows"] = len(cr["table"].get("table", []))
    cr["meta_n_table_cols"] = (
        len(cr["table"]["table"][0]) if cr["table"].get("table") else 0
    )
    cr["meta_n_questions"]  = len(cr.get("questions", []))
    cr["questions"] = [clean_question(q, split) for q in cr.get("questions", [])]
    return cr


def clean_split(data, split_name):
    """Clean an entire split (list of records)."""
    return [clean_record(rec, split_name) for rec in data]

# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

REQUIRED_Q_FIELDS = [
    "uid", "question", "answer", "answer_type", "answer_from",
    "scale", "req_comparison", "derivation", "rel_paragraphs",
]
REQUIRED_R_FIELDS = ["table", "paragraphs", "questions"]


def audit_split(data, split_name):
    """Return dict of {issue_type: [occurrences]} for a split."""
    from collections import Counter, defaultdict
    issues = defaultdict(list)
    q_uids = []
    for ri, rec in enumerate(data):
        for f in REQUIRED_R_FIELDS:
            if f not in rec:
                issues["missing_record_field"].append((ri, f))
        if not rec.get("table", {}).get("table"):
            issues["empty_table"].append(ri)
        if not rec.get("paragraphs"):
            issues["no_paragraphs"].append(ri)
        for qi, q in enumerate(rec.get("questions", [])):
            q_uids.append(q.get("uid", ""))
            for f in REQUIRED_Q_FIELDS:
                if f not in q:
                    issues["missing_q_field"].append((ri, qi, f))
            ans = q.get("answer", [])
            if ans is None:
                issues["null_answer"].append((ri, qi))
            elif not isinstance(ans, list):
                issues["non_list_answer"].append((ri, qi))
            elif len(ans) == 0 and split_name != "test":
                issues["empty_answer"].append((ri, qi))
            d = q.get("derivation")
            if d is not None and not isinstance(d, str):
                issues["non_str_derivation"].append((ri, qi))
            if q.get("scale", "") not in VALID_SCALES:
                issues["invalid_scale"].append((ri, qi, q.get("scale")))
            if q.get("answer_type", "") not in VALID_ANSWER_TYPES:
                issues["invalid_answer_type"].append((ri, qi, q.get("answer_type")))
    uid_counts = Counter(q_uids)
    dup_uids = {uid: cnt for uid, cnt in uid_counts.items() if cnt > 1}
    if dup_uids:
        issues["duplicate_q_uid"] = list(dup_uids.keys())
    return dict(issues)


def integrity_check(raw_data, clean_data):
    """
    Verify gold answers and derivations are identical between raw and cleaned.
    Returns list of (uid, issue_type, raw_value, clean_value) violations.
    """
    violations = []
    for raw_rec, clean_rec in zip(raw_data, clean_data):
        raw_qs   = {q["uid"]: q for q in raw_rec["questions"]}
        clean_qs = {q["uid"]: q for q in clean_rec["questions"]}
        for uid, raw_q in raw_qs.items():
            clean_q = clean_qs.get(uid)
            if clean_q is None:
                violations.append((uid, "QUESTION_MISSING", None, None))
                continue
            raw_ans   = raw_q.get("answer") or []
            clean_ans = clean_q.get("answer") or []
            if not isinstance(raw_ans, list):
                raw_ans = [raw_ans]
            if raw_ans != clean_ans:
                violations.append((uid, "ANSWER_ALTERED", raw_ans, clean_ans))
            raw_d   = raw_q.get("derivation") or ""
            clean_d = clean_q.get("derivation") or ""
            if raw_d != clean_d:
                violations.append((uid, "DERIVATION_ALTERED", raw_d, clean_d))
    return violations

# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def load_and_clean(raw_dir):
    """
    Load all three TAT-QA splits, clean them, run integrity check.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing tatqa_dataset_{train,dev,test}.json

    Returns
    -------
    (train_clean, dev_clean, test_clean) : tuple of lists
    """
    raw_dir = Path(raw_dir)

    def _load(name):
        with open(raw_dir / f"tatqa_dataset_{name}.json", encoding="utf-8") as f:
            return json.load(f)

    train_raw = _load("train")
    dev_raw   = _load("dev")
    test_raw  = _load("test")

    train_clean = clean_split(train_raw, "train")
    dev_clean   = clean_split(dev_raw,   "dev")
    test_clean  = clean_split(test_raw,  "test")

    # Integrity check
    for spl, raw, clean in [("train", train_raw, train_clean),
                              ("dev",   dev_raw,   dev_clean),
                              ("test",  test_raw,  test_clean)]:
        violations = integrity_check(raw, clean)
        if violations:
            raise RuntimeError(
                f"Gold integrity check FAILED for {spl}: {len(violations)} violations\n"
                f"First violation: {violations[0]}"
            )

    return train_clean, dev_clean, test_clean


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else r"d:\Inevitable\Finance-AI\TAT-QA\dataset_raw"
    print(f"Cleaning TAT-QA from: {raw_dir}")
    train, dev, test = load_and_clean(raw_dir)
    print(f"Done. train={len(train)} dev={len(dev)} test={len(test)} records")
    print("Gold integrity: PASS")
