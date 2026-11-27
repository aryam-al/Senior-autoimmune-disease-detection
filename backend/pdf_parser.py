"""Extract the 23 model inputs from a lab-report PDF.

Values that cannot be found are reported as missing - they are never filled with
made-up defaults, because a silent default changes the prediction.
"""

import re

import pymupdf as fitz

from features import BINARY, CONTINUOUS, FEATURES

# How far after a label we look for its value / result (characters).
WINDOW = 80

# Label patterns. Order inside each list does not matter.
LABELS = {
    "ESR": [r"\besr\b", r"sedimentation\s+rate", r"\bsed\s+rate"],
    "CRP": [r"\b(?:hs[\s-]?)?crp\b", r"c[\s-]?reactive\s+protein"],
    "C3": [r"\bc3\b", r"complement\s+c3"],
    "C4": [r"\bc4\b", r"complement\s+c4"],
    "RF": [r"\brf\b", r"rheumatoid\s+factor"],
    "Anti_CCP": [r"\banti[\s-]?ccp\b", r"\bccp\b", r"cyclic\s+citrullinated"],
    "HLA_B27": [r"\bhla[\s-]?b[\s-]?27\b", r"\bb27\b"],
    "ANA": [r"\bana\b", r"anti[\s-]?nuclear\s+antibod", r"antinuclear"],
    "Anti_Ro": [r"\banti[\s-]?ro\b", r"\banti[\s-]?ssa\b", r"\bssa\b", r"\bro\s*/\s*ssa\b"],
    "Anti_La": [r"\banti[\s-]?la\b", r"\banti[\s-]?ssb\b", r"\bssb\b", r"\bla\s*/\s*ssb\b"],
    "Anti_dsDNA": [r"\banti[\s-]?ds[\s-]?dna\b", r"\bds[\s-]?dna\b", r"double[\s-]stranded\s+dna"],
    "Anti_Sm": [r"\banti[\s-]?sm\b(?!a)", r"\bsmith\s+antibod"],
    "ASCA": [r"\basca\b", r"saccharomyces"],
    "Anti_CBir1": [r"\banti[\s-]?cbir[\s-]?1\b", r"\bcbir[\s-]?1\b"],
    "Anti_OmpC": [r"\banti[\s-]?ompc\b", r"\bompc\b"],
    "pANCA": [r"\bp[\s-]?anca\b", r"perinuclear\s+anca"],
    "EMA": [r"\bema\b", r"endomysial"],
    "DGP": [r"\bdgp\b", r"deamidated\s+gliadin"],
    "Anti_tTG": [r"\banti[\s-]?t[\s-]?tg\b", r"\bt[\s-]?tg\b", r"transglutaminase"],
    "Anti_TPO": [r"\banti[\s-]?tpo\b", r"\btpo\b", r"thyroid\s+peroxidase"],
    "Anti_SMA": [r"\banti[\s-]?sma\b", r"smooth\s+muscle\s+antibod"],
}

# Negative words are checked first: "non-reactive" contains "reactive".
_NEG = re.compile(r"\b(negative|neg|non[\s-]?reactive|not\s+detected|undetected|absent|nil)\b|(?<![\w+])-ve\b")
_POS = re.compile(r"\b(positive|pos|reactive|detected|present)\b|\+ve\b|(?<=\s)\+(?=\s|$)")
_TITER = re.compile(r"1\s*:\s*(\d+)")
_NUMBER = re.compile(r"(?<![\d.:/-])(\d+(?:\.\d+)?)(?!\s*[-–]\s*\d)(?!\s*:)")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _windows(text: str, name: str):
    """Text right after every occurrence of the feature's label."""
    for pattern in LABELS[name]:
        for m in re.finditer(pattern, text):
            yield text[m.end(): m.end() + WINDOW]


def _binary(text: str, name: str):
    for window in _windows(text, name):
        neg, pos = _NEG.search(window), _POS.search(window)
        titer = _TITER.search(window)
        hits = [h for h in (neg, pos, titer) if h]
        if not hits:
            continue
        first = min(hits, key=lambda h: h.start())  # the result closest to the label
        if first is neg:
            return 0
        if first is pos:
            return 1
        return 1 if int(titer.group(1)) >= 80 else 0  # titer 1:80 or higher = positive
    return None


def _number(text: str, name: str, warnings: list):
    for window in _windows(text, name):
        m = _NUMBER.search(window)
        if not m:
            continue
        value = float(m.group(1))
        after = window[m.end(): m.end() + 12]

        # Unit conversion to the units the model was trained on.
        if name in ("C3", "C4") and re.match(r"\s*g\s*/\s*l", after):
            value = round(value * 100, 2)
            warnings.append(f"{name} converted from g/L to mg/dL ({value}).")
        elif name in ("C3", "C4") and value < 5 and not re.match(r"\s*mg", after):
            value = round(value * 100, 2)
            warnings.append(f"{name} looked like g/L and was converted to mg/dL ({value}). Please check.")
        elif name == "CRP" and re.match(r"\s*mg\s*/\s*dl", after):
            value = round(value * 10, 2)
            warnings.append(f"CRP converted from mg/dL to mg/L ({value}).")
        return value
    return None


def _age(text: str):
    for pattern in (r"\bage\s*[:\-]?\s*(\d{1,3})\b", r"\b(\d{1,3})\s*(?:years?|yrs?|y)\s*(?:old)?\b"):
        m = re.search(pattern, text)
        if m and 1 <= int(m.group(1)) <= 120:
            return float(m.group(1))
    return None


def _gender(text: str):
    m = re.search(r"\b(?:sex|gender)\s*[:\-]?\s*(female|male|f|m)\b", text)
    if m:
        return "Female" if m.group(1).startswith("f") else "Male"
    if re.search(r"\bfemale\b", text):
        return "Female"
    if re.search(r"\bmale\b", text):
        return "Male"
    return None


# ---------------------------------------------------------------- other lab results
# Results that are not among the 23 model inputs (e.g. hemoglobin, TSH, vitamin D).
# XGBoost cannot use them; they are passed to Qwen and the chat as extra context.

MAX_EXTRA = 40
_KNOWN = re.compile("|".join(p for pats in LABELS.values() for p in pats))
_SKIP_NAME = re.compile(
    r"^(test|tests|result|results|unit|units|reference|ref\.?|range|reference range|normal|flag|status|method|"
    r"patient.*|name|date.*|dob|age|sex|gender|doctor|physician|referred.*|page.*|report.*|lab.*|laboratory.*|"
    r"hospital|clinic|phone|tel|fax|address|id|mrn|specimen|sample|collected|received|reported|comments?|notes?|"
    r"interpretation|signature|printed.*|request.*|order.*)$"
)
_VALUE = re.compile(
    r"^(?P<val>[<>]?\s?\d+(?:\.\d+)?|positive|negative|reactive|non[\s-]?reactive|detected|not detected|borderline|equivocal)"
    r"\s*(?P<flag>\b[HL]\b|high|low)?$",
    re.IGNORECASE,
)
_UNIT = re.compile(r"^(?=.*[a-zA-Z%µ])[a-zA-Z%µ/^0-9.*·×]{1,14}(?:/[a-zA-Z0-9.]{1,6})?$")
_RANGE = re.compile(r"^[<>]?\s?\d+(?:\.\d+)?\s*(?:[-–]|to)\s*\d+(?:\.\d+)?|^(?:negative|positive|<\s?\d|>\s?\d)", re.IGNORECASE)
_SAME_LINE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9 ()/,'.+-]{1,48}?)\s*(?::|=|\t|\s{2,})\s*"
    r"(?P<val>[<>]?\s?\d+(?:\.\d+)?|positive|negative|reactive|non[\s-]?reactive|detected|not detected)"
    r"\s*(?P<unit>[a-zA-Z%µ][a-zA-Z0-9%µ/^.]{0,12})?\s*(?P<flag>\b[HL]\b)?$",
    re.IGNORECASE,
)


def _name_ok(name: str) -> bool:
    n = name.strip().rstrip(":").strip()
    low = n.lower()
    if not (2 <= len(n) <= 50) or sum(c.isalpha() for c in n) < 2:
        return False
    if _SKIP_NAME.match(low) or _KNOWN.search(low):
        return False
    return not _VALUE.match(n) and not _RANGE.match(n)


def _clean_value(v: str) -> str:
    v = re.sub(r"\s+", " ", v.strip())
    return v.capitalize() if v[0].isalpha() else v.replace(" ", "")


def extract_other_results(text: str) -> list[dict]:
    """Lab results in the report that are not one of the 23 model inputs.
    Understands "Name: value unit" on one line and table cells on separate lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: dict[str, dict] = {}
    i = 0
    while i < len(lines) and len(out) < MAX_EXTRA:
        line = lines[i]
        m = _SAME_LINE.match(line)
        if m and _name_ok(m.group("name")):
            name = m.group("name").strip()
            out.setdefault(name.lower(), {"name": name, "value": _clean_value(m.group("val")), "unit": (m.group("unit") or "").strip(), "flag": _flag(m.group("flag"))})
            i += 1
            continue
        if i + 1 < len(lines) and _name_ok(line):
            v = _VALUE.match(lines[i + 1])
            if v:
                unit, step = "", 2
                if i + 2 < len(lines) and _UNIT.match(lines[i + 2]) and not _RANGE.match(lines[i + 2]) and not _name_ok_strict(lines[i + 2]):
                    unit, step = lines[i + 2], 3
                if i + step < len(lines) and _RANGE.match(lines[i + step]):
                    step += 1
                out.setdefault(line.lower(), {"name": line.rstrip(":").strip(), "value": _clean_value(v.group("val")), "unit": unit, "flag": _flag(v.group("flag"))})
                i += step
                continue
        i += 1
    return list(out.values())


def _flag(f: str | None) -> str:
    f = (f or "").strip().lower()
    return "high" if f in ("h", "high") else "low" if f in ("l", "low") else ""


def _name_ok_strict(s: str) -> bool:
    """A unit line never looks like a lab name with spaces (e.g. 'Vitamin D')."""
    return " " in s and _name_ok(s)


def extract_lab_values(text: str) -> dict:
    """Returns {"values": {...found...}, "missing": [...], "warnings": [...], "extra": [...]}."""
    extra = extract_other_results(text)
    text = text.lower()
    warnings: list[str] = []
    values = {"Age": _age(text), "Gender": _gender(text)}
    for name in CONTINUOUS:
        if name != "Age":
            values[name] = _number(text, name, warnings)
    for name in BINARY:
        values[name] = _binary(text, name)

    found = {f: values[f] for f in FEATURES if values.get(f) is not None}
    missing = [f for f in FEATURES if values.get(f) is None]
    return {"values": found, "missing": missing, "warnings": warnings, "extra": extra}


def merge_reports(reports: list[dict]) -> dict:
    """Combine what was found in several PDFs of the same patient.
    The first report that has a value wins; a different value in a later report is
    reported as a warning, never averaged or guessed."""
    values: dict = {}
    sources: dict = {}
    warnings: list[str] = []
    extra: dict[str, dict] = {}
    for rep in reports:
        fname = rep["filename"]
        warnings += [f"{fname}: {w}" for w in rep["warnings"]]
        for k, v in rep["values"].items():
            if k not in values:
                values[k], sources[k] = v, fname
            elif values[k] != v:
                warnings.append(
                    f"{k} is different in {sources[k]} ({values[k]}) and {fname} ({v}). "
                    f"The value from {sources[k]} is used - please check it."
                )
        for e in rep.get("extra", []):
            key = e["name"].lower()
            if key not in extra and len(extra) < MAX_EXTRA:
                extra[key] = {**e, "source": fname}
    found = {f: values[f] for f in FEATURES if f in values}
    return {
        "filename": ", ".join(r["filename"] for r in reports),
        "files": [{"filename": r["filename"], "found": len(r["values"]), "extra": len(r.get("extra", []))} for r in reports],
        "values": found,
        "sources": {f: sources[f] for f in found},
        "missing": [f for f in FEATURES if f not in values],
        "warnings": warnings,
        "extra": list(extra.values()),
    }
