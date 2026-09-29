"""Trusted evidence sources that every explanation is grounded in.

- openFDA drug labels (official FDA prescribing information, linked on DailyMed)
- RxNorm (U.S. National Library of Medicine) to normalise medicine names
- MedlinePlus health topics (U.S. National Library of Medicine) for conditions/tests
- A built-in glossary of prescription abbreviations
The local knowledge base (PDF books) lives in rag.py.
"""
import difflib
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from functools import lru_cache

import requests

from . import config

log = logging.getLogger(__name__)

_http = requests.Session()
_http.headers["User-Agent"] = "MediClear-prescription-explainer/1.0"


@dataclass
class Source:
    kind: str  # prescription | fda_label | medlineplus | knowledge_base | glossary
    title: str
    publisher: str
    location: str  # section / page inside the source
    excerpt: str
    url: str | None = None
    id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _get(url: str, **params):
    try:
        resp = _http.get(url, params=params, timeout=config.HTTP_TIMEOUT)
        return resp if resp.ok else None
    except requests.RequestException as exc:
        log.warning("Source request failed: %s (%s)", url, exc)
        return None


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > limit * 0.5 else cut.rsplit(" ", 1)[0]) + " …"


# --- Medicine names -------------------------------------------------------------

# Indian/British names -> US names used by FDA labels and RxNorm.
US_NAMES = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "frusemide": "furosemide",
    "glibenclamide": "glyburide",
    "lignocaine": "lidocaine",
    "pethidine": "meperidine",
    "rifampicin": "rifampin",
    "amoxycillin": "amoxicillin",
    "cyclosporin": "cyclosporine",
    "levothyroxine sodium": "levothyroxine",
    "thyroxine": "levothyroxine",
    "clavulanic acid": "clavulanate",
}


def ingredient_names(generic: str | None) -> list[str]:
    """Split 'amoxicillin + clavulanic acid 625mg' into normalised ingredient names."""
    if not generic:
        return []
    names = []
    for part in re.split(r"\s*(?:\+|/|,|&|\band\b|\bwith\b)\s*", generic.lower()):
        part = re.sub(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%)?", "", part).strip(" .-()")
        if len(part) < 3:
            continue
        part = US_NAMES.get(part, part)
        part = rxnorm_normalize(part) or part
        if part not in names:
            names.append(part)
    return names[:4]


@lru_cache(maxsize=512)
def rxnorm_normalize(name: str) -> str | None:
    """Return the canonical RxNorm spelling of an ingredient, or None if not confident."""
    if not config.ENABLE_ONLINE_SOURCES:
        return None
    base = "https://rxnav.nlm.nih.gov/REST"
    resp = _get(f"{base}/rxcui.json", name=name, search=2)
    if resp and resp.json().get("idGroup", {}).get("rxnormId"):
        return name
    resp = _get(f"{base}/approximateTerm.json", term=name, maxEntries=1)
    candidates = (resp.json().get("approximateGroup", {}).get("candidate") or []) if resp else []
    if not candidates:
        return None
    props = _get(f"{base}/rxcui/{candidates[0]['rxcui']}/properties.json")
    found = (props.json().get("properties") or {}).get("name", "").lower() if props else ""
    # Only accept close spelling matches, never a different drug.
    return found if found and difflib.SequenceMatcher(None, name, found).ratio() >= 0.8 else None


# --- openFDA drug labels ------------------------------------------------------

FDA_SECTIONS = [
    (("boxed_warning",), "Boxed warning (most serious risks)"),
    (("indications_and_usage", "purpose"), "Uses"),
    (("contraindications", "do_not_use"), "Who should not take it"),
    (("warnings_and_cautions", "warnings", "ask_doctor"), "Warnings and precautions"),
    (("adverse_reactions", "stop_use"), "Side effects"),
    (("drug_interactions",), "Drug interactions"),
]


def fda_label_sources(ingredient: str) -> list[Source]:
    return [Source(**d) for d in _fda_label(ingredient)]


@lru_cache(maxsize=256)
def _fda_label(ingredient: str) -> tuple:
    if not config.ENABLE_ONLINE_SOURCES:
        return ()
    label = None
    for field_name in ("openfda.generic_name", "openfda.substance_name", "openfda.brand_name"):
        params = {"search": f'{field_name}:"{ingredient}"', "limit": 5}
        if config.OPENFDA_API_KEY:
            params["api_key"] = config.OPENFDA_API_KEY
        resp = _get("https://api.fda.gov/drug/label.json", **params)
        results = resp.json().get("results", []) if resp else []
        results = [r for r in results if r.get("indications_and_usage") or r.get("purpose")]
        if results:
            # Prefer single-ingredient labels (shortest generic name).
            label = min(results, key=lambda r: len(" ".join(r.get("openfda", {}).get("generic_name", ["x" * 99]))))
            break
    if not label:
        return ()

    openfda = label.get("openfda", {})
    product = (openfda.get("generic_name") or openfda.get("brand_name") or [ingredient])[0].title()
    set_id = label.get("set_id")
    url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if set_id else None
    date = label.get("effective_time", "")
    version = f" (label dated {date[:4]}-{date[4:6]}-{date[6:8]})" if len(date) == 8 else ""
    out = []
    for keys, section in FDA_SECTIONS:
        text = next((" ".join(label[k]) for k in keys if label.get(k)), "")
        if text:
            out.append(
                dict(
                    kind="fda_label",
                    title=f"FDA drug label: {product}",
                    publisher="U.S. Food & Drug Administration via openFDA / DailyMed" + version,
                    location=f"Section: {section}",
                    excerpt=_clip(text, 700),
                    url=url,
                )
            )
    return tuple(out)


# --- MedlinePlus -------------------------------------------------------------------


def medlineplus_sources(term: str) -> list[Source]:
    return [Source(**d) for d in _medlineplus(term.strip().lower()[:120])]


@lru_cache(maxsize=256)
def _medlineplus(term: str) -> tuple:
    if not config.ENABLE_ONLINE_SOURCES or not term:
        return ()
    resp = _get("https://wsearch.nlm.nih.gov/ws/query", db="healthTopics", term=term, retmax=1)
    if not resp:
        return ()
    try:
        doc = ET.fromstring(resp.content).find(".//document")
    except ET.ParseError:
        return ()
    if doc is None:
        return ()

    def content(name):
        node = doc.find(f"content[@name='{name}']")
        raw = "".join(node.itertext()) if node is not None else ""
        return re.sub(r"<[^>]+>", " ", html.unescape(raw))

    title = _clip(content("title"), 120)
    summary = content("FullSummary") or content("snippet")
    if not summary:
        return ()
    return (
        dict(
            kind="medlineplus",
            title=f"MedlinePlus: {title}",
            publisher="U.S. National Library of Medicine (MedlinePlus)",
            location="Health topic summary",
            excerpt=_clip(summary, 900),
            url=doc.get("url"),
        ),
    )


# --- Prescription abbreviation glossary --------------------------------------------

ABBREVIATIONS = {
    "OD": "once a day", "QD": "once a day", "BD": "twice a day", "BID": "twice a day",
    "TDS": "three times a day", "TID": "three times a day", "QID": "four times a day",
    "QDS": "four times a day", "HS": "at bedtime", "SOS": "only when needed",
    "PRN": "only when needed", "STAT": "immediately, one time", "AC": "before food",
    "PC": "after food", "BBF": "before breakfast", "ABF": "after breakfast",
    "Q4H": "every 4 hours", "Q6H": "every 6 hours", "Q8H": "every 8 hours", "Q12H": "every 12 hours",
    "TAB": "tablet", "CAP": "capsule", "SYP": "syrup", "INJ": "injection", "SUSP": "suspension",
    "OINT": "ointment", "GTT": "drops", "PO": "by mouth", "SL": "under the tongue",
}
_ABBR_RE = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\b\.?", re.IGNORECASE)
_PATTERN_RE = re.compile(r"(?<![\d/])([0-2]|½|1/2)\s*-\s*([0-2]|½|1/2)\s*-\s*([0-2]|½|1/2)(?![\d/])")


def glossary_source(text: str) -> Source | None:
    found = {}
    for match in _ABBR_RE.finditer(text):
        key = match.group(1).upper()
        found[key] = f"{key} = {ABBREVIATIONS[key]}"
    for m, a, n in _PATTERN_RE.findall(text):
        key = f"{m}-{a}-{n}"
        found[key] = f"{key} = {m} in the morning, {a} in the afternoon, {n} at night (0 means none)"
    if not found:
        return None
    return Source(
        kind="glossary",
        title="Prescription abbreviation glossary",
        publisher="Built into this app (standard medical abbreviations)",
        location="Abbreviations found in your prescription",
        excerpt="; ".join(found.values()),
    )
