"""Prescription analysis pipeline.

1. Llama (medical model) turns the raw text into structured data.
2. Evidence is gathered per medicine / condition / test (openFDA, MedlinePlus,
   local knowledge base, abbreviation glossary) and numbered S1, S2, ...
3. Llama explains each item in plain English using ONLY its evidence and cites IDs.
4. Citations are validated: unknown IDs are dropped, uncited claims are flagged.
5. A short overview + warnings + questions for the doctor are written.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, llm, prompts, rag, sources
from .extract import Extraction
from .prompts import NOT_FOUND
from .sources import Source

DISCLAIMER = (
    "This explanation is for understanding only and is not medical advice. "
    "Always follow your doctor's instructions and check with your doctor or pharmacist before making any change."
)
RX_CITABLE_FIELDS = {"how_to_take", "what_it_is_for"}
EVIDENCE_CHARS = 500  # per source in each item prompt; keeps calls small for rate-limited APIs
_ID_RE = re.compile(r"S\s*(\d+)", re.IGNORECASE)
_INLINE_CITE_RE = re.compile(r"\s*[\[(]\s*S\d+(?:\s*[,;/ ]\s*S?\d+)*\s*[\])]", re.IGNORECASE)


class SourceRegistry:
    """Numbers every piece of evidence (S1, S2, ...) and de-duplicates it."""

    def __init__(self):
        self._items: list[Source] = []
        self._keys: dict = {}

    def add(self, src: Source) -> str:
        key = (src.title, src.location, src.excerpt[:120])
        if key not in self._keys:
            src.id = f"S{len(self._items) + 1}"
            self._items.append(src)
            self._keys[key] = src.id
        return self._keys[key]

    def get(self, sid: str):
        idx = int(sid[1:]) - 1 if re.fullmatch(r"S\d+", sid or "") else -1
        return self._items[idx] if 0 <= idx < len(self._items) else None

    def ids(self) -> list[str]:
        return [s.id for s in self._items]

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self._items]

    def evidence_block(self, ids, max_chars=900, overrides=None) -> str:
        """Evidence text for the prompt; `overrides` replaces a source's excerpt (e.g. just one Rx line)."""
        blocks = []
        for sid in ids:
            s = self.get(sid)
            if s:
                excerpt = (overrides or {}).get(sid, s.excerpt)
                excerpt = excerpt if len(excerpt) <= max_chars else excerpt[:max_chars] + " …"
                blocks.append(f"[{s.id}] {s.title} | {s.location}\n{excerpt}")
        return "\n\n".join(blocks) or "(no evidence found)"


def cited_ids(text: str, allowed) -> list[str]:
    """IDs cited inline in `text` like [S3] or [S3, S5] that exist in `allowed`."""
    out = []
    for group in re.findall(r"[\[(]([^\])]*S\s*\d+[^\])]*)[\])]", text or "", flags=re.IGNORECASE):
        for n in re.findall(r"\d+", group):
            sid = f"S{n}"
            if sid in allowed and sid not in out:
                out.append(sid)
    return out


def clean_field(value, allowed) -> dict:
    """Normalise a model field to {text, sources, verified}, keeping only real source IDs."""
    if isinstance(value, dict):
        text, srcs = value.get("text"), value.get("sources") or []
    else:
        text, srcs = value, []
    if isinstance(text, list):
        text = "; ".join(str(t) for t in text if t)
    text = str(text or "").strip()
    if isinstance(srcs, (str, int)):
        srcs = [srcs]

    ids = []
    for s in list(srcs) + cited_ids(text, allowed):
        m = _ID_RE.search(str(s)) or re.fullmatch(r"\s*(\d+)\s*", str(s))
        sid = f"S{m.group(1)}" if m else None
        if sid in allowed and sid not in ids:
            ids.append(sid)
    text = _INLINE_CITE_RE.sub("", text).strip()
    if not text or text.lower().startswith("not found in our sources"):
        return {"text": NOT_FOUND, "sources": [], "verified": False}
    return {"text": text, "sources": ids, "verified": bool(ids)}


def _items(structured: dict, key: str) -> list[dict]:
    return [i for i in (structured.get(key) or []) if isinstance(i, dict)][: config.MAX_ITEMS]


def _item_name(item: dict) -> str:
    return item.get("name") or item.get("generic_name") or item.get("brand_name") or item.get("as_written") or ""


# --- evidence gathering (network / IO, runs in parallel) --------------------------


def _medicine_evidence(med: dict) -> list[Source]:
    evidence = []
    for ingredient in sources.ingredient_names(med.get("generic_name")):
        evidence += sources.fda_label_sources(ingredient)
    evidence += rag.search(f"{_item_name(med)} uses side effects precautions")
    return evidence


def _condition_evidence(cond: dict) -> list[Source]:
    name = _item_name(cond)
    return sources.medlineplus_sources(name) + rag.search(f"{name} causes symptoms treatment")


def _test_evidence(test: dict) -> list[Source]:
    name = _item_name(test)
    return sources.medlineplus_sources(name) + rag.search(f"{name} test purpose preparation")


# --- LLM steps ------------------------------------------------------------------------


def structure(text: str) -> dict:
    try:
        data = llm.complete_json(prompts.STRUCTURE_SYSTEM, prompts.structure_user(text), model=config.MEDICAL_MODEL)
    except ValueError:
        data = {}
    for key in ("diagnoses", "complaints", "medicines", "tests", "advice", "unclear_items"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def _explain(system: str, label: str, fields: dict, item: dict, ids: list, reg: SourceRegistry) -> dict:
    # Only this item's own line of the prescription is sent (not the whole document) to save tokens.
    rx_line = {ids[0]: f"(line from the prescription) {item.get('as_written') or _item_name(item)}"}
    evidence = reg.evidence_block(ids, max_chars=EVIDENCE_CHARS, overrides=rx_line)
    user = prompts.item_user(label, item, evidence, fields)
    try:
        raw = llm.complete_json(system, user, model=config.MEDICAL_MODEL)
    except ValueError:
        raw = {}
    allowed = set(ids)
    # The prescription (S1) can say how to take a medicine and why it was given,
    # but it is never evidence for side effects or for what a condition/test is.
    general = allowed - {ids[0]}
    out = {"plain_name": str(raw.get("plain_name") or "").strip(), "evidence": ids}
    out.update({f: clean_field(raw.get(f), allowed if f in RX_CITABLE_FIELDS else general) for f in fields})
    return out


def _summarize(structured: dict, meds: list, conds: list, tests: list, reg: SourceRegistry) -> dict:
    compact = {
        "document_type": structured.get("document_type"),
        "diagnoses": [c["name"] for c in conds],
        "medicines": [
            {
                "name": m["name"],
                "kind": m["plain_name"],
                "what_it_is_for": m["what_it_is_for"]["text"],
                "how_to_take": m["how_to_take"]["text"],
                "cautions": {"text": m["important_cautions"]["text"], "sources": m["important_cautions"]["sources"]},
            }
            for m in meds
        ],
        "tests": [t["name"] for t in tests],
        "advice": structured.get("advice"),
        "follow_up": structured.get("follow_up"),
        "unclear_items": structured.get("unclear_items"),
    }
    try:
        raw = llm.complete_json(prompts.SUMMARY_SYSTEM, json.dumps(compact, ensure_ascii=False), model=config.LLM_MODEL)
    except ValueError:
        raw = {}
    allowed = set(reg.ids())
    warnings = [clean_field(w, allowed) for w in (raw.get("key_warnings") or [])[:4]]
    return {
        "summary": str(raw.get("summary") or "").strip(),
        "warnings": [w for w in warnings if w["verified"]],
        "questions": [str(q) for q in (raw.get("questions_for_doctor") or []) if q][:6],
    }


# --- public entry point -----------------------------------------------------------------


def analyze(extraction: Extraction, filename: str = "typed text", progress=None) -> tuple[dict, SourceRegistry]:
    """`progress(message, done, total)` is called as each step starts."""
    report = progress or (lambda *_: None)
    text = extraction.text.strip()
    report("Finding medicines, diagnoses and tests…", 0, 0)
    structured = structure(text)

    reg = SourceRegistry()
    rx_id = reg.add(
        Source(
            kind="prescription",
            title="Your uploaded prescription",
            publisher="Your document",
            location=f"{filename} (read via {extraction.method})",
            excerpt=text[:1500],
        )
    )
    glossary = sources.glossary_source(text)
    base_ids = [rx_id] + ([reg.add(glossary)] if glossary else [])

    meds, conds, tests = (_items(structured, k) for k in ("medicines", "diagnoses", "tests"))
    total = len(meds) + len(conds) + len(tests) + 1  # +1 for the summary
    report(f"Found {len(meds)} medicines, {len(conds)} diagnoses, {len(tests)} tests. Checking FDA & MedlinePlus…", 0, total)
    with ThreadPoolExecutor(max_workers=8) as pool:
        med_ev = list(pool.map(_medicine_evidence, meds))
        cond_ev = list(pool.map(_condition_evidence, conds))
        test_ev = list(pool.map(_test_evidence, tests))

    jobs = []  # (kind, item, evidence ids) — registered in order so numbering is stable
    jobs += [("medicine", m, base_ids + [reg.add(s) for s in ev]) for m, ev in zip(meds, med_ev)]
    jobs += [("condition", c, [rx_id] + [reg.add(s) for s in ev]) for c, ev in zip(conds, cond_ev)]
    jobs += [("test", t, [rx_id] + [reg.add(s) for s in ev]) for t, ev in zip(tests, test_ev)]

    specs = {
        "medicine": (prompts.MEDICINE_SYSTEM, "MEDICINE", prompts.MEDICINE_FIELDS),
        "condition": (prompts.CONDITION_SYSTEM, "CONDITION", prompts.CONDITION_FIELDS),
        "test": (prompts.TEST_SYSTEM, "TEST", prompts.TEST_FIELDS),
    }

    finished = [0]
    lock = threading.Lock()

    def run(job):
        kind, item, ids = job
        system, label, fields = specs[kind]
        with lock:
            report(f"Explaining {kind}: {_item_name(item)}", finished[0], total)
        result = _explain(system, label, fields, item, ids, reg)
        result.update(kind=kind, name=_item_name(item), details=item)
        with lock:
            finished[0] += 1
        return result

    with ThreadPoolExecutor(max_workers=max(1, config.LLM_PARALLELISM)) as pool:
        explained = list(pool.map(run, jobs))
    med_out = [e for e in explained if e["kind"] == "medicine"]
    cond_out = [e for e in explained if e["kind"] == "condition"]
    test_out = [e for e in explained if e["kind"] == "test"]

    report("Writing the overview…", total - 1, total)
    overview = _summarize(structured, med_out, cond_out, test_out, reg)
    warnings = list(extraction.warnings)
    if structured.get("is_medical_document") is False:
        warnings.insert(0, "This does not look like a prescription or medical document. Results may be meaningless.")
    if not (meds or conds or tests):
        warnings.append("No medicines, diagnoses or tests were found in this document.")

    result = {
        "document": {
            "type": structured.get("document_type"),
            "date": structured.get("date"),
            "doctor": structured.get("doctor"),
            "patient": structured.get("patient"),
        },
        "extraction": {"method": extraction.method, "pages": extraction.pages, "text": text, "warnings": warnings},
        **overview,
        "medicines": med_out,
        "conditions": cond_out,
        "tests": test_out,
        "advice": [str(a) for a in structured.get("advice") or [] if a],
        "follow_up": structured.get("follow_up"),
        "unclear": [str(u) for u in structured.get("unclear_items") or [] if u],
        "sources": reg.to_list(),
        "disclaimer": DISCLAIMER,
    }
    return result, reg


def speech_script(analysis: dict) -> str:
    """English script (no citations) that is translated to Hindi for the voice read-out."""
    parts = ["Here is a simple explanation of your prescription.", analysis.get("summary", "")]
    for i, med in enumerate(analysis.get("medicines", []), 1):
        line = f"Medicine {i}: {med['details'].get('as_written') or med['name']}."
        if med.get("plain_name"):
            line += f" It is a {med['plain_name']}."
        for key in ("what_it_is_for", "how_to_take"):
            if med[key]["verified"]:
                line += " " + med[key]["text"]
        parts.append(line)
    if analysis.get("warnings"):
        parts.append("Important: " + " ".join(w["text"] for w in analysis["warnings"]))
    parts.append("Please follow your doctor's instructions, and ask your doctor or pharmacist if you have any doubt.")
    return "\n".join(p for p in parts if p)
