"""Follow-up chat grounded in the analysed prescription + knowledge base."""
import json
import re

from . import config, llm, prompts, rag, sources
from .analyzer import cited_ids

MAX_HISTORY_TURNS = 6
MAX_CONTEXT_CHARS = 14000

_EMERGENCY_RE = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|trouble breathing|short(ness)? of breath|unconscious|fainted|"
    r"overdose|took too many|seizure|fits|severe bleeding|suicid|kill myself|self[- ]harm|stroke|"
    r"सीने में दर्द|सांस नहीं|साँस नहीं|बेहोश|ज़्यादा दवा|आत्महत्या",
    re.IGNORECASE,
)
EMERGENCY_NOTE = (
    "If this is an emergency, get medical help now: call 112 or 108 (ambulance, India), "
    "or go to the nearest hospital."
)


def _analysis_context(analysis: dict) -> str:
    if not analysis:
        return "(No prescription uploaded yet.)"

    def items(key):  # the analysis comes back from the browser, so read it defensively
        return [i for i in analysis.get(key) or [] if isinstance(i, dict)]

    compact = {
        "summary": analysis.get("summary"),
        "medicines": [
            {
                "as_written": (m.get("details") or {}).get("as_written") if isinstance(m.get("details"), dict) else None,
                "name": m.get("name"),
                **{k: f"{v.get('text')} {v.get('sources')}" for k, v in m.items() if isinstance(v, dict) and "text" in v},
            }
            for m in items("medicines")
        ],
        "conditions": [c.get("name") for c in items("conditions")],
        "tests": [t.get("name") for t in items("tests")],
        "advice": analysis.get("advice"),
    }
    return "PATIENT'S PRESCRIPTION (already explained, with source IDs):\n" + json.dumps(
        compact, ensure_ascii=False, indent=1
    )


def answer(session: dict, question: str) -> dict:
    reg = session["registry"]
    fresh = rag.search(question)
    if config.ENABLE_ONLINE_SOURCES:
        fresh += sources.medlineplus_sources(question)
    fresh_ids = [reg.add(s) for s in fresh]

    # Newest evidence first, then the prescription's own sources, within budget.
    ordered = fresh_ids + [sid for sid in reg.ids() if sid not in fresh_ids]
    evidence, used = [], 0
    for sid in ordered:
        block = reg.evidence_block([sid], max_chars=600)
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        evidence.append(block)
        used += len(block)

    context = _analysis_context(session.get("analysis")) + "\n\nSOURCES:\n" + "\n\n".join(evidence)
    history = session["history"][-MAX_HISTORY_TURNS * 2:]
    messages = (
        [{"role": "system", "content": prompts.CHAT_SYSTEM}]
        + history
        + [{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]
    )
    reply = llm.complete(messages, model=config.LLM_MODEL, temperature=0.2).strip()

    session["history"] += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    cited = cited_ids(reply, set(reg.ids()))
    emergency = bool(_EMERGENCY_RE.search(question))
    return {
        "answer": reply,
        "sources": [reg.get(sid).to_dict() for sid in cited],
        "emergency": emergency,
        "emergency_note": EMERGENCY_NOTE if emergency else None,
    }
