"""Offline tests: Llama and online sources are replaced with fakes."""
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analyzer, chat, llm, prompts, rag, sources  # noqa: E402
from src.sources import Source  # noqa: E402

RX = "Dx: Type 2 diabetes\nTab Metformin 500 mg 1-0-1 after food x 30 days BD\nTest: HbA1c"


def fake_complete_json(system, user, **kwargs):
    if system == prompts.STRUCTURE_SYSTEM:
        return {
            "is_medical_document": True,
            "document_type": "prescription",
            "diagnoses": [{"as_written": "Type 2 diabetes", "name": "Type 2 diabetes"}],
            "medicines": [{"as_written": "Tab Metformin 500 mg 1-0-1", "generic_name": "metformin", "confidence": "high"}],
            "tests": [{"as_written": "HbA1c", "name": "HbA1c"}],
            "advice": [],
            "unclear_items": [],
        }
    if system == prompts.MEDICINE_SYSTEM:
        return {
            "plain_name": "Diabetes medicine",
            "what_it_is_for": {"text": "Lowers blood sugar [S3].", "sources": ["S3"]},
            "how_to_take": {"text": "One tablet morning and night after food.", "sources": ["S1", "S2"]},
            "common_side_effects": {"text": "Upset stomach.", "sources": ["S99"]},  # fake ID -> dropped
            "important_cautions": {"text": prompts.NOT_FOUND, "sources": []},
        }
    if system == prompts.SUMMARY_SYSTEM:
        return {
            "summary": "You have diabetes medicine.",
            "key_warnings": [{"text": "Unsourced warning", "sources": []}],
            "questions_for_doctor": ["How long do I take it?"],
        }
    if system == prompts.HINDI_SYSTEM:
        return {"hindi": "नमस्ते"}
    return {"plain_name": "x", "what_it_is": {"text": "A condition.", "sources": ["S1"]}}


def fda(name):
    return [Source("fda_label", f"FDA drug label: {name}", "FDA", "Section: Uses", "Treats type 2 diabetes.", "https://x")]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", fake_complete_json)
    monkeypatch.setattr(llm, "complete", lambda messages, **kw: "Metformin lowers sugar [S3]. Ask your doctor [S42].")
    monkeypatch.setattr(sources, "fda_label_sources", fda)
    monkeypatch.setattr(sources, "rxnorm_normalize", lambda name: name)
    monkeypatch.setattr(sources, "medlineplus_sources", lambda term: [])
    monkeypatch.setattr(rag, "search", lambda query, k=3: [])


def run():
    from src.extract import Extraction

    return analyzer.analyze(Extraction(RX, "typed text"))


def test_sources_are_numbered_and_validated():
    result, reg = run()
    ids = [s["id"] for s in result["sources"]]
    assert ids[:3] == ["S1", "S2", "S3"]
    assert result["sources"][0]["kind"] == "prescription"
    assert result["sources"][1]["kind"] == "glossary"
    med = result["medicines"][0]
    assert med["what_it_is_for"] == {"text": "Lowers blood sugar.", "sources": ["S3"], "verified": True}
    assert med["how_to_take"]["sources"] == ["S1", "S2"]
    # A source ID the model invented is dropped and the claim is flagged as unverified.
    assert med["common_side_effects"]["sources"] == [] and not med["common_side_effects"]["verified"]
    assert med["important_cautions"]["text"] == prompts.NOT_FOUND


def test_unsourced_warnings_are_removed():
    result, _ = run()
    assert result["warnings"] == []
    assert result["questions"] == ["How long do I take it?"]


def test_glossary_decodes_abbreviations_but_not_dates():
    src = sources.glossary_source("Tab X 1-0-1 BD, date 12-05-2024")
    assert "BD = twice a day" in src.excerpt
    assert "1-0-1 = 1 in the morning" in src.excerpt
    assert "12-05" not in src.excerpt


def test_ingredient_names_map_indian_names():
    assert sources.ingredient_names("Paracetamol 650mg + Caffeine") == ["acetaminophen", "caffeine"]


def test_chat_cites_only_known_sources():
    _, reg = run()
    session = {"registry": reg, "history": [], "analysis": None}
    reply = chat.answer(session, "What is metformin for?")
    assert [s["id"] for s in reply["sources"]] == ["S3"]
    assert not reply["emergency"]
    assert chat.answer(session, "I have chest pain")["emergency"]


def test_groq_switches_model_when_daily_quota_is_used_up(monkeypatch):
    import importlib

    real_llm = importlib.reload(llm)  # undo the autouse fakes for this module
    monkeypatch.setattr(real_llm.config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(real_llm.config, "GROQ_API_KEY", "test")
    monkeypatch.setattr(real_llm.config, "GROQ_FALLBACK_MODELS", ["small-model"])
    calls = []

    class Resp:
        def __init__(self, status, payload, text=""):
            self.status_code, self._payload, self.text, self.headers = status, payload, text, {}
            self.ok = status < 400

        def json(self):
            return self._payload

    def fake_post(url, json, **kwargs):
        calls.append(json["model"])
        if json["model"] == "big-model":
            return Resp(429, {}, "tokens per day (TPD) ... Please try again in 16m15.8s.")
        return Resp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(real_llm.requests, "post", fake_post)
    assert real_llm.complete([{"role": "user", "content": "hi"}], model="big-model") == "ok"
    assert real_llm.complete([{"role": "user", "content": "hi"}], model="big-model") == "ok"
    assert calls == ["big-model", "small-model", "small-model"]  # exhausted model is skipped next time


def test_registry_round_trips_through_the_browser():
    _, reg = run()
    rebuilt = analyzer.SourceRegistry.from_list(list(reversed(reg.to_list())))
    assert rebuilt.to_list() == reg.to_list()


def test_flask_endpoints_are_stateless():
    import app as app_module

    client = app_module.app.test_client()

    def analyze(**data):
        res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
        assert res.status_code == 200 and res.mimetype == "application/x-ndjson"
        events = [json.loads(line) for line in res.get_data(as_text=True).splitlines() if line.strip()]
        assert events[-1]["type"] in ("result", "error")
        return events

    events = analyze(text=RX)
    assert any(e["type"] == "progress" for e in events)
    body = events[-1]["result"]
    assert body["medicines"][0]["name"] == "metformin"

    # Chat and Hindi get everything from the request (no server session), as on Vercel.
    context = {"analysis": body, "sources": body["sources"], "history": []}
    res = client.post("/api/chat", json={**context, "message": "What is it for?"}).get_json()
    assert res["sources"][0]["id"] == "S3"
    assert [s["id"] for s in res["all_sources"]] == [s["id"] for s in body["sources"]]

    res = client.post("/api/hindi", json={"target": "analysis", "analysis": body})
    assert res.get_json()["hindi"] == "नमस्ते"
    # Malformed analysis from a client must not crash the server.
    assert client.post("/api/hindi", json={"target": "analysis", "analysis": {"medicines": [1, {"x": 2}]}}).status_code == 200
    assert client.post("/api/chat", json={"message": "hi", "analysis": {"medicines": "bad"}, "sources": ["x"]}).status_code == 200

    assert client.post("/api/analyze", data={}).status_code == 400
    bad = analyze(file=(io.BytesIO(b"xx"), "a.exe"))
    assert bad[-1]["type"] == "error" and "Unsupported file type" in bad[-1]["error"]
