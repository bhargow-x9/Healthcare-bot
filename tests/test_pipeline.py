"""Offline tests: Llama and online sources are replaced with fakes."""
import io
import sys
import time
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


def test_flask_endpoints(monkeypatch):
    import app as app_module

    client = app_module.app.test_client()
    def analyze(**data):
        res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
        assert res.status_code == 202
        for _ in range(100):
            job = client.get(f"/api/analyze/{res.get_json()['job_id']}").get_json()
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("analysis did not finish")

    job = analyze(text=RX)
    assert job["status"] == "done"
    body = job["result"]
    assert body["medicines"][0]["name"] == "metformin"

    res = client.post("/api/chat", json={"session_id": body["session_id"], "message": "What is it for?"})
    assert res.status_code == 200 and res.get_json()["sources"][0]["id"] == "S3"

    res = client.post("/api/hindi", json={"session_id": body["session_id"], "target": "analysis"})
    assert res.get_json()["hindi"] == "नमस्ते"

    assert client.post("/api/analyze", data={}).status_code == 400
    bad = analyze(file=(io.BytesIO(b"xx"), "a.exe"))
    assert bad["status"] == "error" and "Unsupported file type" in bad["error"]
