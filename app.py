"""MediClear — prescription explainer & health chatbot (Flask).

Stateless on purpose so it runs the same locally, on Render and on Vercel
(serverless): the analysis streams its progress and result in one request,
and the browser sends the analysis/sources/chat history back with each
follow-up request instead of the server keeping sessions in memory.
"""
import json
import logging
import os
import queue
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, url_for

from src import analyzer, chat, config, extract, llm, rag, voice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mediclear")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

HEARTBEAT_SECONDS = 5  # keeps the streamed response alive while the model is busy
MAX_HISTORY_MESSAGES = 12


@app.context_processor
def _asset_versions():
    """static_url('app.js') -> /static/app.js?v=<mtime>, so browsers never run a stale script."""

    def static_url(filename):
        path = Path(app.static_folder) / filename
        version = int(path.stat().st_mtime) if path.exists() else 0
        return url_for("static", filename=filename, v=version)

    return {"static_url": static_url}


def _error(message: str, status: int):
    return jsonify({"error": message}), status


@app.errorhandler(llm.LLMError)
def _llm_error(exc):
    return _error(str(exc), 503)


@app.errorhandler(extract.ExtractionError)
def _extract_error(exc):
    return _error(str(exc), 422)


@app.errorhandler(413)
def _too_large(_exc):
    return _error(f"File is too large (max {config.MAX_UPLOAD_MB} MB).", 413)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify({"llm": llm.health(), "knowledge_base": rag.status(), "online_sources": config.ENABLE_ONLINE_SOURCES})


def _analysis_worker(events: queue.Queue, filename: str, data: bytes | None, typed: str):
    def progress(message, done, total):
        events.put({"type": "progress", "message": message, "done": done, "total": total})
        log.info("%s (%s/%s)", message, done, total)

    try:
        if data is not None:
            progress("Reading your document…", 0, 0)
            extraction = extract.extract(filename, data)
        else:
            extraction = extract.Extraction(typed, "typed text")
        if len(extraction.text.strip()) < 5:
            raise extract.ExtractionError("No readable text was found. Try a clearer photo or type the prescription.")
        result, _ = analyzer.analyze(extraction, filename, progress)
        events.put({"type": "result", "result": result})
    except (llm.LLMError, extract.ExtractionError) as exc:
        events.put({"type": "error", "error": str(exc)})
    except Exception as exc:
        log.exception("Analysis failed")
        events.put({"type": "error", "error": f"Something went wrong: {exc}"})
    finally:
        events.put(None)


@app.post("/api/analyze")
def api_analyze():
    """Streams newline-delimited JSON events: progress…, then result or error."""
    upload = request.files.get("file")
    typed = (request.form.get("text") or "").strip()
    if upload and upload.filename:
        filename, data = upload.filename, upload.read()
    elif typed:
        filename, data = "typed text", None
    else:
        return _error("Upload a file or type the prescription text.", 400)

    events: queue.Queue = queue.Queue()
    threading.Thread(target=_analysis_worker, args=(events, filename, data, typed), daemon=True).start()

    def stream():
        while True:
            try:
                event = events.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                yield json.dumps({"type": "ping"}) + "\n"
                continue
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return Response(
        stream(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _client_session(body: dict) -> dict:
    """Rebuild the chat context the browser sent (analysis, numbered sources, history)."""
    history = [
        {"role": h["role"], "content": h["content"][:4000]}
        for h in (body.get("history") or [])[-MAX_HISTORY_MESSAGES:]
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)
    ]
    analysis = body.get("analysis")
    return {
        "registry": analyzer.SourceRegistry.from_list(body.get("sources") or []),
        "history": history,
        "analysis": analysis if isinstance(analysis, dict) else None,
    }


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return _error("Message is empty.", 400)
    session = _client_session(body)
    reply = chat.answer(session, message[:2000])
    # All sources, including any found for this question, so the browser keeps the numbering.
    return jsonify({**reply, "all_sources": session["registry"].to_list()})


@app.post("/api/hindi")
def api_hindi():
    """Hindi version of the whole analysis (target=analysis) or of any given text."""
    body = request.get_json(silent=True) or {}
    if body.get("target") == "analysis":
        analysis = body.get("analysis")
        if not isinstance(analysis, dict):
            return _error("Analyse a prescription first.", 400)
        hindi = voice.to_hindi(analyzer.speech_script(analysis))
    else:
        text = (body.get("text") or "").strip()
        if not text:
            return _error("Nothing to translate.", 400)
        hindi = voice.to_hindi(text[:4000])
    if not hindi:
        return _error("Hindi translation failed. Please try again.", 502)
    return jsonify({"hindi": hindi})


@app.post("/api/tts")
def api_tts():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return _error("Nothing to speak.", 400)
    try:
        audio = voice.hindi_speech(text)
    except Exception as exc:  # offline, gTTS blocked, ...
        log.warning("Server TTS failed, client will use browser voice: %s", exc)
        return _error("Server voice unavailable", 502)
    return Response(audio, mimetype="audio/mpeg")


if __name__ == "__main__":
    rag.warm_up()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False, threaded=True)
