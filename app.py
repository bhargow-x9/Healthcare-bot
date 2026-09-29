"""MediClear — prescription explainer & health chatbot (Flask)."""
import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, url_for

from src import analyzer, chat, config, extract, llm, rag, voice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mediclear")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


class SessionStore:
    """In-memory sessions (analysis + sources + chat history). Oldest evicted first."""

    def __init__(self, limit: int):
        self._data: OrderedDict = OrderedDict()
        self._limit = limit
        self._lock = threading.Lock()

    def create(self, analysis=None, registry=None) -> tuple[str, dict]:
        sid = uuid.uuid4().hex
        session = {
            "analysis": analysis,
            "registry": registry or analyzer.SourceRegistry(),
            "history": [],
            "hindi": None,
        }
        with self._lock:
            self._data[sid] = session
            while len(self._data) > self._limit:
                self._data.popitem(last=False)
        return sid, session

    def get(self, sid):
        with self._lock:
            return self._data.get(sid)


sessions = SessionStore(config.MAX_SESSIONS)


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


jobs: dict = {}  # job_id -> {status, message, done, total, result | error}


def _run_analysis(job_id: str, filename: str, data: bytes | None, typed: str):
    job = jobs[job_id]

    def progress(message, done, total):
        job.update(message=message, done=done, total=total)
        log.info("[%s] %s (%s/%s)", job_id[:6], message, done, total)

    try:
        if data is not None:
            progress("Reading your document…", 0, 0)
            extraction = extract.extract(filename, data)
        else:
            extraction = extract.Extraction(typed, "typed text")
        if len(extraction.text.strip()) < 5:
            raise extract.ExtractionError("No readable text was found. Try a clearer photo or type the prescription.")
        result, registry = analyzer.analyze(extraction, filename, progress)
        sid, _ = sessions.create(result, registry)
        result["session_id"] = sid
        job.update(status="done", result=result)
    except (llm.LLMError, extract.ExtractionError) as exc:
        job.update(status="error", error=str(exc))
    except Exception as exc:
        log.exception("Analysis failed")
        job.update(status="error", error=f"Something went wrong: {exc}")


@app.post("/api/analyze")
def api_analyze():
    """Starts the analysis in the background; poll /api/analyze/<job_id> for progress."""
    upload = request.files.get("file")
    typed = (request.form.get("text") or "").strip()
    if upload and upload.filename:
        filename, data = upload.filename, upload.read()
    elif typed:
        filename, data = "typed text", None
    else:
        return _error("Upload a file or type the prescription text.", 400)

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "running", "message": "Starting…", "done": 0, "total": 0}
    while len(jobs) > config.MAX_SESSIONS:
        jobs.pop(next(iter(jobs)))
    threading.Thread(target=_run_analysis, args=(job_id, filename, data, typed), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/analyze/<job_id>")
def api_analyze_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return _error("This analysis was not found. Please start again.", 404)
    return jsonify(job)


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return _error("Message is empty.", 400)
    sid = body.get("session_id")
    session = sessions.get(sid)
    if session is None:
        sid, session = sessions.create()
    reply = chat.answer(session, message[:2000])
    return jsonify({"session_id": sid, **reply})


@app.post("/api/hindi")
def api_hindi():
    """Hindi version of the whole analysis (target=analysis) or of any given text."""
    body = request.get_json(silent=True) or {}
    session = sessions.get(body.get("session_id"))
    if body.get("target") == "analysis":
        if not session or not session["analysis"]:
            return _error("Analyse a prescription first.", 400)
        if not session["hindi"]:
            session["hindi"] = voice.to_hindi(analyzer.speech_script(session["analysis"]))
        hindi = session["hindi"]
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
    app.run(host="0.0.0.0", port=8080, debug=False)
