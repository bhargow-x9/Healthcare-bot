"""Hindi translation (Llama) and Hindi text-to-speech (gTTS)."""
import io
import re

from . import config, llm, prompts

MAX_TTS_CHARS = 6000


def to_hindi(english: str) -> str:
    english = re.sub(r"\s*\[S\d+(?:\s*,\s*S\d+)*\]", "", english).strip()
    try:
        data = llm.complete_json(prompts.HINDI_SYSTEM, english, model=config.LLM_MODEL, temperature=0.2)
        return str(data.get("hindi") or "").strip()
    except ValueError:
        return ""


def hindi_speech(text: str) -> bytes:
    """MP3 bytes of `text` spoken in Hindi. Needs internet (Google TTS)."""
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text[:MAX_TTS_CHARS], lang="hi").write_to_fp(buf)
    return buf.getvalue()
