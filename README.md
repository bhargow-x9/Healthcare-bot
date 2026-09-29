# MediClear: Prescription Explainer & Health Chatbot (Llama)

Upload a prescription as a **photo, scanned or digital PDF, Word file or typed text**. MediClear reads it,
works out every medicine, diagnosis and test, and explains them in **plain English**. Every statement
has a **numbered source**. You can ask follow-up questions in chat and **hear the explanation in Hindi**.

Built on the same stack as the reference repo
[entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS](https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS)
(Flask + LangChain + MiniLM embeddings + Pinecone/FAISS). The OpenAI model is replaced with **Llama**,
and the app adds prescription reading, grounded citations and Hindi voice.

## How it works

```
upload ──► read text ─────────────► Llama (medical model) ──► structured prescription
           • PDF text (pypdf)                                   medicines / diagnoses / tests
           • scans & photos → Llama 3.2 Vision (Tesseract fallback)
                                                  │
              evidence per item, numbered S1, S2, … ◄┘
              • S1 = your prescription itself
              • abbreviation glossary (BD, TDS, 1-0-1, SOS …)
              • FDA drug label sections (openFDA → DailyMed link)
              • MedlinePlus health topics (U.S. National Library of Medicine)
              • your reference books (RAG: data/*.pdf → FAISS/Pinecone, file + page)
                                                  │
              Llama explains each item using ONLY its evidence and cites IDs
                                                  │
              validation: made-up IDs are dropped, uncited claims are flagged ⚠,
              warnings without a source are removed
                                                  │
              dashboard (English) ── chat (cited) ── 🔊 Hindi voice (Llama translation + gTTS)
```

**Safety rules built into the prompts:** the app never changes a dose, and dosing comes only from your
prescription (US label dosing is ignored). If there is no evidence, it says "Not found in our sources,
ask your doctor or pharmacist". It warns about hard-to-read words and detects emergency questions
(112 / 108).

## Setup

### 1. Install Python packages
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

### 2. Choose your Llama backend

**Option A: local and private (Ollama, default).** Install from https://ollama.com, then:
```bash
ollama pull llama3.1:8b        # chat, explanations, Hindi
ollama pull llama3.2-vision    # reads photos / scanned prescriptions
```
To use a **specific medical model**, pull any Llama-based medical fine-tune into Ollama and set
`MEDICAL_MODEL=<its name>` in `.env`. It will handle prescription extraction and explanations,
while `LLM_MODEL` handles chat and Hindi. Pick a model that follows JSON instructions well.

**Option B: hosted Llama (Groq, fast, no GPU needed).** In `.env` set `LLM_PROVIDER=groq`,
`GROQ_API_KEY=...`, `LLM_MODEL=llama-3.3-70b-versatile` and
`VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`.

### 3. (Optional) Add medical reference books
Put PDFs in `data/` (as in the reference repo), then run:
```bash
python store_index.py
```
Use `VECTOR_STORE=pinecone` + `PINECONE_API_KEY` to use Pinecone like the reference repo
(uncomment the Pinecone lines in `requirements.txt`). Without books, the app still cites the FDA,
MedlinePlus and the prescription itself.

### 4. Run
```bash
python app.py
```
Open http://localhost:8080. The pills at the top show whether the model, knowledge base and online
sources are ready.

### Tests
```bash
python -m pytest -q
```
Tests use a fake Llama and fake sources, so they run offline.

## Project layout
| Path | What it does |
|---|---|
| `app.py` | Flask routes: `/api/analyze`, `/api/chat`, `/api/hindi`, `/api/tts`, `/api/status` |
| `src/extract.py` | Text from images (Llama vision / Tesseract), PDFs (text + OCR for scanned pages), DOCX, TXT |
| `src/analyzer.py` | The pipeline: structure → evidence → grounded explanations → citation validation → summary |
| `src/sources.py` | openFDA labels, RxNorm name matching (paracetamol→acetaminophen etc.), MedlinePlus, glossary |
| `src/rag.py` | Knowledge base: PDF → chunks → MiniLM embeddings → FAISS / Pinecone |
| `src/chat.py` | Follow-up chat grounded in the prescription + sources, emergency detection |
| `src/voice.py` | Hindi translation (Llama) and Hindi speech (gTTS; the browser voice is the offline fallback) |
| `src/llm.py` | Ollama / Groq client (text, JSON mode, images) |
| `src/prompts.py` | All prompts |
| `templates/`, `static/` | Dashboard UI (upload, cited result cards, sources panel, chat, mic, Hindi voice) |

## Notes and limits
- **Educational tool, not medical advice.** Always confirm with a doctor or pharmacist.
- FDA labels are for US products. Indian brand names (e.g. "Dolo 650") are matched through the generic
  name that Llama reads from the prescription. Check the "contains …" line on each medicine card.
- Handwriting recognition depends on the vision model and photo quality. Compare the "Text we read"
  panel with your prescription.
- Hindi speech uses Google TTS (needs internet). Without internet, the browser's Hindi voice is used.
  Voice questions use the browser's speech recognition (Chrome/Edge).
- Sessions are kept in memory. Add a database and authentication before handling real patient data at scale.
