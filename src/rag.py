"""Local medical knowledge base (RAG), following the reference repo:
PDF books in data/ -> chunks -> MiniLM embeddings -> FAISS (local) or Pinecone.
"""
import logging
import threading
from pathlib import Path

from . import config
from .sources import Source

log = logging.getLogger(__name__)

_store = None
_store_error = None
_lock = threading.Lock()


def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def load_and_split(data_dir: Path = config.DATA_DIR):
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_index(data_dir: Path = config.DATA_DIR) -> int:
    """Embed every PDF in data_dir into the configured vector store. Returns chunk count."""
    chunks = load_and_split(data_dir)
    if not chunks:
        raise RuntimeError(f"No PDF text found in {data_dir}")
    embeddings = _embeddings()
    if config.VECTOR_STORE == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        if not pc.has_index(config.PINECONE_INDEX_NAME):
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=384,  # all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        PineconeVectorStore.from_documents(chunks, embeddings, index_name=config.PINECONE_INDEX_NAME)
    else:
        from langchain_community.vectorstores import FAISS

        FAISS.from_documents(chunks, embeddings).save_local(str(config.INDEX_DIR))
    return len(chunks)


def _get_store():
    global _store, _store_error
    if _store is not None or _store_error is not None:
        return _store
    with _lock:
        if _store is not None or _store_error is not None:
            return _store
        try:
            if config.VECTOR_STORE == "pinecone":
                if not config.PINECONE_API_KEY:
                    raise RuntimeError("PINECONE_API_KEY not set")
                from langchain_pinecone import PineconeVectorStore

                _store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=_embeddings())
            else:
                if not (config.INDEX_DIR / "index.faiss").exists():
                    raise RuntimeError("No knowledge-base index yet. Add PDFs to data/ and run: python store_index.py")
                from langchain_community.vectorstores import FAISS

                _store = FAISS.load_local(str(config.INDEX_DIR), _embeddings(), allow_dangerous_deserialization=True)
        except Exception as exc:  # missing deps, no index, bad key...
            _store_error = str(exc)
            log.warning("Knowledge base disabled: %s", exc)
    return _store


def search(query: str, k: int = config.RAG_TOP_K) -> list[Source]:
    store = _get_store()
    if store is None or not query.strip():
        return []
    with _lock:
        try:
            docs = store.similarity_search(query, k=k)
        except Exception as exc:
            log.warning("Knowledge-base search failed: %s", exc)
            return []
    results = []
    for doc in docs:
        meta = doc.metadata or {}
        page = meta.get("page")
        location = f"{Path(meta.get('source', 'document')).name}" + (f", page {int(page) + 1}" if page is not None else "")
        results.append(
            Source(
                kind="knowledge_base",
                title=Path(meta.get("source", "Medical reference")).stem.replace("_", " "),
                publisher="Local medical reference library (data/ folder)",
                location=location,
                excerpt=" ".join(doc.page_content.split()),
            )
        )
    return results


def warm_up():
    """Load embeddings/index in the background so the first request is fast."""
    threading.Thread(target=_get_store, daemon=True).start()


def status() -> dict:
    if config.VECTOR_STORE == "pinecone":
        ready = bool(config.PINECONE_API_KEY)
    else:
        ready = (config.INDEX_DIR / "index.faiss").exists()
    ready = ready and _store_error is None
    return {
        "backend": config.VECTOR_STORE,
        "ready": ready,
        "message": _store_error or ("ready" if ready else "no index - run python store_index.py"),
    }
