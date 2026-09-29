"""Build the knowledge-base index from the PDFs in data/ (run once, and again after adding books)."""
from src import config, rag

if __name__ == "__main__":
    print(f"Indexing PDFs in {config.DATA_DIR} into {config.VECTOR_STORE} …")
    count = rag.build_index()
    print(f"Done: {count} chunks indexed.")
