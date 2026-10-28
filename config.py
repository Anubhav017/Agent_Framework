import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents_langchain")
TOP_K = int(os.getenv("TOP_K", "4"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
PREINDEXED_PDF_PATH = os.getenv(
    "PREINDEXED_PDF_PATH",
    str(BASE_DIR / "data" / "sample_company_rag_test.pdf"),
)
