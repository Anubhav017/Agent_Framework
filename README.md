# Local PDF RAG App with LangChain

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # On Windows: copy .env.example .env
```

## How it works

FastAPI is only the HTTP/API layer. The LangChain components are organized by responsibility:

- `ingestion/`: `PyMuPDFLoader` and `RecursiveCharacterTextSplitter` load and split the PDF.
- `embeddings/`: `OllamaEmbeddings` creates document and question vectors.
- `vector_store/`: LangChain `Chroma` provides persistent storage.
- `retrieval/`: searches the PDF chunks using LangChain Chroma.
- `generation/`: LangChain prompt/model/output-parser chain creates and validates answers.

There are no custom `Retriever`, `LLMService`, or `ChromaStore` classes; these folders expose small functions that use LangChain components.

## Run API

Index the PDF before starting the API:

```bash
python -m ingestion.index_pdf
```

To use a different PDF, set `PREINDEXED_PDF_PATH` in `.env` before indexing and starting the API. Both indexing and question retrieval use this same PDF path.

The command uses the local Ollama embedding model (`nomic-embed-text`), so Ollama must be running and that model must be available. Re-running the command skips chunks already in the vector store.

LangChain uses a separate `rag_documents_langchain` collection so the existing custom Chroma collection is left untouched. Run the indexing command once after upgrading, then start the API.

Then start the API:

```bash
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs and try:

- `POST /ask`: `{"question":"What does the policy say?","top_k":4}`

To change the default PDF, set `PREINDEXED_PDF_PATH` in `.env`. Document ingestion is disabled through the runtime API; `POST /ingest/pdf` returns `405`. Runtime API usage is limited to asking questions.

The model must provide a valid source citation internally; missing or out-of-range citations produce the grounded fallback response. Citation labels are removed from the `answer` text, while the retrieved source references are returned in `sources`.

## Notes

- PDF text extraction works for text-based PDFs; scanned PDFs need OCR.
- Embedding and generation use local Ollama models through LangChain integrations.
- This is a **local development starter**, not a production-ready public API. Add authentication, rate limiting, upload limits, tenant isolation, access control and stronger citation validation before deployment.
- Retrieved excerpts are untrusted; prompt-injection mitigation is not foolproof.
