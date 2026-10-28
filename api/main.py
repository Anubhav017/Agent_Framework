from fastapi import FastAPI, HTTPException

from api.schema import PDFRequest, QuestionRequest, QuestionResponse, SourceReference
from generation.llm_service import generate_answer
from retrieval.retriever import retrieve

app = FastAPI(title="LangChain PDF RAG API")


@app.post("/ingest/pdf")
def ingest_pdf(payload: PDFRequest):
    raise HTTPException(
        status_code=405,
        detail="PDF indexing is an offline operation. Run `python -m ingestion.index_pdf` before starting the API.",
    )


@app.post("/ask", response_model=QuestionResponse)
def ask(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    chunks = retrieve(payload.question, payload.top_k)
    answer = generate_answer(payload.question, chunks)
    return QuestionResponse(
        answer=answer,
        sources=[
            SourceReference(
                citation_id=index,
                source=chunk["source"],
                page=chunk["page"],
                distance=chunk["distance"],
            )
            for index, chunk in enumerate(chunks, 1)
        ],
    )

