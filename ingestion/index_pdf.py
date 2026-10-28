import argparse
import hashlib
from pathlib import Path

import config
from ingestion.chunking import chunk_documents
from ingestion.pdf_loader import load_pdf
from vector_store.chroma_store import get_vector_store


def index_pdf() -> dict:
    path = str(Path(config.PREINDEXED_PDF_PATH).resolve())
    chunks = chunk_documents(load_pdf(path))
    vector_store = get_vector_store()
    if not chunks:
        return {"chunks_indexed": 0, "chunks_skipped": 0, "path": path}

    existing_ids = set(vector_store.get(where={"source": path})["ids"])
    chunk_ids = [
        hashlib.sha256(
            f'{chunk.metadata["source"]}|{chunk.metadata["page"]}|{chunk.page_content}'.encode()
        ).hexdigest()
        for chunk in chunks
    ]
    new_chunks = []
    new_ids = []
    seen_ids = set(existing_ids)
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in seen_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)
            seen_ids.add(chunk_id)

    if not new_chunks:
        return {"chunks_indexed": 0, "chunks_skipped": len(chunks), "path": path}

    vector_store.add_documents(documents=new_chunks, ids=new_ids)
    return {
        "chunks_indexed": len(new_chunks),
        "chunks_skipped": len(chunks) - len(new_chunks),
        "path": path,
    }


def main() -> None:
    argparse.ArgumentParser(description="Index the configured PDF into the local vector store.").parse_args()
    result = index_pdf()
    print(
        f'Indexed {result["chunks_indexed"]} new chunks; '
        f'skipped {result["chunks_skipped"]} existing chunks from {result["path"]}.'
    )


if __name__ == "__main__":
    main()
