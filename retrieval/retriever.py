from pathlib import Path

import config
from vector_store.chroma_store import get_vector_store


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    results = get_vector_store().similarity_search_with_score(
        question,
        k=top_k or config.TOP_K,
        filter={"source": str(Path(config.PREINDEXED_PDF_PATH).resolve())},
    )
    return [
        {
            "text": document.page_content,
            **document.metadata,
            "distance": float(distance),
        }
        for document, distance in results
    ]
