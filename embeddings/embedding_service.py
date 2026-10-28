import argparse

import config
from langchain_ollama import OllamaEmbeddings
from ingestion.chunking import chunk_documents
from ingestion.pdf_loader import load_pdf


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the first PDF chunk's embedding.")
    parser.add_argument("pdf_path", nargs="?", default=config.PREINDEXED_PDF_PATH)
    args = parser.parse_args()

    chunks = chunk_documents(load_pdf(args.pdf_path))
    if not chunks:
        print("No text chunks found in the PDF.")
        return

    embedding = get_embeddings().embed_documents([chunks[0].page_content])[0]
    print(f"Embedding for first chunk (page {chunks[0].metadata['page']}):")
    print(f"Vector length: {len(embedding)}")
    print(embedding)


if __name__ == "__main__":
    main()