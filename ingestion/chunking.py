import argparse

import config
from ingestion.pdf_loader import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size: int = 900, overlap: int = 150):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the first chunk from a PDF.")
    parser.add_argument("pdf_path", nargs="?", default=config.PREINDEXED_PDF_PATH)
    args = parser.parse_args()

    chunks = chunk_documents(load_pdf(args.pdf_path))
    if not chunks:
        print("No text chunks found in the PDF.")
        return

    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk (page {chunks[0].metadata['page']}):\n")
    print(chunks[0].page_content)


if __name__ == "__main__":
    main()
