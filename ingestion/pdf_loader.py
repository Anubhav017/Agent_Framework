import argparse
from pathlib import Path

import config
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(path: str):
    """Load PDF pages as LangChain Documents with source and one-based page metadata."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(path)

    source = str(file.resolve())
    documents = PyMuPDFLoader(source).load()
    pages = []
    for document in documents:
        document.page_content = document.page_content.strip()
        if document.page_content:
            document.metadata["source"] = source
            document.metadata["page"] = int(document.metadata.get("page", 0)) + 1
            pages.append(document)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Print extracted PDF text page by page.")
    parser.add_argument("pdf_path", nargs="?", default=config.PREINDEXED_PDF_PATH)
    args = parser.parse_args()

    pages = load_pdf(args.pdf_path)
    for page in pages:
        print(f"\n--- Page {page.metadata['page']} ---")
        print(page.page_content)


if __name__ == "__main__":
    main()
