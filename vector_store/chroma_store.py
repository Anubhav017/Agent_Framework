import config
from embeddings.embedding_service import get_embeddings
from langchain_chroma import Chroma


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_PATH,
        embedding_function=get_embeddings(),
    )
