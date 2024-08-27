import logging
import os
import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Constants
DATA_DIR = "app/data/"
MODEL_NAME = "llama3.1"
QDRANT_URL = "http://qdrant:6333"
OLLAMA_URL = "http://ollama-container:11434"
QDRANT_COLLECTION_NAME = "document_collection"

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("pikepdf._core").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Set up caching
set_llm_cache(InMemoryCache())


def load_and_split_documents(directory):
    try:
        loader = DirectoryLoader(directory, glob="**/*")
        documents = loader.load()
        logger.info("Documet read")
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda x: len(x) - 1
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Error loading or splitting documents: {e}")
        raise


def embed_documents_batch(embeddings, docs, batch_size=45):
    all_embeddings = []
    logger.info("Batch embeddings begun")
    response = requests.get(OLLAMA_URL)
    logger.info(f"{response}")
    for i in range(0, len(docs), batch_size):
        logger.info(f"BATCHING : {i}")
        batch = docs[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
        all_embeddings.extend(batch_embeddings)
    logger.info("Batch embeddings done")
    return all_embeddings


def main():
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_URL)
        logger.info(f"Got embeddings {embeddings}")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return
    try:
        logger.info(os.getcwd())
        logger.info(os.listdir(os.getcwd()))
        logger.info(os.listdir("app/data/"))
        docs = load_and_split_documents(DATA_DIR)
        logger.info("Data loaded and splitted")
    except Exception as e:
        logger.error(f"Error loading and splitting documents: {e}")
        return

    # Create vector store
    try:
        response = requests.get(QDRANT_URL)
        logger.info(f"{response}")
        embedded_docs = embed_documents_batch(embeddings, docs)
        logger.info("Calling Qdrant")
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION_NAME,
            vector_name="document_vector",
            retrieval_mode=RetrievalMode.DENSE,
        )
        logger.info(f"Documents successfully embedded and stored in Qdrant.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")


if __name__ == "__main__":
    main()
