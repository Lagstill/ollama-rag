import logging
from functools import lru_cache
import requests
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_qdrant import RetrievalMode
import os
myhost = os.uname()[1]

MODEL_NAME = "llama3.1"
QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION_NAME = "document_collection"
OLLAMA_URL = "http://ollama-container:11434"

logging.basicConfig(level=logging.INFO)
logging.getLogger("pikepdf._core").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.info(f"MY SREAMLIT HOST: {myhost}")
set_llm_cache(InMemoryCache())

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

@lru_cache(maxsize=100)
def get_cached_response(query):
    return generate_response(query)

def generate_response(query):
    try:
        docs = vectorstore.similarity_search(query, k=4)
        logger.info("Doc susseccful")
        context = "\n\n".join([doc.page_content for doc in docs])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("qa_chain also successful")
        result = qa_chain.invoke({"query": query})
        logger.info("result successful")
        res = result.get("result", None)
        if res:
            logger.info(f"\n\n{res}\n\n")
        else:
            logger.info("No result found")
        return res
    except Exception as e:
        logger.error(f"Error during querying or response generation: {e}")
        return "I'm sorry, I encountered an error while processing your question."

def initialize_components():
    global llm, vectorstore

    try:
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_URL)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return

    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_URL)
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return

    try:
        response = requests.get(QDRANT_URL)
        logger.info(f"{response}")
        qdrant_client = QdrantClient(url=QDRANT_URL)
        try:
            embeddings.embed_documents(
                ["This is a content of the document", "This is another document"]
            )
        except Exception as e:
            logger.error(f"Error connecting to OLLAMA EMBEDDINGS: {e}")
            return  
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
            vector_name="document_vector",
            retrieval_mode=RetrievalMode.DENSE,
        )
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        return

def main():
    response = requests.get(OLLAMA_URL)
    logger.info(f"OLLAMA URL: {response}")

    initialize_components()

    st.title("Docker Only App")

    query = st.text_input("Enter your question:")

    if query:
        response = get_cached_response(query)
        st.write(response)

if __name__ == "__main__":
    main()
