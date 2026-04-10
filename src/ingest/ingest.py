from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import os
from sklearn.manifold import TSNE
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from src.utils.logger import get_logger
from src.utils.custom_exception import CustomException
load_dotenv(override=True)
DB_NAME = str(Path(__file__).parent.parent.parent / "vector_db")
CHROMA_COLLECTION_NAME = "mortgageexpert"
KNOWLEDGE_BASE = str(Path(__file__).parent.parent.parent / "data" / "raw")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger = get_logger(__name__)

logger.info("Starting the ingestion process.")



def fetch_documents():
    """Fetches documents from the specified knowledge base directory."""
    try:
        logger.info(f"Fetching documents from {KNOWLEDGE_BASE}")
        folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
        documents = []
        for folder in folders:
            if os.path.isdir(folder):
                logger.info(f"Processing folder: {folder}")
                loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
                documents.extend(loader.load())
        logger.info(f"Fetched {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise CustomException("Failed to fetch documents", e)
    
def create_chunks(documents):
    """Splits documents into smaller chunks for better processing."""
    try:
        logger.info("Creating chunks from documents.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = text_splitter.create_documents([doc.page_content for doc in documents])
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error creating chunks: {e}")
        raise CustomException("Failed to create chunks", e)
def create_embeddings(chunks):
    """Creates embeddings for the given chunks using HuggingFaceEmbeddings."""
    try:
        logger.info("Creating embeddings for chunks.")
        if os.path.exists(DB_NAME):
            Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=DB_NAME,
                embedding_function=embeddings,
            ).delete_collection()
        
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=DB_NAME,
        )
        logger.info("Embeddings created successfully.")
        collection = vectorstore._collection
        count = collection.count()

        sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)
        logger.info(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise CustomException("Failed to create embeddings", e) 


def ingest_data():
    """Runs the full ingestion pipeline and returns the vector store."""
    documents = fetch_documents()
    logger.info("Completed fetching documents.")
    chunks = create_chunks(documents)
    logger.info(f"Completed creating chunks. {len(chunks)} chunks created.")
    vectorstore = create_embeddings(chunks)
    logger.info("Completed creating embeddings for chunks.")
    return vectorstore


if __name__ == "__main__":
    ingest_data()